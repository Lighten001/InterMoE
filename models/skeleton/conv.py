import torch
import torch.nn as nn
from torch.nn import functional as F


def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_norm(name, dim):
    if name.lower() == "layer":
        return nn.LayerNorm(dim)
    elif name.lower() == "batch":
        return nn.BatchNorm1d(dim)
    elif name.lower() == "group":
        return nn.GroupNorm(32, dim)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization function: {name}")


class GraphConv(nn.Module):
    """
    Graph Convolution.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.linear2 = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, adj_matrix):
        """
        x: [B, T, J, D]
        adj_matrix: [J, J]

        return: linear1(x) + sum_{j \in N(i)} linear2(x_j)
        """

        h1 = self.linear1(x)  # [B, T, J, D]
        h2 = torch.matmul(adj_matrix, self.linear2(x)) / (
            adj_matrix.sum(dim=-1, keepdim=True) + 1e-6
        )  # [B, T, J, D]

        out = h1 + h2

        return out


class CausalConv2d(nn.Conv2d):
    """
    Causal 2d convolusion.
    B D T J
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[2] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[2] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class STConv(nn.Module):
    """
    Skeleto-Temporal Convolution.
    """

    def __init__(
        self,
        edges,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
    ):
        assert (
            kernel_size % 2 == 1
        ), f"kernel_size should be odd number, but got {kernel_size}."

        super(STConv, self).__init__()

        # grpah convolution
        self.graph_conv = GraphConv(in_channels, out_channels, bias=bias)
        self.adj_matrix = nn.Parameter(
            self._get_adj_matrix(edges), requires_grad=False
        )  # symmetric matrix

        # temporal convolution
        self.temp_conv = CausalConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )

    def _get_adj_matrix(self, edges, add_self_loop=True):
        max_idx = -1
        for i, j in edges:
            max_idx = max(max_idx, i, j)

        adj_matrix = torch.zeros(max_idx + 1, max_idx + 1)
        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        if add_self_loop:
            for i in range(max_idx + 1):
                adj_matrix[i, i] = 1

        return adj_matrix

    def forward(self, x, cache_x=None):
        B, T, J, D = x.size()

        # graph conv
        graph_out = self.graph_conv.forward(x, self.adj_matrix)

        if cache_x is not None:
            # cache_x: [B, T, J, D]
            cache_x = cache_x.permute(0, 3, 1, 2)

        # temporal conv
        # temp_in = x.permute(0, 2, 3, 1).reshape(B * J, D, T)
        # temp_out = self.temp_conv(temp_in)
        # temp_out = temp_out.reshape(B, J, -1, T).permute(0, 3, 1, 2)
        temp_in = x.permute(0, 3, 1, 2)  # B D T J
        temp_out = self.temp_conv(temp_in, cache_x)
        temp_out = temp_out.permute(0, 2, 3, 1)  # B T J D

        out = graph_out + temp_out

        return out


class ResSTConv(nn.Module):
    """
    Residual Skeleto-Temporal Convolution.
    """

    def __init__(
        self,
        edges,
        dim_channels,
        kernel_size,
        bias=True,
        activation="gelu",
        norm="none",
        dropout=0.1,
    ):
        super(ResSTConv, self).__init__()
        self.layers = nn.Sequential(
            get_norm(norm, dim_channels),
            get_activation(activation),
            STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias),
            get_norm(norm, dim_channels),
            get_activation(activation),
            STConv(edges, dim_channels, dim_channels, kernel_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = x.clone()
        CACHE_T = 2
        for layer in self.layers:
            if isinstance(layer, STConv) and feat_cache is not None:
                # import ipdb; ipdb.set_trace()
                idx = feat_idx[0]
                cache_x = x[:, -CACHE_T:, :, :].clone()
                if cache_x.shape[1] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, -1, :, :]
                            .unsqueeze(1)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=1,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return h + x
