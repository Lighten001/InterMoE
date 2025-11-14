import torch
import torch.nn as nn


def adj_list_to_edges(adj_list):
    edges = []
    for i, adj in enumerate(adj_list):
        for j in adj:
            if i < j:
                edges.append((i, j))
    return edges

def edges_to_adj_list(edges):
    max_idx = -1
    for i, j in edges:
        max_idx = max(max_idx, i, j)

    adj_list = [[] for _ in range(max_idx + 1)]
    for i, j in edges:
        adj_list[i].append(j)
        adj_list[j].append(i)

    return adj_list


class STPool(nn.Module):
    """
    Skeleto-Temporal Pooling.
    """

    def __init__(
        self,
        dataset="t2m",
        depth=0,
    ):
        if not dataset in ["t2m", "kit", "interhuman", "interx"]:
            raise ValueError("dataset should be 't2m' or 'kit'")

        super(STPool, self).__init__()

        self.skeleton_pool, self.skeleton_mapping, self.new_edges = (
            self._get_skeleton_pooling(dataset, depth)
        )
        self.skeleton_pool = nn.Parameter(
            self.skeleton_pool, requires_grad=False
        )  # [J_out, J_in]
        self.temporal_pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def _get_skeleton_pooling(self, dataset, depth):
        if depth == 0:
            if dataset == "t2m" or dataset == "interhuman":
                weight = torch.zeros(12, 22)
                mapping = [
                    [(0, 1, 2, 3), 0],  # root
                    [(0, 1, 4), 1],  # left hip
                    [(4, 7, 10), 2],  # left leg
                    [(0, 2, 5), 3],  # right hip
                    [(5, 8, 11), 4],  # right leg
                    [(0, 3, 6, 9), 5],  # spine
                    [(9, 6, 12, 13, 14), 6],  # chest
                    [(9, 13, 16), 7],  # left shoulder
                    [(16, 18, 20), 8],  # left arm
                    [(9, 14, 17), 9],  # right shoulder
                    [(17, 19, 21), 10],  # right arm
                    [(9, 12, 15), 11],  # head
                ]
            elif dataset == "interx":  # smplx
                weight = torch.zeros(14, 55)
                mapping = [
                    [(0, 1, 2, 3), 0],  # root
                    [(0, 1, 4), 1],  # left hip
                    [(4, 7, 10), 2],  # left leg
                    [(0, 2, 5), 3],  # right hip
                    [(5, 8, 11), 4],  # right leg
                    [(0, 3, 6, 9), 5],  # spine
                    [(9, 6, 12, 13, 14), 6],  # chest
                    [(9, 13, 16), 7],  # left shoulder
                    [(16, 18, 20), 8],  # left arm
                    [(9, 14, 17), 9],  # right shoulder
                    [(17, 19, 21), 10],  # right arm
                    [(9, 12, 15, 22, 23, 24), 11],  # head
                    [tuple(range(25, 40)), 12],  # left hand
                    [tuple(range(40, 55)), 13],  # right hand
                ]
            else:
                weight = torch.zeros(12, 21)
                mapping = [
                    [(0, 1, 11, 16), 0],  # root
                    [(0, 16, 17, 18), 1],  # left hip
                    [(17, 18, 19, 20), 2],  # left leg
                    [(0, 11, 12, 13), 3],  # right hip
                    [(12, 13, 14, 15), 4],  # right leg
                    [(0, 1, 2, 3), 5],  # spine
                    [(2, 3, 5, 8, 4), 6],  # chest
                    [(3, 8, 9), 7],  # left shoulder
                    [(8, 9, 10), 8],  # left arm
                    [(3, 5, 6), 9],  # right shoulder
                    [(5, 6, 7), 10],  # right arm
                    [(3, 4), 11],  # head
                ]

            if dataset == "interx":  # smplx
                new_edges = adj_list_to_edges(
                    [
                        [1, 3, 5],
                        [0, 2],
                        [1],
                        [0, 4],
                        [3],
                        [0, 6],
                        [5, 7, 9, 11],
                        [6, 8, 12],
                        [7],
                        [6, 10, 13],
                        [9],
                        [6],
                        [12],
                        [13],
                    ]
                )
            else:
                new_edges = adj_list_to_edges(
                    [
                        [1, 3, 5],
                        [0, 2],
                        [1],
                        [0, 4],
                        [3],
                        [0, 6],
                        [5, 7, 9, 11],
                        [6, 8],
                        [7],
                        [6, 10],
                        [9],
                        [6],
                    ]
                )

        elif depth == 1:
            if dataset == "interx":  # smplx
                weight = torch.zeros(7, 14)
                mapping = [
                    [(0, 1, 3, 5), 0],  # root
                    [(0, 1, 2), 1],  # left lower
                    [(0, 3, 4), 2],  # right lower
                    [(0, 5, 6), 3],  # spine
                    [(6, 7, 8, 12), 4],  # left upper
                    [(6, 9, 10, 13), 5],  # right upper
                    [(6, 11), 6],  # head
                ]
            else:
                weight = torch.zeros(7, 12)
                mapping = [
                    [(0, 1, 3, 5), 0],  # root
                    [(0, 1, 2), 1],  # left lower
                    [(0, 3, 4), 2],  # right lower
                    [(0, 5, 6), 3],  # spine
                    [(6, 7, 8), 4],  # left upper
                    [(6, 9, 10), 5],  # right upper
                    [(6, 11), 6],  # head
                ]
            new_edges = adj_list_to_edges(
                [
                    [1, 2, 3],
                    [0],
                    [0],
                    [0, 4, 5, 6],
                    [3],
                    [3],
                    [3],
                ]
            )

        else:
            weight = torch.ones(7, 7)
            mapping = [
                [(0, 1, 2, 3), 0],  # root
                [(0, 1), 1],  # left lower
                [(0, 2), 2],  # right lower
                [(0, 3), 3],  # spine
                [(3, 4), 4],  # left upper
                [(3, 5), 5],  # right upper
                [(3, 6), 6],  # head
            ]
            new_edges = adj_list_to_edges(
                [
                    [1, 2, 3],
                    [0],
                    [0],
                    [0, 4, 5, 6],
                    [3],
                    [3],
                    [3],
                ]
            )

        for joints, idx in mapping:
            weight[idx, joints] = 1
        weight = weight / weight.sum(dim=1, keepdim=True)

        return weight, mapping, new_edges

    def forward(self, x):
        """
        x: [B, T, J, D]
        out: [B, T // 2, J_out, D]
        """
        B, T, J_in, D = x.size()

        # skeleton pooling
        out = torch.matmul(self.skeleton_pool, x)  # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal pooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_pool(out)
        out = out.reshape(B, J_out, D, -1).permute(0, 3, 1, 2)  # [B, T // 2, J_out, D]

        return out


class STUnpool(nn.Module):
    """
    Skeleton-Temporal Unpooling.
    """

    def __init__(
        self,
        skeleton_mapping,
    ):
        super(STUnpool, self).__init__()
        self.skeleton_unpool = nn.Parameter(
            self._get_skeleton_unpool(skeleton_mapping), requires_grad=False
        )  # [J_out, J_in]
        self.temporal_unpool = nn.Upsample(scale_factor=2, mode="linear")

    def _get_skeleton_unpool(self, skeleton_mapping):
        max_idx = -1
        for joints, idx in skeleton_mapping:
            max_idx = max(max_idx, *joints)

        weight = torch.zeros(max_idx + 1, len(skeleton_mapping))
        for joints, idx in skeleton_mapping:
            weight[joints, idx] = 1

        return weight

    def forward(self, x):
        """
        x: [B, T, J_in, D]
        out: [B, T * upsample_rate, J_in, D]
        """

        B, T, J_in, D = x.size()

        # skeleton unpooling
        out = torch.matmul(self.skeleton_unpool, x)  # [B, T, J_out, D]
        J_out = out.size(2)

        # temporal unpooling
        out = out.permute(0, 2, 3, 1).reshape(B * J_out, D, T)
        out = self.temporal_unpool(out)
        out = out.reshape(B, J_out, D, -1).permute(
            0, 3, 1, 2
        )  # [B, T * upsample_rate, J_out, D]

        return out
