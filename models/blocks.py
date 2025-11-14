import torch
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F

from .layers import VanillaSelfAttention, VanillaCrossAttention, FFN, MoeFFN, AdaLN


class TransformerBlock(nn.Module):
    def __init__(
        self,
        latent_dim=512,
        num_heads=8,
        ff_size=1024,
        dropout=0.0,
        cond_abl=False,
        use_moe=False,
        MoE_config=None,
        **kargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl
        self.use_moe = use_moe

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(
            latent_dim, latent_dim, num_heads, dropout, latent_dim
        )

        self.ffn = DualDyDiffMoEBlock(
            experts=[
                MoeFFN(
                    latent_dim=latent_dim,
                    intermediate_size=ff_size,
                )
                for _ in range(MoE_config.num_experts)
            ],
            hidden_dim=latent_dim,
            num_experts=MoE_config.num_experts,
            capacity=MoE_config.capacity,
            n_shared_experts=MoE_config.n_shared_experts,
        )

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        if isinstance(out, tuple):
            out, ones, capacity_pred = out
        else:
            ones, capacity_pred = None, None
        out = out + h2
        return out, ones, capacity_pred


class DualDyDiffMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        experts,
        hidden_dim,
        num_experts,
        embed_dim=None,
        n_shared_experts=0,
        capacity=2,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm = AdaLN(hidden_dim, embed_dim)

        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        self.text_gate_weight = nn.Parameter(
            torch.empty((num_experts, embed_dim or hidden_dim))
        )
        nn.init.normal_(self.gate_weight, std=0.006)
        nn.init.normal_(self.text_gate_weight, std=0.006)

        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.use_aux_bias = True
        if self.use_aux_bias:
            self.aux_bias = nn.Parameter(
                torch.ones((num_experts, 1)) * -0.5, requires_grad=False
            )
            self.his_moment = torch.zeros((num_experts, 1))

        self.n_shared_experts = n_shared_experts

        if self.n_shared_experts > 0:
            # NOTE:
            print("self.n_shared_experts", self.n_shared_experts)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio * 2)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = FFN(
                latent_dim=hidden_dim,
                ffn_dim=mlp_hidden_dim,
                act_layer=approx_gelu,
                dropout=0,
            )

        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0] * num_experts)
        self.register_buffer("expert_threshold", expert_threshold)
        ema_decay = torch.tensor([ema_decay])
        self.register_buffer("ema_decay", ema_decay)

    def forward(self, x, emb=None):

        return self.forward_train_and_eval(x, emb)

    def update_aux_bias(self, alpha=0.001):
        if self.training:
            self.aux_bias -= alpha * self.his_moment.to(self.aux_bias.device)
        return

    @torch.no_grad()
    def update_his_moment(self, mask, k):
        """
        mask: num_experts, S
        """
        if self.training:
            # import ipdb; ipdb.set_trace()
            act_e = mask.detach().sum(dim=-1, keepdim=True)
            moment = torch.sign(act_e - k)
            self.his_moment = moment
        return None

    def forward_train_and_eval(self, x, emb):
        # import ipdb; ipdb.set_trace()
        # update bias (only for training)
        self.update_aux_bias()

        B, s, D = x.shape
        identity = x

        # Flatten the input for processing
        x = x.reshape(-1, D)  # (S, D), where S = B * s
        S = x.shape[0]

        k = int((S / self.num_experts) * self.capacity)

        # Compute gating logits and scores
        logits_m = F.linear(x, self.gate_weight, None)  # (S, num_experts)
        logits_t = F.linear(emb, self.text_gate_weight, None)  # (B, num_experts)

        logits = logits_m + logits_t.unsqueeze(1).expand(-1, s, -1).reshape(
            S, -1
        )  # (S, num_experts)

        scores = logits.softmax(dim=-1).permute(1, 0)  # (num_experts, S)

        scores_bias = logits.sigmoid().permute(1, 0) + self.aux_bias  # (num_experts, S)

        # Apply normalization if emb is provided
        x = self.norm(x.reshape(B, s, D), emb).reshape(-1, D)

        # Get top-k gating values and indices for each expert
        mask = torch.where(scores_bias > 0, 1.0, 0)  # num_experts, S
        index = torch.argwhere(mask.reshape(-1) > 0)  # K, 1

        # Expand gating weights for broadcasting
        gating_expanded = (scores[mask > 0]).unsqueeze(-1)  # (K, 1)

        # # Gather inputs for each expert
        expert_inputs = x  # (num_experts, S, D)
        expert_outputs = torch.cat(
            [
                expert(expert_inputs[mask[i] > 0])
                for i, expert in enumerate(self.experts)
            ],
            dim=0,
        )  # (K, D)
        # Apply gating to expert outputs
        gated_outputs = gating_expanded * expert_outputs  # (K, D)

        y = torch.zeros((S * self.num_experts, D), dtype=x.dtype, device=x.device)
        y = torch.scatter(
            y,  # Target tensor of shape [S * num_experts, D]
            0,  # Dimension to scatter along
            index.expand(-1, D),  # Indices of shape [K, D]
            gated_outputs,  # Source values of shape [K, D]
        )

        # Sum the outputs from all experts
        y = y.reshape(self.num_experts, S, D).sum(dim=0, keepdim=False)  # (S, 1, D)

        # only for training, update his moment
        self.update_his_moment(mask=mask, k=k)

        # Reshape the output to match the input shape
        x_out = y.reshape(B, s, D)

        # Add shared expert outputs if applicable
        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out
