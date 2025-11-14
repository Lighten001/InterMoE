import torch
import torch.nn as nn
import math


class MultiLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_experts,
        bias: bool = True,
    ):
        super(MultiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        x: [*, num_experts, in_features]
        out: [*, num_experts, out_features]
        """
        out = torch.matmul(x.unsqueeze(-2), self.weight).squeeze(-2)
        if self.bias is not None:
            out += self.bias
        return out
