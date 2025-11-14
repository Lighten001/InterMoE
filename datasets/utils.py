import torch


def lengths_to_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    mask = torch.arange(max_length, device=lengths.device).expand(
        len(lengths), max_length
    ) < lengths.unsqueeze(1)
    return mask
