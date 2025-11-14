import torch
import torch.nn as nn
from models.skeleton.linear import MultiLinear
from models.vae.encdec import (
    MotionEncoder,
    MotionDecoder,
    STConvEncoder,
    STConvDecoder,
)
from datasets.interhuman import MotionNormalizerTorch

from models.skeleton.conv import STConv


def count_conv2d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, STConv):
            count += 1
    return count


class CasualSTVAE(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self._stage = "vae"

        self.joints_num = opt.joints_num
        self.latent_dim = opt.latent_dim
        self.dataset_name = opt.dataset_name

        # motion encoder and decoder
        self.motion_enc = MotionEncoder(opt)
        self.motion_dec = MotionDecoder(opt)

        # skeleto-temporal convolutional encoder and decoder
        self.conv_enc = STConvEncoder(opt)
        self.conv_dec = STConvDecoder(opt, self.conv_enc)

        self.dist = MultiLinear(opt.latent_dim, opt.latent_dim * 2, 7)

        if self.dataset_name == "interhuman":
            self._normalizer = MotionNormalizerTorch()
        else:
            self._normalizer = nn.Identity()

    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def encode(self, x):
        _enc_conv_num = count_conv2d(self.conv_enc)
        _enc_conv_idx = [0]
        _enc_feat_map = [None] * _enc_conv_num

        T = x.shape[1]  # B T J C
        x = self.motion_enc(x)
        iter_ = (T) // 4
        for i in range(iter_):
            _enc_conv_idx = [0]
            if i == 0:
                out = self.conv_enc(
                    x[:, 4 * (i) : 4 * (i + 1), :, :],
                    feat_cache=_enc_feat_map,
                    feat_idx=_enc_conv_idx,
                )
            else:
                out_ = self.conv_enc(
                    x[:, 4 * (i) : 4 * (i + 1), :, :],
                    feat_cache=_enc_feat_map,
                    feat_idx=_enc_conv_idx,
                )
                out = torch.cat([out, out_], 1)
        x = out

        # latent space
        x = self.dist(x)
        mu, logvar = x.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        loss_kl = 0.5 * torch.mean(torch.pow(mu, 2) + torch.exp(logvar) - logvar - 1.0)

        return z, {"loss_kl": loss_kl}

    def decode(self, x):
        _dec_conv_num = count_conv2d(self.conv_dec)
        _dec_conv_idx = [0]
        _dec_feat_map = [None] * _dec_conv_num

        T = x.shape[1]  # B T J C
        iter_ = T
        for i in range(iter_):
            _dec_conv_idx = [0]
            if i == 0:
                out = self.conv_dec(
                    x[:, i : i + 1, :, :],
                    feat_cache=_dec_feat_map,
                    feat_idx=_dec_conv_idx,
                )
            else:
                out_ = self.conv_dec(
                    x[:, i : i + 1, :, :],
                    feat_cache=_dec_feat_map,
                    feat_idx=_dec_conv_idx,
                )
                out = torch.cat([out, out_], 1)
        x = out
        x = self.motion_dec(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """

        # encode
        x = x.detach().float()
        z, loss_dict = self.encode(x)
        out = self.decode(z)

        return out, loss_dict

    # NOTE: only for evaluation vae
    def forward_test(self, batch):
        """
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """

        device = next(self.motion_enc.parameters()).device
        motions = batch["motions"].to(device)  # B T D
        B = motions.shape[0]
        assert B == 1, "now only support batch_size == 1"

        motion_lens = batch["motion_lens"]

        m1, m2 = motions.chunk(2, dim=-1)

        m1 = m1[:, : motion_lens[0], :]
        m2 = m2[:, : motion_lens[0], :]

        normed_m1 = self._normalizer.forward(m1)
        normed_m2 = self._normalizer.forward(m2)
        pred_m1, _ = self.forward(normed_m1)
        pred_m2, _ = self.forward(normed_m2)

        output = torch.cat([pred_m1, pred_m2], dim=-1)

        batch["output"] = output
        return batch

    def forward_test_batch(self, input_dict):
        """
        input_dict: {
            "motions": [B, T, D],
            "motion_lens": [B]
        }
        x: [B, T, D]
        z: [B, T, J_out, D]
        out: [B, T, D]
        """

        device = next(self.motion_enc.parameters()).device
        motions = input_dict["motions"].to(device)  # B T D
        B = motions.shape

        motion_lens = input_dict["motion_lens"]

        m1, m2 = motions.chunk(2, dim=-1)

        m1 = m1[:, : motion_lens[0], :]
        m2 = m2[:, : motion_lens[0], :]

        normed_m1 = self._normalizer.forward(m1)
        normed_m2 = self._normalizer.forward(m2)
        pred_m1, _ = self.forward(normed_m1)
        pred_m2, _ = self.forward(normed_m2)

        output = torch.cat([pred_m1, pred_m2], dim=-1)

        input_dict["output"] = output
        return input_dict
