import torch
from torch import nn
from torch.nn import functional as F
import clip

from models.nets import LatentInterDiffusion
from models.utils import set_requires_grad, load_from_ckpt
from models.vae import CasualSTVAE


class InterMoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM

        vae = CasualSTVAE(cfg.vae)
        if torch.__version__ >= "2.0":
            ckpt = torch.load(cfg.vae_ckpt, map_location="cpu", weights_only=False)
        else:
            ckpt = torch.load(cfg.vae_ckpt, map_location="cpu")
        vae = load_from_ckpt(vae, ckpt)
        vae.freeze()
        self.vae = vae

        print(f"Loading VAE Model {cfg.vae_ckpt}")

        self.denoiser = LatentInterDiffusion(cfg, sampling_strategy=cfg.STRATEGY)

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer, num_layers=2
        )
        self.clip_ln = nn.LayerNorm(768)

    def compute_loss(self, batch):
        # import ipdb; ipdb.set_trace()
        batch = self.text_process(batch)
        batch = self.vae_encode(batch)
        losses = self.denoiser.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.denoiser(batch))
        batch.update(self.vae_decode(batch))
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():

            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(
                self.dtype
            )  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)
            x = pe_tokens.permute(1, 0, 2)  # NLD -> LND
            x = self.clip_transformer(x)
            x = x.permute(1, 0, 2)
            clip_out = self.ln_final(x).type(self.dtype)

        out = self.clipTransEncoder(clip_out)
        out = self.clip_ln(out)

        cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        batch["cond"] = cond

        return batch

    def lengths_to_mask(self, lengths: torch.Tensor) -> torch.Tensor:
        max_frames = torch.max(lengths)
        mask = torch.arange(max_frames, device=lengths.device).expand(
            len(lengths), max_frames
        ) < lengths.unsqueeze(1)
        return mask

    def vae_decode(self, batch):
        # decode
        # import ipdb; ipdb.set_trace()
        output: torch.Tensor = batch["output"]  # B T D
        device: torch.device = output.device

        motion_lens: torch.Tensor = batch["motion_lens"].to(device)  # B
        latents1, latents2 = output.chunk(2, dim=-1)

        B, T = output.shape[:2]

        latents1 = latents1.reshape(B, T, -1, self.vae.latent_dim)
        latents2 = latents2.reshape(B, T, -1, self.vae.latent_dim)

        pred_motion1 = self.vae.decode(latents1)
        pred_motion2 = self.vae.decode(latents2)
        if isinstance(pred_motion1, tuple) or isinstance(pred_motion1, list):
            pred_motion1 = pred_motion1[0]
            pred_motion2 = pred_motion2[0]

        batch["output"] = torch.cat([pred_motion1, pred_motion2], dim=-1)

        batch["generate_lens"] = batch["denoiser_lens"] * 4
        return batch

    def vae_encode(self, batch):
        # import ipdb; ipdb.set_trace()
        device = next(self.clip_transformer.parameters()).device
        raw_text = batch["text"]

        motions = batch["normed_motions"]
        motion_lens = batch["motion_lens"]

        with torch.no_grad():
            motion1, motion2 = torch.chunk(motions, chunks=2, dim=-1)
            lm1, _ = self.vae.encode(motion1)  # B T J C
            lm2, _ = self.vae.encode(motion2)

            len_mask = self.lengths_to_mask(motion_lens // 4)  # [B, T]
            len_mask = F.pad(
                len_mask,
                (0, lm1.shape[1] - len_mask.shape[1]),
                mode="constant",
                value=False,
            )

            if len(lm1.shape) == 4:
                lm1 = lm1 * len_mask[..., None, None].float()
                lm2 = lm2 * len_mask[..., None, None].float()
            else:
                lm1 = lm1 * len_mask[..., None].float()
                lm2 = lm2 * len_mask[..., None].float()

        B, T = lm1.shape[:2]
        # import ipdb; ipdb.set_trace()

        lm1 = lm1.reshape(B, T, -1)
        lm2 = lm2.reshape(B, T, -1)

        batch.update({"latent_motions": torch.cat([lm1, lm2], dim=-1)})

        return batch
