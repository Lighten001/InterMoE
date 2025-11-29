import torch
from torch import nn

from models.utils import PositionalEncoding, TimestepEmbedder, zero_module
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import TransformerBlock
from models.layers import FinalLayer

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType,
)


class LatentInterDenoiser(nn.Module):
    def __init__(
        self,
        input_feats,
        latent_dim=512,
        num_frames=240,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        cfg_weight=0.0,
        use_moe=False,
        MoE_config=None,
        CapacityPred_loss_weight=0.01,
        use_motion_embed=True,
        **kargs
    ):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim
        self.use_moe = use_moe
        self.CapacityPred_loss_weight = CapacityPred_loss_weight

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(
            self.latent_dim, self.sequence_pos_encoder
        )

        # Input Embedding
        if use_motion_embed:
            self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        else:
            self.motion_embed = nn.Identity()
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    num_heads=num_heads,
                    latent_dim=latent_dim,
                    dropout=dropout,
                    ff_size=ff_size,
                    use_moe=use_moe,
                    MoE_config=MoE_config,
                )
            )
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))

    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        # import ipdb; ipdb.set_trace()
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[..., : self.input_feats], x[..., self.input_feats :]

        if mask is not None:
            mask = mask[..., 0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        a_ones_list = []
        a_pred_c_list = []

        b_ones_list = []
        b_pred_c_list = []
        layer_idx_list = []
        for i, block in enumerate(self.blocks):
            h_a, ones_a, pred_c_a = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            h_b, ones_b, pred_c_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b
            if ones_a is not None:
                a_ones_list.append(ones_a)
                a_pred_c_list.append(pred_c_a)
                b_ones_list.append(ones_b)
                b_pred_c_list.append(pred_c_b)
                layer_idx_list.append(i)

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)
        if not self.use_moe:
            return output
        else:
            return (
                output,
                "Capacity_Pred",
                a_ones_list,
                a_pred_c_list,
                b_ones_list,
                b_pred_c_list,
                layer_idx_list,
                self.CapacityPred_loss_weight,
            )


class LatentInterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = LatentInterDenoiser(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
            use_moe=cfg.use_moe,
            MoE_config=cfg.MoE_config,
            CapacityPred_loss_weight=cfg.CapacityPred_loss_weight,
            use_motion_embed=getattr(cfg, "use_motion_embed", True),
        )

        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing = [self.diffusion_steps]
        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob=0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * cond_mask_prob
            ).view(
                [bs] + [1] * len(cond.shape[1:])
            )  # 1-> use null_cond, 0-> use real cond
            return cond * (1.0 - mask), (1.0 - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):
        cond = batch["cond"]
        x_start = batch["latent_motions"]
        B, T = batch["latent_motions"].shape[:2]

        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)

        seq_mask = self.generate_src_mask(T, batch["motion_lens"] // 4).to(
            x_start.device
        )

        t, _ = self.sampler.sample(B, x_start.device)
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs={
                "mask": seq_mask,
                "cond": cond,
            },
        )
        return output

    def forward(self, batch):
        cond = batch["cond"]
        # x_start = batch["motions"]
        B = cond.shape[0]
        motion_lens = batch["motion_lens"] // 4  # NOTE //4 for vae
        T = motion_lens.max()

        batch["denoiser_lens"] = motion_lens

        seq_mask = self.generate_src_mask(T, batch["motion_lens"] // 4).to(cond.device)

        timestep_respacing = self.sampling_strategy
        self.diffusion_test = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats * 2),
            clip_denoised=False,
            progress=False if B > 1 else True,
            model_kwargs={
                "mask": seq_mask,
                "cond": cond,
            },
            x_start=None,
        )
        return {"output": output}

