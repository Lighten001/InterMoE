from os.path import join as pjoin
import torch
from torch import nn
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.utils import lengths_to_mask
from datasets.interhuman import MotionNormalizer, InterHumanDataset
from models.utils import set_requires_grad, PositionalEncoding
import clip
import copy
from tqdm import tqdm


class BatchEvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=64, num_workers=1, shuffle=True)
        mm_dataloader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}

                batch["text"] = list(text)
                batch["motion_lens"] = motion_lens + 3
                if getattr(self.model, "_stage", "") == "vae":
                    batch["motions"] = torch.cat([motion1, motion2], dim=-1)

                batch = self.model.forward_test_batch(batch)

                # import ipdb; ipdb.set_trace()

                motions_output = batch["output"].reshape(
                    batch["output"].shape[0], batch["output"].shape[1], 2, -1
                )
                # NOTE: important, same as InterMask only compare generationed motion.
                motions_output = self.normalizer.backward(
                    motions_output.cpu().detach().numpy()
                )
                generate_lens = batch["generate_lens"]
                mask = lengths_to_mask(motion_lens, motions_output.shape[1])
                motions_output = motions_output * mask[..., None, None].detach().numpy()


                B, T, _, D = motions_output.shape
                if T < self.max_length:
                    padding_len = self.max_length - T
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate(
                        (motions_output, padding_zeros), axis=1
                    )
                assert motions_output.shape[1] == self.max_length

                # import ipdb; ipdb.set_trace()
                for i in range(B):
                    sub_dict = {
                        "motion1": motions_output[i, :, 0],
                        "motion2": motions_output[i, :, 1],
                        "motion_lens": motion_lens[i],
                        "text": text[i],
                    }
                    generated_motions.append(sub_dict)

            for i, data in tqdm(enumerate(mm_dataloader)):
                if i not in mm_idxs:
                    continue
                name, text, motion1, motion2, motion_lens = data
                batch = {}

                batch["text"] = list(text) * mm_num_repeats
                batch["motion_lens"] = motion_lens + 3
                if getattr(self.model, "_stage", "") == "vae":
                    batch["motions"] = torch.cat([motion1, motion2], dim=-1)

                batch = self.model.forward_test(batch)

                # import ipdb; ipdb.set_trace()

                motions_output = batch["output"].reshape(
                    batch["output"].shape[0], batch["output"].shape[1], 2, -1
                )
                # NOTE: important, same as InterMask only compare generationed motion.
                motions_output = self.normalizer.backward(
                    motions_output.cpu().detach().numpy()
                )
                # generate_lens = batch["generate_lens"]
                generate_lens = motion_lens
                mask = lengths_to_mask(motion_lens, motions_output.shape[1])
                motions_output = motions_output * mask[..., None, None].detach().numpy()

                B, T, _, D = motions_output.shape
                if T < self.max_length:
                    padding_len = self.max_length - T
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate(
                        (motions_output, padding_zeros), axis=1
                    )
                assert motions_output.shape[1] == self.max_length
                # import ipdb; ipdb.set_trace()
                mm_sub_dict = {
                    "mm_motions": motions_output,
                    "motion_lens": generate_lens[0],
                    "text": text[0],
                }
                mm_generated_motions.append(mm_sub_dict)

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = (
            data["motion1"],
            data["motion2"],
            data["motion_lens"],
            data["text"],
        )
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data["mm_motions"]
        motion_lens = data["motion_lens"]
        mm_motions1 = mm_motions[:, :, 0]
        mm_motions2 = mm_motions[:, :, 1]
        text = data["text"]
        motion_lens = np.array([motion_lens] * mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == "interhuman":
        print("Loading dataset %s ..." % opt.NAME)

        dataset = InterHumanDataset(opt)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True
        )
    else:
        raise KeyError("Dataset not Recognized !!")

    print("Ground Truth Dataset Loading Completed!!!")
    return dataloader, dataset


def get_motion_loader(
    batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats
):
    # Currently the configurations of two datasets are almost the same
    dataset = BatchEvaluationDataset(
        model,
        ground_truth_dataset,
        device,
        mm_num_samples=mm_num_samples,
        mm_num_repeats=mm_num_repeats,
    )
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(
        dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True
    )
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print("Generated Dataset Loading Completed!!!")

    return motion_loader, mm_motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(
        pjoin("eval_model/interclip.ckpt"),
        map_location="cpu",
        weights_only=False,
    )
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint[
                "state_dict"
            ].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(
                self.device
            )
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            """Motion Encoding"""
            motion_embedding = self.model.encode_motion(batch)["motion_emb"]

            """Text Encoding"""
            text_embedding = self.model.encode_text(batch)["text_emb"][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(
                self.device
            )
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            """Motion Encoding"""
            motion_embedding = self.model.encode_motion(batch)["motion_emb"]

        return motion_embedding


class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats * 2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout, max_len=2000
        )

        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers
        )
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)

    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat(
            [
                self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][
                    :, None
                ],
                x_emb,
            ],
            dim=1,
        )

        seq_mask = mask > 0.5
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:, 0])

        batch["motion_emb"] = motion_emb

        return batch


loss_ce = torch.nn.CrossEntropyLoss()


class InterCLIP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.motion_encoder = MotionEncoder(cfg)

        self.latent_dim = self.latent_dim

        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.dtype = clip_model.dtype
        self.latent_scale = nn.Parameter(torch.Tensor([1]))

        set_requires_grad(self.token_embedding, False)

        textTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=cfg.FF_SIZE,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.textTransEncoder = nn.TransformerEncoder(
            textTransEncoderLayer, num_layers=8
        )
        self.text_ln = nn.LayerNorm(768)
        self.out = nn.Linear(768, 512)

        self.clip_training = "text_"
        self.l1_criterion = nn.L1Loss(reduction="mean")

    def compute_loss(self, batch):
        losses = {}
        losses["total"] = 0

        # compute clip losses
        batch = self.encode_text(batch)
        batch = self.encode_motion(batch)

        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        losses.update(clip_losses)
        losses["total"] += mixed_clip_loss

        return losses["total"], losses

    def forward(self, batch):
        return self.compute_loss(batch)

    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.0
        clip_losses = {}

        if 1:
            for d in self.clip_training.split("_")[:1]:
                if d == "image":
                    features = self.clip_model.encode_image(
                        batch["images"]
                    ).float()  # preprocess is done in dataloader
                elif d == "text":
                    features = batch["text_emb"]
                motion_features = batch["motion_emb"]
                # normalized features
                features_norm = features / features.norm(dim=-1, keepdim=True)
                motion_features_norm = motion_features / motion_features.norm(
                    dim=-1, keepdim=True
                )

                logit_scale = self.latent_scale**2
                logits_per_motion = (
                    logit_scale * motion_features_norm @ features_norm.t()
                )
                logits_per_d = logits_per_motion.t()

                batch_size = motion_features.shape[0]
                ground_truth = torch.arange(
                    batch_size, dtype=torch.long, device=motion_features.device
                )

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.0

                clip_losses[f"{d}_ce_from_d"] = ce_from_d_loss.item()
                clip_losses[f"{d}_ce_from_motion"] = ce_from_motion_loss.item()
                clip_losses[f"{d}_mixed_ce"] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss

        return mixed_clip_loss, clip_losses

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask

    def encode_motion(self, batch):
        batch["mask"] = self.generate_src_mask(
            batch["motions"].shape[1], batch["motion_lens"]
        ).to(batch["motions"].device)
        batch.update(self.motion_encoder(batch))
        batch["motion_emb"] = (
            batch["motion_emb"]
            / batch["motion_emb"].norm(dim=-1, keepdim=True)
            * self.latent_scale
        )

        return batch

    def encode_text(self, batch):
        device = next(self.parameters()).device
        raw_text = batch["text"]

        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.token_embedding(text).type(
                self.dtype
            )  # [batch_size, n_ctx, d_model]
            pe_tokens = x + self.positional_embedding.type(self.dtype)

        out = self.textTransEncoder(pe_tokens)
        out = self.text_ln(out)

        out = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        out = self.out(out)

        batch["text_emb"] = out
        batch["text_emb"] = (
            batch["text_emb"]
            / batch["text_emb"].norm(dim=-1, keepdim=True)
            * self.latent_scale
        )

        return batch
