import pyrootutils

pyrootutils.setup_root(
    search_from=__file__,
    indicator=".gitignore",
    project_root_env_var=True,
    pythonpath=True,
)
import os
import time
from argparse import ArgumentParser
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.optim as optim
from collections import OrderedDict
from configs import get_config
import torch
from os.path import join as pjoin
from models import build_models
from models.utils import CosineWarmupScheduler
from datasets import DataModule
from datasets.interhuman import MotionNormalizerTorch
from datasets.utils import lengths_to_mask
from utils.metrics import (
    calculate_activation_statistics,
    calculate_frechet_distance,
)
from evaluators.evaluator_interhuman import EvaluatorModelWrapper

os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("medium")


class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg, general_cfg=None):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(general_cfg.CHECKPOINT, general_cfg.EXP_NAME)
        # self.save_root = pjoin(self.save_root, time.strftime("%y%m%d-%H%M%S"))
        self.model_dir = pjoin(self.save_root, "model")
        self.meta_dir = pjoin(self.save_root, "meta")
        self.log_dir = pjoin(self.save_root, "log")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.save_hyperparameters(model.cfg)

        self.dataset_name = general_cfg.DATASET_NAME
        if self.dataset_name == "interhuman":
            self._normalizer = MotionNormalizerTorch()
        elif self.dataset_name == "interx":
            self._normalizer = torch.nn.Identity()
        else:
            raise ValueError

        if getattr(cfg, "EVAL", False):
            evalmodel_cfg = get_config("eval_model/interclip.yaml")
            self.eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device=self.device)

    def on_save_checkpoint(self, checkpoint):
        # pop the backbone here using custom logic
        for k in checkpoint.keys():
            if k.startswith("model.clip_transformer"):
                del checkpoint[k]
            elif k.startswith("model.token_embedding"):
                del checkpoint[k]
            elif k.startswith("model.ln_final"):
                del checkpoint[k]
            elif k.startswith("model.clip_model"):
                del checkpoint[k]

    def _configure_optim(self):
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.LR),
            weight_decay=self.cfg.WEIGHT_DECAY,
        )
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer, warmup=10, max_iters=self.cfg.EPOCH, verbose=True
        )
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        if self.dataset_name == "interhuman":
            name, text, motion1, motion2, motion_lens = batch_data
        elif self.dataset_name == "interx":
            _, _, text, _, motions, motion_lens, _ = batch_data
            motion1, motion2 = motions.chunk(2, dim=-1)
        else:
            raise ValueError

        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)

        # NOTE: normalize!!!!!
        normed_motion1 = self._normalizer.forward(motion1)
        normed_motion2 = self._normalizer.forward(motion2)  # B T C
        mask = lengths_to_mask(motion_lens, normed_motion1.shape[1])
        if len(normed_motion1.shape) == 3:
            normed_motion1 = normed_motion1 * mask[..., None]
            normed_motion2 = normed_motion2 * mask[..., None]
        elif len(normed_motion1.shape) == 4:
            normed_motion1 = normed_motion1 * mask[..., None, None]
            normed_motion2 = normed_motion2 * mask[..., None, None]
        else:
            raise ValueError

        motions = torch.cat([motion1, motion2], dim=-1)
        normed_motions = torch.cat([normed_motion1, normed_motion2], dim=-1)

        batch = OrderedDict({})
        batch["text"] = text
        batch["motions"] = motions.type(torch.float32)
        batch["normed_motions"] = normed_motions.type(torch.float32)
        batch["motion_lens"] = motion_lens.long()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.LAST_ITER if self.cfg.LAST_ITER else 0
        self.epoch = self.cfg.LAST_EPOCH if self.cfg.LAST_EPOCH else 0
        self.logs = OrderedDict()

    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1.0)
        opt.step()

        return {"loss": loss, "loss_logs": loss_logs}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get("skip_batch") or not outputs.get("loss_logs"):
            return
        for k, v in outputs["loss_logs"].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.LOG_STEPS
                self.log(f"Train/{tag}", mean_loss[tag], prog_bar=True)
            self.logs = OrderedDict()

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def on_validation_epoch_start(self):
        self.val_text_embeddings = []
        self.val_gen_motion_embeddings = []
        self.val_gt_motion_embeddings = []
        return

    def validation_step(self, batch_data, batch_idx):
        if not getattr(self.cfg, "EVAL", False):
            return

        if self.dataset_name == "interhuman":
            _, text, motion1, motion2, motion_lens = batch_data
        elif self.dataset_name == "interx":
            _, _, text, _, motions, motion_lens, _ = batch_data
            motion1, motion2 = motions.chunk(2, dim=-1)
        else:
            raise ValueError

        batch_dict = {
            "text": list(text),
            "motion_lens": motion_lens.long() + 3,
        }
        # NOTE: normalize!!!!!
        batch_output = self.model.forward_test(batch_dict)
        motions_output = batch_output["output"].reshape(
            batch_output["output"].shape[0], batch_output["output"].shape[1], 2, -1
        )
        motions_output = self._normalizer.backward(motions_output)

        mask = lengths_to_mask(motion_lens.long(), motions_output.shape[1])
        motions_output = motions_output * mask[..., None, None]

        B, T, _, D = motions_output.shape
        if T < motion1.shape[1]:
            padding_len = motion1.shape[1] - T
            padding_zeros = torch.zeros(
                (B, padding_len, 2, D), device=motions_output.device
            )
            motions_output = torch.cat([motions_output, padding_zeros], dim=1)
        assert motions_output.shape[1] == motion1.shape[1]

        text_embeddings, gen_motion_embeddings = self.eval_wrapper.get_co_embeddings(
            [
                None,
                text,
                motions_output[:, :, 0],
                motions_output[:, :, 1],
                motion_lens,
            ]
        )
        gt_motion_embeddings = self.eval_wrapper.get_motion_embeddings(
            [None, text, motion1, motion2, motion_lens]
        )

        self.val_text_embeddings.append(text_embeddings)
        self.val_gen_motion_embeddings.append(gen_motion_embeddings)
        self.val_gt_motion_embeddings.append(gt_motion_embeddings)
        return

    def on_validation_epoch_end(self):
        if not getattr(self.cfg, "EVAL", False):
            return

        gt_motion_embeddings = torch.cat(self.val_gt_motion_embeddings, dim=0)
        motion_embeddings = torch.cat(self.val_gen_motion_embeddings, dim=0)
        text_embeddings = torch.cat(self.val_text_embeddings, dim=0)

        gt_m_embs = self.all_gather(gt_motion_embeddings)
        m_embs = self.all_gather(motion_embeddings)
        text_embs = self.all_gather(text_embeddings)

        self.val_gt_motion_embeddings = []
        self.val_gen_motion_embeddings = []
        self.val_text_embeddings = []

        if self.trainer.is_global_zero:
            gt_m_embs = gt_m_embs.flatten(0, 1).cpu().numpy()
            m_embs = m_embs.flatten(0, 1).cpu().numpy()
            text_embs = text_embs.flatten(0, 1).cpu().numpy()
            gt_mu, gt_cov = calculate_activation_statistics(gt_m_embs)
            mu, cov = calculate_activation_statistics(m_embs)
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        else:
            fid = 0
        self.log(
            f"Val/FID",
            fid,
            on_epoch=True,
            prog_bar=True,
            reduce_fx="sum",
            sync_dist=True,
        )
        return

    def save(self, file_name):
        state = {}
        try:
            state["model"] = self.model.module.state_dict()
        except:
            state["model"] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return


def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=False, default=None, help="total config file."
    )
    params = parser.parse_args()
    return params


if __name__ == "__main__":
    print(os.getcwd())
    params = parse_args()

    cfg = OmegaConf.load(params.cfg)
    OmegaConf.resolve(cfg)  # support reference in yaml
    model_cfg = cfg.model
    train_cfg = cfg.get("TRAIN")
    data_cfg = cfg.dataset
    general_cfg = cfg.get("GENERAL")

    datamodule = DataModule(data_cfg, train_cfg.BATCH_SIZE, train_cfg.NUM_WORKERS)
    model = build_models(model_cfg)

    if train_cfg.RESUME:
        ckpt = torch.load(train_cfg.RESUME, map_location="cpu", weights_only=False)
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg, general_cfg)

    # save config
    OmegaConf.save(
        cfg,
        pjoin(general_cfg.CHECKPOINT, general_cfg.EXP_NAME, "config.yaml"),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=litmodel.model_dir,
        every_n_epochs=train_cfg.SAVE_EPOCH,
        save_top_k=-1,  # -1: save all models, 5 for save disk memory
        save_last=True,
    )
    fid_checkpoint_callback = ModelCheckpoint(
        monitor="Val/FID",
        mode="min",
        dirpath=litmodel.model_dir,
        filename="FID-{Val/FID:.4f}-epoch{epoch:02d}",
        save_weights_only=True,
        save_top_k=5,
        auto_insert_metric_name=False,
    )

    if getattr(train_cfg, "EVAL", False):
        callbacks = [
            checkpoint_callback,
            fid_checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ]
    else:
        callbacks = [
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=litmodel.log_dir)

    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices="auto",
        accelerator="gpu",
        max_epochs=train_cfg.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=callbacks,
        logger=[tb_logger],
        check_val_every_n_epoch=50,
    )

    trainer.fit(model=litmodel, datamodule=datamodule)
