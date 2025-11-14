import copy
from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import torch
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter1d
import os
from datasets.interhuman import MotionNormalizer
from os.path import join as pjoin
from models import build_models
from collections import OrderedDict
from configs import get_config
from utils.plot import plot_interaction


class LitGenModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(cfg.GENERAL.CHECKPOINT, cfg.GENERAL.EXP_NAME)

        self.result_dir = pjoin(
            self.save_root, "results", datetime.now().strftime("%y%m%d-%H%M%S")
        )
        print(self.result_dir)

        os.makedirs(self.result_dir, exist_ok=True)

        # train model init
        self.model = model

        # others init
        self.normalizer = MotionNormalizer()

    def generate_one_sample(self, prompt, name):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1).long().cuda()
        batch["prompt"] = prompt

        window_size = 210
        motion_output = self.generate_loop(batch, window_size, name=name)

        return motion_output

    def generate_loop(self, batch, window_size, name="output"):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][:] = window_size

        sequences = [[], []]

        batch["text"] = [prompt]
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"][0].reshape(
            batch["output"][0].shape[0], 2, -1
        )
        motion_output_both = self.normalizer.backward(
            motion_output_both.cpu().detach().numpy()
        )

        plot_interaction(motion_output_both, prompt, f"{self.result_dir}", name)

        for j in range(2):
            motion_output = motion_output_both[:, j]

            joints3d = motion_output[:, : 22 * 3].reshape(-1, 22, 3)
            joints3d = gaussian_filter1d(joints3d, 3, axis=0, mode="nearest")
            sequences[j].append(joints3d)

        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)
        return sequences


def parse_args():

    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=False, default=None, help="total config file."
    )
    params = parser.parse_args()
    return params


if __name__ == "__main__":
    # torch.manual_seed(37)
    params = parse_args()
    cfg_path = params.cfg
    model_cfg = get_config(cfg_path).model
    infer_cfg = get_config(cfg_path)

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        if torch.__version__ >= "2.0":
            ckpt = torch.load(
                model_cfg.CHECKPOINT, map_location="cpu", weights_only=False
            )
        else:
            ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))

    with open("./prompts.txt") as f:
        texts = f.readlines()
    texts = [text.strip("\n") for text in texts]

    for text in texts:
        name = text[:48]
        for i in range(3):
            litmodel.generate_one_sample(text, name + "_" + str(i))
