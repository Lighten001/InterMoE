from os.path import join as pjoin
import pytorch_lightning as pl
import torch
from .interhuman import InterHumanDataset, InterHumanMotion
from .interx import MotionDatasetV2HHI, Text2MotionDatasetV2HHI



class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "interhuman":
            if self.cfg.STAGE == "VAE":
                self.train_dataset = InterHumanMotion(self.cfg.interhuman)
            else:
                self.train_dataset = InterHumanDataset(self.cfg.interhuman)
            self.val_dataset = InterHumanDataset(self.cfg.interhuman_val)
        elif self.cfg.NAME == "interx":
            from utils.word_vectorizer import WordVectorizer
            if self.cfg.STAGE == "VAE":
                self.train_dataset = MotionDatasetV2HHI(
                    self.cfg.train,
                    self.cfg.train.split_file,
                    pjoin(self.cfg.train.motion_dir, "train.h5"),
                )
            else:
                self.train_dataset = Text2MotionDatasetV2HHI(
                    self.cfg.train,
                    self.cfg.train.split_file,
                    WordVectorizer(pjoin(self.cfg.DATA_PROCESSED, "glove"), "hhi_vab"),
                    pjoin(self.cfg.train.motion_dir, "train.h5"),
                )
            self.val_dataset = Text2MotionDatasetV2HHI(
                    self.cfg.val,
                    self.cfg.val.split_file,
                    WordVectorizer(pjoin(self.cfg.DATA_PROCESSED, "glove"), "hhi_vab"),
                    pjoin(self.cfg.val.motion_dir, "val.h5"),
                )
        else:
            raise NotImplementedError


        self._train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

        self._val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=32,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=True,
        )
    

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return self._train_dataloader
    
    def val_dataloader(self):
        return self._val_dataloader