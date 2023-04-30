import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pretrain.dataloader_nuscenes import (
        NuScenesMatchDataset,
        minkunet_collate_pair_fn,
)

from pretrain.dataloader_kitti import (
    KittiMatchDataset,
    kitti_collate_pair_fn,
)

from pretrain.dataloader_scannet import (
    scannet_Dataset,
    scannet_collate_pair_fn,
)

# try:
#     from pretrain.dataloader_scannet import (
#         scannet_Dataset,
#         scannet_collate_pair_fn,
#     )
# except ImportError:
#     scannet_Dataset = None
#     scannet_collate_pair_fn = None

try:
    from pretrain.dataloader_nuscenes_spconv import NuScenesMatchDatasetSpconv, spconv_collate_pair_fn
except ImportError:
    NuScenesMatchDatasetSpconv = None
    spconv_collate_pair_fn = None
from utils.transforms import (
    make_transforms_clouds,
    make_transforms_asymmetrical,
    make_transforms_asymmetrical_val,
)


class PretrainDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["num_gpus"]:
            self.batch_size = config["batch_size"] // config["num_gpus"]
        else:
            self.batch_size = config["batch_size"]

    def setup(self, stage):
        cloud_transforms_train = make_transforms_clouds(self.config)
        mixed_transforms_train = make_transforms_asymmetrical(self.config)
        cloud_transforms_val = None
        mixed_transforms_val = make_transforms_asymmetrical_val(self.config)

        if self.config["dataset"].lower() == "nuscenes" and self.config["model_points"] == "minkunet":
            Dataset = NuScenesMatchDataset
        elif self.config["dataset"].lower() == "kitti":
            Dataset = KittiMatchDataset
        elif self.config["dataset"].lower() == "scannet":
            Dataset = scannet_Dataset
        elif self.config["dataset"].lower() == "nuscenes" and self.config["model_points"] == "voxelnet":
            Dataset = NuScenesMatchDatasetSpconv
        else:
            raise Exception("Dataset Unknown")

        # print(self.config["dataset"].lower())
        # print(type(Dataset))

        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"
        self.train_dataset = Dataset(
            phase=phase_train,
            config=self.config,
            shuffle=True,
            cloud_transforms=cloud_transforms_train,
            mixed_transforms=mixed_transforms_train,
        )
        print("Dataset Loaded")
        print("training size: ", len(self.train_dataset))

        if self.config["dataset"].lower() == "nuscenes":
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
                # cached_nuscenes=None,
            )
        else:
            self.val_dataset = Dataset(
                phase=phase_val,
                shuffle=False,
                cloud_transforms=cloud_transforms_val,
                mixed_transforms=mixed_transforms_val,
                config=self.config,
                # cached_nuscenes=self.train_dataset.nusc,
                # cached_nuscenes=None,
            )

        print("validation size: ", len(self.val_dataset))

    def train_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]

        if self.config["dataset"].lower() == "nuscenes":
            default_collate_pair_fn = minkunet_collate_pair_fn
        elif self.config["dataset"].lower() == "kitti":
            default_collate_pair_fn = kitti_collate_pair_fn
        elif self.config["dataset"].lower() == "scannet":
            default_collate_pair_fn = scannet_collate_pair_fn

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):

        if self.config["num_gpus"]:
            num_workers = self.config["num_threads"] // self.config["num_gpus"]
        else:
            num_workers = self.config["num_threads"]

        if self.config["dataset"].lower() == "nuscenes":
            default_collate_pair_fn = minkunet_collate_pair_fn
        elif self.config["dataset"].lower() == "kitti":
            default_collate_pair_fn = kitti_collate_pair_fn
        elif self.config["dataset"].lower() == "scannet":
            default_collate_pair_fn = scannet_collate_pair_fn

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=default_collate_pair_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
