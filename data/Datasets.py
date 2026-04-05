import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytorch3d.transforms as p3dtrans
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.DatasetUtils import (
    color_norm,
    coor_norm,
    get_wpc10_multi_path,
    get_wpc10_patch_paths,
    load_ply,
    validate_files,
)
from utils.PCUtils import pad_to_size


class WPC10PointPatchDataset(Dataset):
    def __init__(
        self,
        pc_path: str,
        mos_path: str,
        patch_num: int = 32,
        select_patch: int = 32,
        mos_scale: float = 100.0,
        random_select: bool = False,
        data_aug: bool = False,
        data_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.records = get_wpc10_patch_paths(pc_path, mos_path, patch_num)
        self.patch_num = patch_num
        self.select_patch = select_patch
        self.mos_scale = mos_scale
        self.random_select = random_select
        self.data_aug = data_aug
        self.data_normalization = data_normalization
        self.pc_aug = p3dtrans.random_rotation()

    def __len__(self) -> int:
        return len(self.records)

    def _select_patch_indices(self) -> List[int]:
        indices = list(range(self.patch_num))
        if self.random_select:
            random.shuffle(indices)
        return indices[: self.select_patch]

    def _load_patch(self, path: str) -> np.ndarray:
        ply = pad_to_size(load_ply(path), pad_size=8192).astype(np.float32)
        if self.data_normalization:
            xyz = coor_norm(ply[:, :3])
            rgb = color_norm(ply[:, 3:])
            ply = np.concatenate([xyz, rgb], axis=1)
        return ply

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        name, pc_paths, mos = self.records[index]
        selected = self._select_patch_indices()
        chosen_paths = [pc_paths[i] for i in selected]
        validate_files(chosen_paths)
        patches = [self._load_patch(path) for path in chosen_paths]
        pcs = torch.tensor(np.stack(patches), dtype=torch.float32)
        if self.data_aug:
            pcs[:, :, :3] = torch.bmm(pcs[:, :, :3], self.pc_aug.to(pcs.device).expand(pcs.shape[0], -1, -1))
        return name, pcs, torch.tensor(mos / self.mos_scale, dtype=torch.float32)


class WPC10MultiDataset(Dataset):
    def __init__(
        self,
        pc_path: str,
        pic_path: str,
        mos_path: str,
        patch_num: int = 32,
        select_patch: int = 32,
        mos_scale: float = 100.0,
        random_select: bool = False,
        data_aug: bool = False,
        data_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.records = get_wpc10_multi_path(pc_path, pic_path, mos_path, patch_num)
        self.patch_num = patch_num
        self.select_patch = select_patch
        self.mos_scale = mos_scale
        self.random_select = random_select
        self.data_aug = data_aug
        self.data_normalization = data_normalization
        self.to_tensor = transforms.ToTensor()
        self.pic_aug = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        )
        self.pc_aug = p3dtrans.random_rotation()

    def __len__(self) -> int:
        return len(self.records)

    def _select_patch_indices(self) -> List[int]:
        indices = list(range(self.patch_num))
        if self.random_select:
            random.shuffle(indices)
        return indices[: self.select_patch]

    def _load_patch(self, path: str) -> np.ndarray:
        ply = pad_to_size(load_ply(path), pad_size=8192).astype(np.float32)
        if self.data_normalization:
            xyz = coor_norm(ply[:, :3])
            rgb = color_norm(ply[:, 3:])
            ply = np.concatenate([xyz, rgb], axis=1)
        return ply

    def __getitem__(self, index: int):
        name, pc_paths, pic_paths, mos = self.records[index]
        selected = self._select_patch_indices()
        chosen_pc = [pc_paths[i] for i in selected]
        chosen_pic = [pic_paths[i] for i in selected]
        validate_files(chosen_pc + chosen_pic)

        pc_patches = [self._load_patch(path) for path in chosen_pc]
        pic_patches = []
        for path in chosen_pic:
            pic = cv2.imread(path)
            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
            pic_patches.append(self.to_tensor(pic))

        pcs = torch.tensor(np.stack(pc_patches), dtype=torch.float32)
        pics = torch.stack(pic_patches, dim=0).to(torch.float32)

        if self.data_aug:
            pcs[:, :, :3] = torch.bmm(pcs[:, :, :3], self.pc_aug.to(pcs.device).expand(pcs.shape[0], -1, -1))
            pics = self.pic_aug(pics)

        return name, pcs, pics, torch.tensor(mos / self.mos_scale, dtype=torch.float32)


def build_tnet_datasets(config: Dict) -> Tuple[WPC10PointPatchDataset, WPC10PointPatchDataset]:
    dataset_cfg = config["dataset"]
    train = WPC10PointPatchDataset(
        pc_path=dataset_cfg["pc_path"],
        mos_path=dataset_cfg["traininfo"],
        patch_num=dataset_cfg["patch_num"],
        select_patch=dataset_cfg["patch_num"],
        mos_scale=dataset_cfg["train_mos_scale"],
        random_select=dataset_cfg.get("random", False),
        data_aug=dataset_cfg.get("data_aug", False),
        data_normalization=dataset_cfg.get("data_normalization", False),
    )
    val = WPC10PointPatchDataset(
        pc_path=dataset_cfg["pc_path"],
        mos_path=dataset_cfg["vailinfo"],
        patch_num=dataset_cfg["patch_num"],
        select_patch=dataset_cfg["patch_num"],
        mos_scale=dataset_cfg["vail_mos_scale"],
        random_select=False,
        data_aug=False,
        data_normalization=dataset_cfg.get("data_normalization", False),
    )
    return train, val


def build_multi_datasets(config: Dict) -> Tuple[WPC10MultiDataset, WPC10MultiDataset]:
    dataset_cfg = config["dataset"]
    train = WPC10MultiDataset(
        pc_path=dataset_cfg["pc_path"],
        pic_path=dataset_cfg["pic_path"],
        mos_path=dataset_cfg["traininfo"],
        patch_num=dataset_cfg["patch_num"],
        select_patch=dataset_cfg["train_patch_num"],
        mos_scale=dataset_cfg["train_mos_scale"],
        random_select=dataset_cfg.get("random", False),
        data_aug=dataset_cfg.get("data_aug", False),
        data_normalization=dataset_cfg.get("data_normalization", False),
    )
    val = WPC10MultiDataset(
        pc_path=dataset_cfg["pc_path"],
        pic_path=dataset_cfg["pic_path"],
        mos_path=dataset_cfg["vailinfo"],
        patch_num=dataset_cfg["patch_num"],
        select_patch=dataset_cfg["vail_patch_num"],
        mos_scale=dataset_cfg["vail_mos_scale"],
        random_select=False,
        data_aug=False,
        data_normalization=dataset_cfg.get("data_normalization", False),
    )
    return train, val
