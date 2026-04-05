import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import ops
from torch.optim.lr_scheduler import StepLR
from vector_quantize_pytorch import ResidualVQ

from model.adaptProjection import build_stn
from model.metrics import IQA_accruacy
from model.pointnet2 import pointnet2
from model.swin_trans import SwinTransformer


def distance_correlation(latent: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    latent = F.normalize(latent, dim=-1)
    control = F.normalize(control, dim=-1)

    matrix_a = torch.sqrt(torch.sum((latent.unsqueeze(0) - latent.unsqueeze(1)) ** 2, dim=-1) + 1e-12)
    matrix_b = torch.sqrt(torch.sum((control.unsqueeze(0) - control.unsqueeze(1)) ** 2, dim=-1) + 1e-12)

    matrix_A = matrix_a - matrix_a.mean(dim=0, keepdim=True) - matrix_a.mean(dim=1, keepdim=True) + matrix_a.mean()
    matrix_B = matrix_b - matrix_b.mean(dim=0, keepdim=True) - matrix_b.mean(dim=1, keepdim=True) + matrix_b.mean()

    gamma_xy = torch.sum(matrix_A * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    gamma_xx = torch.sum(matrix_A * matrix_A) / (matrix_A.shape[0] * matrix_A.shape[1])
    gamma_yy = torch.sum(matrix_B * matrix_B) / (matrix_A.shape[0] * matrix_A.shape[1])
    return gamma_xy / torch.sqrt(gamma_xx * gamma_yy + 1e-9)


def orient_normals_outward(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    center = points.mean(dim=1, keepdim=True)
    directions = points - center
    flip_mask = (directions * normals).sum(dim=-1, keepdim=True) < 0
    return torch.where(flip_mask, -normals, normals)


class FeatureTransformRegularizer(nn.Module):
    def forward(self, trans: torch.Tensor) -> torch.Tensor:
        dim = trans.size(1)
        identity = torch.eye(dim, device=trans.device)[None, :, :]
        return torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - identity, dim=(1, 2)))


class NormalAlignmentLoss(nn.Module):
    def forward(self, aligned_normals: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
        plane_normal = plane_normal.unsqueeze(dim=-1).expand_as(aligned_normals)
        loss = torch.cosine_similarity(aligned_normals, plane_normal, dim=1)
        return 1 - torch.mean(loss, dim=-1)


class TNetAlignmentModel(pl.LightningModule):
    def __init__(self, learning_rate: float, weight_decay: float, rotation_mode: str = "matrix") -> None:
        super().__init__()
        self.save_hyperparameters()
        self.stn = build_stn(rotation_mode)
        self.regularizer = FeatureTransformRegularizer()
        self.normal_loss = NormalAlignmentLoss()
        self.rotation_mode = rotation_mode

    def forward(self, pcs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, patch_num, num_points, channels = pcs.shape
        pcs = pcs.view(batch_size * patch_num, num_points, channels)
        pcs = pcs.permute(0, 2, 1)

        xyz = pcs[:, :3, :]
        points = xyz.permute(0, 2, 1)
        normals = ops.points_normals.estimate_pointcloud_normals(points, 50)
        normals = orient_normals_outward(points, normals).permute(0, 2, 1)
        transform = self.stn(normals)
        aligned_xyz = torch.bmm(transform, xyz)
        aligned_normals = torch.bmm(transform, normals)
        aligned_pcs = torch.cat([aligned_xyz, pcs[:, 3:, :]], dim=1)

        plane_normal = torch.tensor([0.0, 0.0, 1.0], device=pcs.device)
        normal_loss = self.normal_loss(aligned_normals, plane_normal)
        reg_loss = self.regularizer(transform) if self.rotation_mode == "matrix" else torch.zeros(1, device=pcs.device)
        loss = normal_loss + reg_loss
        return loss, normal_loss, reg_loss, aligned_pcs

    def training_step(self, batch, batch_idx):
        _, pcs, mos = batch
        loss, normal_loss, reg_loss, _ = self(pcs)
        return {
            "loss": loss.mean(),
            "normal_loss": normal_loss.mean(),
            "reg_loss": reg_loss.mean(),
        }

    def validation_step(self, batch, batch_idx):
        _, pcs, mos = batch
        loss, normal_loss, reg_loss, _ = self(pcs)
        return {
            "loss": loss.mean(),
            "normal_loss": normal_loss.mean(),
            "reg_loss": reg_loss.mean(),
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("loss/train_loss", outputs["loss"], prog_bar=True)
        self.log("loss/train_normal_loss", outputs["normal_loss"], prog_bar=False)
        self.log("loss/train_reg_loss", outputs["reg_loss"], prog_bar=False)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("loss/val_loss", outputs["loss"], prog_bar=True)
        self.log("loss/val_normal_loss", outputs["normal_loss"], prog_bar=False)
        self.log("loss/val_reg_loss", outputs["reg_loss"], prog_bar=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.97, verbose=True)
        return [optimizer], [scheduler]


class MultiModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_param: Dict,
        embedding_size: int = 128,
        embedding_dim: int = 1024,
        l_w_embedding: float = 0.5,
        l_w_commitment: float = 0.8,
        swin_ckpt_path: str = "checks/swin_tiny_patch4_window7_224.pth",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = optimizer_param["learning_rate"]
        self.weight_decay = optimizer_param["weight_decay"]
        self.l_w_embedding = l_w_embedding
        self.l_w_commitment = l_w_commitment
        self.embedding_size = embedding_size
        self.embedding_dim = embedding_dim

        self.pic_backbone = SwinTransformer(num_classes=0)
        state = torch.load(swin_ckpt_path, map_location="cpu")
        self.pic_backbone.load_state_dict(state["model"], strict=False)

        self.pc_backbone = pointnet2(8192)
        self.loss = nn.MSELoss()
        self.val_acc = IQA_accruacy
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.best_metrics = {
            "srocc": float("-inf"),
            "plcc": float("-inf"),
            "krocc": float("-inf"),
            "epoch": -1,
        }

        self.pc_linear = nn.Linear(512, embedding_dim)
        self.pic_linear = nn.Linear(768, embedding_dim)

        self.vq = ResidualVQ(
            dim=embedding_size,
            num_quantizers=4,
            codebook_size=max(embedding_size, 128),
            stochastic_sample_codes=True,
            sample_codebook_temp=0.1,
            shared_codebook=True,
        )

        def make_projection_head():
            return nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(embedding_dim // 2, embedding_dim // 4),
                nn.LeakyReLU(),
                nn.Linear(embedding_dim // 4, embedding_size),
            )

        self.modal_map_pc = make_projection_head()
        self.modal_map_pic = make_projection_head()
        self.semantic_map_pc = make_projection_head()
        self.semantic_map_pic = make_projection_head()

        self.reconstruction_pc = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )
        self.reconstruction_pic = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
        )

        fusion_dim = embedding_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(fusion_dim // 4, 1),
        )

    def forward(self, pcs: torch.Tensor, pics: torch.Tensor, mos: torch.Tensor = None):
        batch_size, patch_num, _, height, width = pics.shape
        _, _, num_points, channels = pcs.shape

        pcs = pcs.view(batch_size * patch_num, num_points, channels).permute(0, 2, 1)
        pics = pics.view(batch_size * patch_num, 3, height, width)

        pc_feature = self.pc_backbone(pcs)
        pic_feature = self.pic_backbone(pics)

        pc_feature = self.pc_linear(pc_feature)
        pic_feature = self.pic_linear(pic_feature)

        modal_pc = self.modal_map_pc(pc_feature)
        modal_pic = self.modal_map_pic(pic_feature)
        q_pc, _, l_pc = self.vq(modal_pc)
        q_pic, _, l_pic = self.vq(modal_pic)

        semantic_pc = self.semantic_map_pc(pc_feature)
        semantic_pic = self.semantic_map_pic(pic_feature)

        rec_pc = self.reconstruction_pc(torch.cat([q_pc, semantic_pc], dim=-1))
        rec_pic = self.reconstruction_pic(torch.cat([q_pic, semantic_pic], dim=-1))
        rec_loss = (F.mse_loss(rec_pc, pc_feature) + F.mse_loss(rec_pic, pic_feature)) / 2

        multi_loss = (
            distance_correlation(modal_pc, semantic_pc) + distance_correlation(modal_pic, semantic_pic)
        ) / 2

        pc_feature = pc_feature.view(batch_size, patch_num, -1)
        pic_feature = pic_feature.view(batch_size, patch_num, -1)
        fused = torch.cat([pc_feature, pic_feature], dim=-1).mean(dim=1)
        prediction = self.fc(fused)

        embedding_loss = ((l_pc + l_pic) / 2).mean()
        if mos is None:
            total_loss = None
        else:
            regression_loss = self.loss(prediction, mos.unsqueeze(-1))
            total_loss = (
                regression_loss
                + 0.5 * embedding_loss
                + 0.1 * rec_loss
                + ((self.current_epoch + 1) / 10) * 0.01 * multi_loss
            )

        losses = {
            "loss": total_loss,
            "embedding_loss": embedding_loss,
            "align_loss": rec_loss,
            "multi_loss": multi_loss,
        }
        return prediction, losses

    def training_step(self, batch, batch_idx):
        _, pcs, pics, mos = batch
        prediction, losses = self(pcs, pics, mos)
        self.train_step_outputs.extend(
            (a, b) for a, b in zip(prediction.detach().flatten().tolist(), mos.detach().flatten().tolist())
        )
        return {
            "loss": losses["loss"],
            "embedding_loss": losses["embedding_loss"],
            "align_loss": losses["align_loss"],
            "multi_loss": losses["multi_loss"],
        }

    def validation_step(self, batch, batch_idx):
        _, pcs, pics, mos = batch
        prediction, losses = self(pcs, pics, mos)
        self.validation_step_outputs.extend(
            (a, b) for a, b in zip(prediction.detach().flatten().tolist(), mos.detach().flatten().tolist())
        )
        return {
            "loss": losses["loss"],
            "embedding_loss": losses["embedding_loss"],
            "align_loss": losses["align_loss"],
            "multi_loss": losses["multi_loss"],
        }

    def test_step(self, batch, batch_idx):
        _, pcs, pics, mos = batch
        prediction, _ = self(pcs, pics, mos)
        self.validation_step_outputs.extend(
            (a, b) for a, b in zip(prediction.detach().flatten().tolist(), mos.detach().flatten().tolist())
        )
        return {"prediction": prediction.detach(), "mos": mos.detach()}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log("loss/train_loss", outputs["loss"], prog_bar=True)
        self.log("loss/train_embedding_loss", outputs["embedding_loss"], prog_bar=False)
        self.log("loss/train_align_loss", outputs["align_loss"], prog_bar=False)
        self.log("loss/train_multi_loss", outputs["multi_loss"], prog_bar=False)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        self.log("loss/val_loss", outputs["loss"], prog_bar=True)
        self.log("loss/val_embedding_loss", outputs["embedding_loss"], prog_bar=False)
        self.log("loss/val_align_loss", outputs["align_loss"], prog_bar=False)
        self.log("loss/val_multi_loss", outputs["multi_loss"], prog_bar=False)

    def on_train_epoch_end(self):
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) <= 2:
            return
        vails = torch.transpose(torch.tensor(self.validation_step_outputs, dtype=torch.float32), 0, 1)
        accs = self.val_acc(vails[1].numpy(), vails[0].numpy())
        current_metrics = {
            "srocc": float(accs[0]),
            "plcc": float(accs[1]),
            "krocc": float(accs[2]),
        }
        if current_metrics["srocc"] > self.best_metrics["srocc"]:
            self.best_metrics = {
                "srocc": current_metrics["srocc"],
                "plcc": current_metrics["plcc"],
                "krocc": current_metrics["krocc"],
                "epoch": int(self.current_epoch),
            }
        self.logger.experiment.add_scalars(
            "acc/vail",
            {"srocc": accs[0], "plcc": accs[1], "krocc": accs[2]},
            self.global_step,
        )
        self.log_dict(
            {
                "acc/vail_srocc": accs[0],
                "acc/vail_plcc": accs[1],
                "acc/vail_krocc": accs[2],
            },
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        print(
            (
                f"[val] epoch={self.current_epoch} "
                f"current: srocc={current_metrics['srocc']:.4f}, "
                f"plcc={current_metrics['plcc']:.4f}, "
                f"krocc={current_metrics['krocc']:.4f} | "
                f"best: epoch={self.best_metrics['epoch']}, "
                f"srocc={self.best_metrics['srocc']:.4f}, "
                f"plcc={self.best_metrics['plcc']:.4f}, "
                f"krocc={self.best_metrics['krocc']:.4f}"
            ),
            flush=True,
        )

        output_dir = os.path.join(self.trainer.logger.log_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({"pred": vails[0].numpy(), "mos": vails[1].numpy()}).to_csv(
            os.path.join(output_dir, f"result_{self.current_epoch}.csv"),
            index=False,
        )
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        if not self.validation_step_outputs:
            return
        vails = torch.transpose(torch.tensor(self.validation_step_outputs, dtype=torch.float32), 0, 1)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame({"pred": vails[0].numpy(), "mos": vails[1].numpy()}).to_csv(
            os.path.join(output_dir, "result_test.csv"),
            index=False,
        )
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = StepLR(
            optimizer,
            step_size=self.hparams.optimizer_param["lr_decay_step_size"],
            gamma=self.hparams.optimizer_param["lr_decay_gamma"],
            verbose=True,
        )
        return [optimizer], [scheduler]
