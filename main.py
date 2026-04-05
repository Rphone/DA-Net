import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytorch_lightning as pl
import torch
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping
from pytorch3d import ops
from torch.utils.data import DataLoader

from data.DatasetUtils import ensure_dir, get_wpc10_split_records
from data.Datasets import build_multi_datasets, build_tnet_datasets
from model.IQA_Model import MultiModel, TNetAlignmentModel
from model.adaptProjection import build_stn
from utils.projection import (
    PLANE_NORMAL_Z,
    build_pytorch3d_scatter_renderer,
    render_projection_image_pytorch3d_scatter,
    render_projection_image_reference,
)


REPO_ROOT = Path(__file__).resolve().parent


def orient_normals_outward(points: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    center = points.mean(dim=1, keepdim=True)
    directions = points - center
    flip_mask = (directions * normals).sum(dim=-1, keepdim=True) < 0
    return torch.where(flip_mask, -normals, normals)


def resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((REPO_ROOT / candidate).resolve())


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    dataset_cfg = config["dataset"]
    dataset_cfg["traininfo"] = resolve_path(dataset_cfg["traininfo"])
    dataset_cfg["vailinfo"] = resolve_path(dataset_cfg["vailinfo"])
    dataset_cfg["pc_path"] = resolve_path(dataset_cfg["pc_path"])
    dataset_cfg["pic_path"] = resolve_path(dataset_cfg["pic_path"])
    return config


def seed_and_precision(config: Dict) -> None:
    torch.set_float32_matmul_precision(config["torch"]["precision"])
    pl.seed_everything(config["torch"]["seed"])


def trainer_precision(config: Dict):
    return "16-mixed" if config["torch"].get("mix_precision16", False) and torch.cuda.is_available() else 32


def trainer_accelerator() -> str:
    return "gpu" if torch.cuda.is_available() else "cpu"


def trainer_devices() -> int:
    return 1


def build_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def rotation_mode_name(config: Dict, override: Optional[str] = None) -> str:
    if override:
        return override
    return config["hyperparameters"].get("tnet_rotation_mode", "matrix")


def tnet_log_name(rotation_mode: str) -> str:
    return "tnet" if rotation_mode == "matrix" else f"tnet_{rotation_mode}"


def latest_tnet_ckpt(rotation_mode: str = "matrix") -> Optional[str]:
    preferred = [
        REPO_ROOT / "checks" / f"wpc10_tnet_{rotation_mode}.ckpt",
    ]
    if rotation_mode == "matrix":
        preferred.append(REPO_ROOT / "checks" / "wpc10_tnet.ckpt")
    for candidate in preferred:
        if candidate.exists():
            return str(candidate)
    logs_root = REPO_ROOT / "logs" / tnet_log_name(rotation_mode)
    if logs_root.exists():
        ckpts = sorted(logs_root.glob("version_*/checkpoints/*.ckpt"))
        for candidate in reversed(ckpts):
            if candidate.name != "last.ckpt":
                return str(candidate)
        if ckpts:
            return str(ckpts[-1])
    return None


def split_names(config: Dict) -> Iterable[str]:
    seen = set()
    for split_path in (config["dataset"]["traininfo"], config["dataset"]["vailinfo"]):
        for name, _ in get_wpc10_split_records(split_path):
            if name not in seen:
                seen.add(name)
                yield name


def required_projection_paths(config: Dict) -> Iterable[Path]:
    pic_root = Path(config["dataset"]["pic_path"])
    patch_num = config["dataset"]["patch_num"]
    for name in split_names(config):
        for patch_idx in range(patch_num):
            yield pic_root / f"{name}_part{patch_idx}_1.png"


def projections_ready(config: Dict) -> bool:
    return all(path.exists() for path in required_projection_paths(config))


def patch_paths_ready(config: Dict) -> bool:
    pc_root = Path(config["dataset"]["pc_path"])
    patch_num = config["dataset"]["patch_num"]
    for name in split_names(config):
        for patch_idx in range(patch_num):
            if not (pc_root / f"{name}_part{patch_idx}.ply").exists():
                return False
    return True


def ensure_patch_assets_ready(config: Dict) -> None:
    pc_root = Path(config["dataset"]["pc_path"])
    missing = []
    for name in split_names(config):
        for patch_idx in range(config["dataset"]["patch_num"]):
            target = pc_root / f"{name}_part{patch_idx}.ply"
            if not target.exists():
                missing.append(str(target))
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            "Missing patch point clouds under dataset.pc_path. "
            "Please run `python utils/PCUtils.py` first.\n"
            f"Examples:\n{preview}"
        )


def load_stn_from_checkpoint(ckpt_path: str, device: torch.device, rotation_mode: Optional[str] = None):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if rotation_mode is None:
        rotation_mode = checkpoint.get("hyper_parameters", {}).get("rotation_mode", "matrix")
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean_state = {}
    for key, value in state_dict.items():
        if key.startswith("stn."):
            clean_state[key[len("stn."):]] = value
    if not clean_state:
        clean_state = state_dict
    stn = build_stn(rotation_mode)
    stn.load_state_dict(clean_state, strict=False)
    stn.to(device)
    stn.eval()
    return stn


def projection_backend_name(config: Dict, override: Optional[str] = None) -> str:
    if override:
        return override
    return config["dataset"].get("projection_backend", "pytorch3d_scatter")


@torch.no_grad()
def generate_projection_image(renderer, stn, patch: torch.Tensor, backend: str) -> Image.Image:
    patch = patch.to(next(stn.parameters()).device)
    xyz = patch[:, :3]
    rgb = torch.clamp(patch[:, 3:] / 255.0, 0.0, 1.0)

    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    scale = torch.max(torch.abs(xyz))
    if scale > 0:
        xyz = xyz / scale

    normals = ops.points_normals.estimate_pointcloud_normals(xyz.unsqueeze(0), 64)
    normals = orient_normals_outward(xyz.unsqueeze(0), normals).permute(0, 2, 1)
    transform = stn(normals)
    aligned_xyz = torch.bmm(transform, xyz.unsqueeze(0).permute(0, 2, 1)).permute(0, 2, 1)[0]
    aligned_cloud = torch.cat([aligned_xyz, rgb], dim=1).cpu().numpy()

    if backend == "matplotlib_ref":
        return render_projection_image_reference(aligned_cloud, plane_normal=PLANE_NORMAL_Z)
    if backend == "pytorch3d_scatter":
        return render_projection_image_pytorch3d_scatter(renderer, aligned_xyz, rgb)
    raise ValueError(f"Unsupported projection backend: {backend}")


def run_tnet_stage(
    config: Dict,
    max_epochs: Optional[int] = None,
    limit_batches: Optional[int] = None,
    dev_mode: Optional[bool] = None,
    rotation_mode: Optional[str] = None,
) -> str:
    hyper = config["hyperparameters"]
    rotation_mode = rotation_mode_name(config, rotation_mode)
    train_set, val_set = build_tnet_datasets(config)
    train_loader = build_loader(
        train_set,
        batch_size=hyper["batch_size"],
        shuffle=hyper["shuffle"],
        num_workers=hyper["num_workers"],
        pin_memory=hyper["pin_memory"],
    )
    val_loader = build_loader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=hyper["num_workers"],
        pin_memory=hyper["pin_memory"],
    )

    model = TNetAlignmentModel(
        learning_rate=hyper["optimizer_param"]["learning_rate"],
        weight_decay=hyper["optimizer_param"]["weight_decay"],
        rotation_mode=rotation_mode,
    )
    logger = pl.loggers.TensorBoardLogger("logs", name=tnet_log_name(rotation_mode))
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="loss/val_loss",
        filename=f"wpc10-tnet-{rotation_mode}" + "-{epoch:02d}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )
    early_stop = EarlyStopping(
        monitor="loss/val_loss",
        patience=10,
        min_delta=1e-4,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=trainer_accelerator(),
        devices=trainer_devices(),
        max_epochs=max_epochs if max_epochs is not None else hyper["num_epochs"],
        precision=trainer_precision(config),
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint, early_stop],
        fast_dev_run=hyper["dev_mode"] if dev_mode is None else dev_mode,
        limit_train_batches=hyper["limit_batchs"] if limit_batches is None else limit_batches,
        check_val_every_n_epoch=hyper["val_frequency"],
    )
    trainer.fit(model, train_loader, val_loader)

    saved_ckpt = checkpoint.last_model_path or checkpoint.best_model_path
    if not saved_ckpt:
        existing = latest_tnet_ckpt(rotation_mode)
        if existing:
            return existing
        if dev_mode or hyper["dev_mode"]:
            raise RuntimeError("T-Net dev run finished without a saved checkpoint")
        raise RuntimeError("T-Net training finished without a checkpoint")
    final_path = REPO_ROOT / "checks" / f"wpc10_tnet_{rotation_mode}.ckpt"
    shutil.copy2(saved_ckpt, final_path)
    if rotation_mode == "matrix":
        shutil.copy2(saved_ckpt, REPO_ROOT / "checks" / "wpc10_tnet.ckpt")
    return str(final_path)


def run_prepare_stage(
    config: Dict,
    ckpt_path: Optional[str] = None,
    max_epochs: Optional[int] = None,
    limit_batches: Optional[int] = None,
    dev_mode: Optional[bool] = None,
    projection_backend: Optional[str] = None,
    rotation_mode: Optional[str] = None,
) -> None:
    ensure_patch_assets_ready(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rotation_mode = rotation_mode_name(config, rotation_mode)
    ckpt_path = ckpt_path or latest_tnet_ckpt(rotation_mode)
    if not ckpt_path:
        raise FileNotFoundError(
            "No usable T-Net checkpoint found. "
            "Please train T-Net first, e.g. `python main.py --config ... --stage tnet`"
        )

    stn = load_stn_from_checkpoint(ckpt_path, device, rotation_mode=rotation_mode)
    backend = projection_backend_name(config, projection_backend)
    renderer = build_pytorch3d_scatter_renderer(device) if backend == "pytorch3d_scatter" else None
    pic_root = Path(config["dataset"]["pic_path"])
    ensure_dir(str(pic_root))

    from data.DatasetUtils import load_ply

    generated = 0
    for name in split_names(config):
        for patch_idx in range(config["dataset"]["patch_num"]):
            png_path = pic_root / f"{name}_part{patch_idx}_1.png"
            if png_path.exists():
                continue
            ply_path = Path(config["dataset"]["pc_path"]) / f"{name}_part{patch_idx}.ply"
            patch = torch.tensor(load_ply(str(ply_path)), dtype=torch.float32)
            image = generate_projection_image(renderer, stn, patch, backend)
            image.save(png_path)
            generated += 1
            if generated % 100 == 0:
                print(f"[prepare:{backend}] generated {generated} new projections", flush=True)


def run_train_stage(
    config: Dict,
    max_epochs: Optional[int] = None,
    limit_batches: Optional[int] = None,
    dev_mode: Optional[bool] = None,
) -> None:
    smoke_mode = bool(dev_mode) or (limit_batches is not None and limit_batches <= 1)
    if not smoke_mode and not projections_ready(config):
        raise FileNotFoundError(
            "Missing projection images under dataset.pic_path. "
            "Please run prepare first, e.g. `python main.py --config ... --stage prepare`"
        )

    hyper = config["hyperparameters"]
    train_set, val_set = build_multi_datasets(config)
    train_shuffle = False if smoke_mode else hyper["shuffle"]
    worker_count = 0 if smoke_mode else hyper["num_workers"]
    train_loader = build_loader(
        train_set,
        batch_size=hyper["batch_size"],
        shuffle=train_shuffle,
        num_workers=worker_count,
        pin_memory=hyper["pin_memory"],
    )
    val_loader = build_loader(
        train_set if smoke_mode else val_set,
        batch_size=1,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=hyper["pin_memory"],
    )

    model = MultiModel(
        optimizer_param=hyper["optimizer_param"],
        embedding_size=hyper["model"]["embedding_size"],
        embedding_dim=hyper["model"]["embedding_dim"],
        l_w_embedding=hyper["model"]["l_w_embedding"],
        l_w_commitment=hyper["model"]["l_w_commitment"],
        swin_ckpt_path=str(REPO_ROOT / "checks" / "swin_tiny_patch4_window7_224.pth"),
    )
    logger = pl.loggers.TensorBoardLogger("logs", name="train")
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="acc/vail_srocc",
        filename="wpc10-{epoch:02d}",
        save_top_k=3,
        mode="max",
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator=trainer_accelerator(),
        devices=trainer_devices(),
        max_epochs=max_epochs if max_epochs is not None else hyper["num_epochs"],
        precision=trainer_precision(config),
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint],
        fast_dev_run=hyper["dev_mode"] if dev_mode is None else dev_mode,
        limit_train_batches=hyper["limit_batchs"] if limit_batches is None else limit_batches,
        check_val_every_n_epoch=hyper["val_frequency"],
    )

    ckpt_path = hyper.get("ckpt_path") if hyper.get("continue_train") else None
    if ckpt_path:
        ckpt_path = resolve_path(ckpt_path)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    if not (hyper["dev_mode"] or dev_mode):
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(Path(logger.log_dir) / "config.json", "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/wpc/1/1.json")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "tnet", "prepare", "train"])
    parser.add_argument("--tnet-ckpt", type=str, default=None)
    parser.add_argument("--rotation-mode", type=str, default="quaternion", choices=["matrix", "quaternion"])
    parser.add_argument("--projection-backend", type=str, default="pytorch3d_scatter", choices=["pytorch3d_scatter", "matplotlib_ref"])
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--dev-mode", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_and_precision(config)

    if args.stage == "tnet":
        run_tnet_stage(
            config,
            max_epochs=args.max_epochs,
            limit_batches=args.limit_batches,
            dev_mode=args.dev_mode,
            rotation_mode=args.rotation_mode,
        )
        return
    if args.stage == "prepare":
        run_prepare_stage(
            config,
            ckpt_path=args.tnet_ckpt,
            max_epochs=args.max_epochs,
            limit_batches=args.limit_batches,
            dev_mode=args.dev_mode,
            projection_backend=args.projection_backend,
            rotation_mode=args.rotation_mode,
        )
        return
    if args.stage == "train":
        run_train_stage(
            config,
            max_epochs=args.max_epochs,
            limit_batches=args.limit_batches,
            dev_mode=args.dev_mode,
        )
        return

    raise RuntimeError("Unsupported stage")


if __name__ == "__main__":
    main()
