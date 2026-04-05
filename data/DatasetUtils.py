import os
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_excel(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_ply(path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) * 255.0
    return np.concatenate([points, colors], axis=1)


def get_origin_ply(ply_dir: str) -> List[str]:
    if not os.path.isdir(ply_dir):
        return []
    res: List[str] = []
    for root, dirs, files in os.walk(ply_dir):
        dirs.sort()
        for file_name in sorted(files):
            if file_name.endswith(".ply"):
                res.append(os.path.join(root, file_name))
    return res


def get_wpc10_split_records(mos_path: str) -> List[Tuple[str, float]]:
    mos = read_excel(mos_path)
    return [
        (str(row.Ply_name).replace(".ply", ""), float(row.MOS))
        for row in mos.itertuples(index=False)
    ]


def get_wpc10_multi_path(
    pc_path: str,
    pic_path: str,
    mos_path: str,
    patch_num: int = 32,
) -> List[Tuple[str, List[str], List[str], float]]:
    records = []
    for name, mos in get_wpc10_split_records(mos_path):
        pc_files = [os.path.join(pc_path, f"{name}_part{i}.ply") for i in range(patch_num)]
        pic_files = [os.path.join(pic_path, f"{name}_part{i}_1.png") for i in range(patch_num)]
        records.append((name, pc_files, pic_files, mos))
    return records


def get_wpc10_patch_paths(
    pc_path: str,
    mos_path: str,
    patch_num: int = 32,
) -> List[Tuple[str, List[str], float]]:
    records = []
    for name, mos in get_wpc10_split_records(mos_path):
        pc_files = [os.path.join(pc_path, f"{name}_part{i}.ply") for i in range(patch_num)]
        records.append((name, pc_files, mos))
    return records


def union_split_names(paths: Sequence[str]) -> List[str]:
    merged = []
    seen = set()
    for path in paths:
        for name, _ in get_wpc10_split_records(path):
            if name not in seen:
                seen.add(name)
                merged.append(name)
    return merged


def coor_norm(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    scale = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if scale < 1e-12:
        return pc.astype(np.float32)
    return (pc / scale).astype(np.float32)


def color_norm(colors: np.ndarray) -> np.ndarray:
    return (colors / 255.0).astype(np.float32)


def project_cloud_onto_plane(cloud: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    d = -np.dot(cloud, plane_normal)
    projected_cloud = cloud + d[:, np.newaxis] * plane_normal[np.newaxis, :]
    return projected_cloud[:, :2]


def render_projection_image(
    cloud: np.ndarray,
    plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float32),
    image_size: int = 224,
    point_size: float = 3.0,
) -> Image.Image:
    distances = np.dot(cloud[:, :3], plane_normal)
    order = np.argsort(-distances)
    cloud_location = cloud[order, :3]
    cloud_colour = np.clip(cloud[order, 3:] / 255.0, 0.0, 1.0)
    projected_cloud = project_cloud_onto_plane(cloud_location, plane_normal)

    fig = plt.figure(figsize=[2.24, 2.24], dpi=image_size / 2.24)
    ax = fig.add_subplot(111)
    ax.scatter(projected_cloud[:, 0], projected_cloud[:, 1], s=point_size, c=cloud_colour, marker=",")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return image


def validate_files(paths: Iterable[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing required files:\n{preview}")
