from io import BytesIO
from typing import Dict

import matplotlib
import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PLANE_NORMAL_Z = np.array([0.0, 0.0, 1.0], dtype=np.float32)


def project_cloud_onto_plane(cloud: np.ndarray, plane_normal: np.ndarray = PLANE_NORMAL_Z) -> np.ndarray:
    d = -np.dot(cloud, plane_normal)
    projected_cloud = cloud + d[:, np.newaxis] * plane_normal[np.newaxis, :]
    return projected_cloud[:, :2]


def render_projection_image_reference(
    cloud: np.ndarray,
    plane_normal: np.ndarray = PLANE_NORMAL_Z,
    image_size: int = 224,
    point_size: float = 3.0,
) -> Image.Image:
    distances = np.dot(cloud[:, :3], plane_normal)
    order = np.argsort(-distances)
    cloud_location = cloud[order, :3]
    cloud_colour = np.clip(cloud[order, 3:] / 255.0, 0.0, 1.0)
    projected_cloud = project_cloud_onto_plane(cloud_location, plane_normal)

    fig = plt.figure(figsize=[2.24, 2.24], dpi=image_size / 2.24, facecolor="white")
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.scatter(projected_cloud[:, 0], projected_cloud[:, 1], s=point_size, c=cloud_colour, marker=",")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.BILINEAR)
    return image


def build_pytorch3d_scatter_renderer(
    device: torch.device,
    image_size: int = 224,
    radius: float = 0.015,
) -> Dict:
    R, T = look_at_view_transform(2.0, 0.0, 0.0)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=1,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    return {
        "type": "pytorch3d_scatter",
        "rasterizer": rasterizer,
        "image_size": image_size,
    }


@torch.no_grad()
def render_projection_image_pytorch3d_scatter(renderer: Dict, xyz: torch.Tensor, rgb: torch.Tensor) -> Image.Image:
    point_cloud = Pointclouds(points=[xyz], features=[rgb])
    fragments = renderer["rasterizer"](point_cloud)
    point_indices = fragments.idx[0, ..., 0]

    image = torch.ones((renderer["image_size"], renderer["image_size"], 3), device=xyz.device, dtype=torch.float32)
    mask = point_indices >= 0
    if mask.any():
        image[mask] = rgb[point_indices[mask]]

    image = torch.flip(image, dims=[0])
    image = (image.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
    return Image.fromarray(image)
