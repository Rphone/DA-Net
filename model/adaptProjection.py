import torch
import torch.nn as nn


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(dim=1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=1,
    ).view(-1, 3, 3)


class _STN3dBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        return x


class STN3d(_STN3dBase):
    def __init__(self) -> None:
        super().__init__()
        self.fc3 = nn.Linear(256, 9)
        self.register_buffer(
            "iden",
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).view(1, 9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.encode(x)
        x = self.fc3(x)
        x = x + self.iden.repeat(batch_size, 1)
        return x.view(-1, 3, 3)


class STN3dQuaternion(_STN3dBase):
    def __init__(self) -> None:
        super().__init__()
        self.fc3 = nn.Linear(256, 4)
        self.register_buffer("iden_quat", torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.encode(x)
        quat = self.fc3(x) + self.iden_quat.repeat(batch_size, 1)
        return quaternion_to_matrix(quat)


def build_stn(rotation_mode: str = "matrix") -> nn.Module:
    if rotation_mode == "matrix":
        return STN3d()
    if rotation_mode == "quaternion":
        return STN3dQuaternion()
    raise ValueError(f"Unsupported rotation_mode: {rotation_mode}")
