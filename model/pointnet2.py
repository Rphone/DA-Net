import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
import torch.nn as nn
import torch.nn.functional as F
import torch

def square_distance(xyz, center_xyz):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm:
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = center_xyz.shape
    _, M, _ = xyz.shape
    dist = -2 * torch.matmul(center_xyz, xyz.permute(0, 2, 1))   # Matrix multiplication
    dist += torch.sum(center_xyz ** 2, -1).view(B, N, 1)
    dist += torch.sum(xyz ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def query_ball_point(radius, nsample, xyz, center_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = center_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(xyz, center_xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(neighbor, xyz, center_xyz):
    """ KNN algorithm
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(xyz, center_xyz)
    _, group_idx = torch.topk(sqrdists, neighbor, dim = -1, largest=False, sorted=False)
    return group_idx

def furthest_point_sample(xyz, npoint):
    """ FPS algorithm
    batch_size: 
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape     # B=32，N=1024，C=6
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)     # 32*npoint，init
    distance = torch.ones(B, N).to(device) * 1e10                       # 32*1024，Distance matrix initialization
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)   # 24 random points, B number 0~N
    batch_indices = torch.arange(B, dtype=torch.long).to(device)        # 0~31 increasing sequence
    for i in range(npoint):                                             # select npoint point
        centroids[:, i] = farthest                                      # B*npoint, put it in
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)        # Find the farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]                           # Find the farthest point
    return centroids    # reture index of farthest point (1-1024)

def sample_and_group(npoint, neighbor, xyz, feature):
    """ SG layer
    Input:
        npoint:sampling point number
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    feature = feature.permute(0, 2, 1)                             # [32, 1024, 64]，
    B, N, C = xyz.shape        # 
    S = npoint 
    
    xyz = xyz.contiguous()   # deep copy
    # fps_idx = furthest_point_sample(xyz, npoint).long()   # USE Uniform sampling here

    noise = torch.rand(B, N, device=xyz.device)             # noise in [0, 1]     USE random sampling here
    ids_shuffle = torch.argsort(noise, dim=1)               # sorted index
    ids_keep = ids_shuffle[:, :S]                           # Select top
    fps_idx = torch.arange(N, dtype=torch.long).to(xyz.device).unsqueeze(0).repeat(B,1)
    fps_idx = torch.gather(fps_idx, dim=1, index=ids_keep)    # random select some points

    center_xyz = index_points(xyz, fps_idx)                 # index point
    center_feature = index_points(feature, fps_idx)         # index point's feature



    idx = knn_point(neighbor, xyz, center_xyz)              # 32 neighbors found
    grouped_xyz = index_points(xyz, idx)                    # [B, npoint, nsample, C] 
    grouped_xyz_norm = grouped_xyz - center_xyz.view(B, S, 1, C)
    grouped_feature = index_points(feature, idx)           
    grouped_feature_center = grouped_feature - center_feature.view(B, S, 1, -1)     # centerize group feature
    res_points = torch.cat([grouped_feature_center, center_feature.view(B, S, 1, -1).repeat(1, 1, neighbor, 1)], dim=-1)
    return center_xyz, res_points

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()        # ([batchsize, npoints, neighbor, feature])
        x = x.permute(0, 1, 3, 2)    # ([batchsize, npoints, feature, neighbor])
        x = x.reshape(-1, d, s)      # ([batchsize*npoints, feature, neighbor])
        batch_size, _, N = x.size()  # ([batchsize*npoints, feature, neighbor])
        x1 = F.relu(self.bn1(self.conv1(x)))    # ([batchsize*npoints, feature, neighbor])
        x2 = F.relu(self.bn2(self.conv2(x1)))    # ([batchsize*npoints, feature, neighbor])
        x3 = F.adaptive_max_pool1d(x2, 1)        # ([batchsize*npoints, feature, 1 ])
        x4 = x3.view(batch_size, -1)             # ([batchsize*npoints, feature])
        x_res = x4.reshape(b, n, -1).permute(0, 2, 1)
        return x_res                             # ([batchsize, feature, npoints])


class pointnet2(nn.Module):
    def __init__(self, point_num=8192):
        super(pointnet2, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 1024, kernel_size=1, stride=int(point_num//256), bias=False)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        self.conv_fuse1 = nn.Sequential(nn.Conv1d(1280, 512, kernel_size=1, bias=False),
                            nn.BatchNorm1d(512),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.CBR = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                        nn.BatchNorm1d(512),
                        nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):    # 32, 6, 1024
        xyz = x[:,0:3,:].permute(0, 2, 1)                  # get xyz axis
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))                # [B 64, 8192]
        # B, D, N
        x_skip = F.relu(self.bn2(self.conv2(x)))           # [B, 1024, 256]

        new_xyz, new_feature = sample_and_group(npoint=512, neighbor=32, xyz=xyz, feature=x)
        feature_0 = self.gather_local_0(new_feature)       # [B, 128, 512] <= [B, 512, 32, 128]

        new_xyz, new_feature = sample_and_group(npoint=256, neighbor=32, xyz=new_xyz, feature=feature_0)
        feature_1 = self.gather_local_1(new_feature)       # [B, 256, 256] <= [B, 256, 32, 256]

        feature_1 = torch.cat((feature_1, x_skip), dim=1)
        res = self.conv_fuse1(feature_1)
        res=self.CBR(res)
        res = F.adaptive_max_pool1d(res, 1).squeeze()
        return res    # 32, 1

if __name__=='__main__':
    #设置路径使得可以调用pointnet2_utils
    
    
    
    model = pointnet2(8192).cuda()
    xyz = torch.rand(4, 6, 8192).cuda()
    out = model(xyz)
    print(out.shape)
    print('done')
