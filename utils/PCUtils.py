import argparse
import os
import sys
import multiprocessing as mp
from multiprocessing import Pool
from time import time

import numpy as np
import open3d as o3d
import pandas as pd
import torch

if __package__ in (None, ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import data.DatasetUtils as datautils


def parse_args():
    parser = argparse.ArgumentParser(description="Generate patch point clouds for DA-Net")
    parser.add_argument(
        "--origin-path",
        type=str,
        default="/root/hot-datas/wuxinqiang/kywxq/datasets/wpc10/origin/",
        help="Directory containing raw .ply files",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="tmps/wpc10/pc/",
        help="Output directory for generated patch .ply files",
    )
    parser.add_argument("--patch-size", type=int, default=8192, help="Points per patch")
    parser.add_argument("--patch-num", type=int, default=32, help="Number of patches per sample")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker processes for patch generation (set 1 for serial mode)",
    )
    parser.add_argument(
        "--mp-start-method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method",
    )
    return parser.parse_args()


def _calc_distances_torch(p0, points):
    return ((p0 - points) ** 2).sum(dim=1)


def fps_samples_torch(pts, k):
    farthest_pts_indexes = torch.zeros(k, dtype=torch.int32)
    farthest_pts_indexes[0] = torch.randint(len(pts), (1,))
    distances = _calc_distances_torch(pts[farthest_pts_indexes[0]], pts)
    for i in range(1, k):
        farthest_pts_indexes[i] = torch.argmax(distances)
        distances = torch.min(distances, _calc_distances_torch(pts[farthest_pts_indexes[i]], pts))
    return farthest_pts_indexes

def pad_to_size(npy, pad_size=8192):
    # 获取原始张量的大小
    size = npy.shape

    # 检查是否需要补全
    if size[0] < pad_size:
        # 计算补全的倍数
        repeat_factor = pad_size // size[0]

        # 使用 tile 函数进行补全
        padded_array = np.tile(npy, (repeat_factor, 1))

        # 判断是否还需要额外的补全
        remaining_size = pad_size - padded_array.shape[0]
        if remaining_size > 0:
            # 切片原始张量，将剩余部分补充到新的张量中
            remaining_part = npy[:remaining_size, :]
            padded_array = np.concatenate((padded_array, remaining_part), axis=0)

        return padded_array
    else:
        # 如果已经足够大，直接返回原始张量
        return npy

def split_PC2patch(paths, save_path, patch_size, path_num, workers=None, start_method=None):
    os.makedirs(save_path, exist_ok=True)
    items = list(paths)
    total = len(items)
    if total == 0:
        print(f"[split_PC2patch] no input point clouds, skip. save_path={save_path}", flush=True)
        return

    if workers is None:
        workers = int(os.environ.get("DANET_PC_WORKERS", "16"))
    if start_method is None:
        start_method = os.environ.get("DANET_MP_START_METHOD", "spawn")

    cpu_total = os.cpu_count() or 1
    worker_count = max(1, min(int(workers), cpu_total))
    progress = {"done": 0, "ok": 0, "failed": 0}
    started = time()

    print(
        f"[split_PC2patch] start: samples={total}, workers={worker_count}, patch_size={patch_size}, path_num={path_num}",
        flush=True,
    )

    def _on_result(result):
        name, success, message = result
        progress["done"] += 1
        if success:
            progress["ok"] += 1
        else:
            progress["failed"] += 1
            print(f"[split_PC2patch] failed: name={name}, reason={message}", flush=True)

        if progress["done"] % 10 == 0 or progress["done"] == total:
            print(
                f"[split_PC2patch] progress: {progress['done']}/{total} (ok={progress['ok']}, failed={progress['failed']})",
                flush=True,
            )

    def _on_error(exc):
        progress["done"] += 1
        progress["failed"] += 1
        print(f"[split_PC2patch] worker exception: {repr(exc)}", flush=True)
        if progress["done"] % 10 == 0 or progress["done"] == total:
            print(
                f"[split_PC2patch] progress: {progress['done']}/{total} (ok={progress['ok']}, failed={progress['failed']})",
                flush=True,
            )

    if worker_count <= 1:
        print("[split_PC2patch] running in serial mode (workers=1)", flush=True)
        for item in items:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                name, path = item[0], item[1]
            else:
                path = item
                name = os.path.splitext(os.path.basename(path))[0]
            _on_result(process(path, save_path, patch_size, path_num, name))
    else:
        print(f"[split_PC2patch] creating process pool with start_method={start_method}", flush=True)
        ctx = mp.get_context(start_method)
        with ctx.Pool(worker_count, maxtasksperchild=8) as pool:
            for item in items:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    name, path = item[0], item[1]
                else:
                    path = item
                    name = os.path.splitext(os.path.basename(path))[0]

                pool.apply_async(
                    process,
                    args=(path, save_path, patch_size, path_num, name),
                    callback=_on_result,
                    error_callback=_on_error,
                )
            pool.close()
            pool.join()

    elapsed = time() - started
    print(
        f"[split_PC2patch] done in {elapsed:.1f}s: total={total}, ok={progress['ok']}, failed={progress['failed']}",
        flush=True,
    )

def process(path, save_path, patch_size, path_num, name):
    try:
        pcd = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        fps_index=fps_samples_torch(torch.from_numpy(np.asarray(pcd.points)),path_num)
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        for cnt, i in enumerate(fps_index):
            [k, idx, _] = kd_tree.search_knn_vector_3d(pcd.points[i], patch_size)
            coords = xyz[idx]
            rgb = colors[idx]
            res=pad_to_size(np.concatenate((coords,rgb),axis=-1),patch_size)
            sub_pcd=o3d.geometry.PointCloud()
            sub_pcd.points=o3d.utility.Vector3dVector(res[:,:3])
            sub_pcd.colors=o3d.utility.Vector3dVector(res[:,3:])
            save=os.path.join(save_path,"{}_part{}.ply".format(name,cnt))
            o3d.io.write_point_cloud(save, sub_pcd, write_ascii=True)
        return name, True, ""
    except Exception as exc:
        return name, False, f"{type(exc).__name__}: {exc}"
            
            

if __name__ == "__main__":
    args = parse_args()
    paths = datautils.get_origin_ply(args.origin_path)
    split_PC2patch(
        paths,
        args.save_path,
        args.patch_size,
        args.patch_num,
        workers=args.workers,
        start_method=args.mp_start_method,
    )
    
