# DA-Net: A Double Alignment Multimodal Learning Network for Point Cloud Quality Assessment

This repository is the official implementation of our paper ["DA-Net: A Double Alignment Multimodal Learning Network for Point Cloud Quality Assessment", published in IEEE Transactions on Image Processing (TIP) 2025.](https://ieeexplore.ieee.org/document/11291159/)

## Environments
```bash
conda create -n danet python=3.9 -y
conda activate danet

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install pytorch-lightning==2.2.1 timm==1.0.3 pandas==2.3.3 openpyxl pillow matplotlib scipy tensorboard vector-quantize-pytorch==1.14.24

#need to downgrade numpy for PyTorch3D compatibility
conda install "numpy<2" -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
pip install open3d==0.19.0
```

## Prepare the datasets

- The default release config uses:
  - split metadata: `configs/wpc/1/train.xlsx`, `configs/wpc/1/vail.xlsx`
  - patch point clouds: `tmps/wpc10/pc`
  - aligned projections: `tmps/wpc10/proj`
  - weights: `checks`

Example: if you want to use your own split files and output directories, you can edit `configs/wpc/1/1.json` like this:
```json
{
  "dataset": {
    "traininfo": "configs/wpc/1/train.xlsx",
    "vailinfo": "configs/wpc/1/vail.xlsx",
    "pc_path": "/path/to/your/32x8192_patches",
    "pic_path": "/path/to/your/generated_projections",
    "patch_num": 32,
    "train_patch_num": 8,
    "vail_patch_num": 32,
    "projection_num": 1
  }
}
```


### Download the preprocessed wpc datasets and pretrained checkpoints

#### [BaiduNetDisk](https://pan.baidu.com/s/1qXzm8nF_H5ABftAJzsD1sw?pwd=wwd3)
#### [OneDrive](https://1drv.ms/f/c/6900cdc80191473b/IgBpMxxEKdAjT4V1KnnltpHQAZpYLgHz0ck6ZBPY04-dmPQ)

**put swin_tiny_patch4_window7_224.pth under `checks/` for the pretrained Swin-T backbone.**

Downloaded split files, patch point clouds, and projections should also be placed under the paths specified by your config file, then run:

```bash
 python main.py --config configs/wpc/1/1.json --stage train
```


### Create the dataset by yourself (optional)
Tips: Also need download the swin_tiny_patch4_window7_224.pth pretrained Swin-T backbone and put it under `checks/`.



Step 1: generate patch point clouds from raw point clouds:
```bash
python utils/PCUtils.py --origin-path /path/to/raw_point_clouds --save-path /path/to/your/32x8192_patches
```
`pc_path` should contain `32x8192` patch point clouds named as:
```text
<sample_name>_part0.ply
...
<sample_name>_part31.ply
```
Step 2: train the PCN (T-Net) checkpoint:
```bash
 python main.py --config configs/wpc/1/1.json --stage tnet
```

Optional: train T-Net with a specific rotation parameterization:
```bash
 python main.py --config configs/wpc/1/1.json --stage tnet --rotation-mode matrix
 python main.py --config configs/wpc/1/1.json --stage tnet --rotation-mode quaternion (will gain better performance)
```

Step 3: generate aligned projections with a trained T-Net checkpoint:
```bash
 python main.py --config configs/wpc/1/1.json --stage prepare
```

Important: `prepare` must run after Step 2, because it needs an existing T-Net checkpoint.

If `--tnet-ckpt` is not provided, `prepare` will pick checkpoint by priority:

- When `--rotation-mode quaternion` (default):
  1) `checks/wpc10_tnet_quaternion.ckpt`
  2) latest checkpoint under `logs/tnet_quaternion/version_*/checkpoints/*.ckpt`

- When `--rotation-mode matrix`:
  1) `checks/wpc10_tnet_matrix.ckpt`
  2) `checks/wpc10_tnet.ckpt`
  3) latest checkpoint under `logs/tnet/version_*/checkpoints/*.ckpt`

Projection backend options:
```bash
 python main.py --config configs/wpc/1/1.json --stage prepare --projection-backend pytorch3d_scatter
 python main.py --config configs/wpc/1/1.json --stage prepare --projection-backend matplotlib_ref
```

Generate projections with an explicit PCN checkpoint and rotation mode:
```bash
 python main.py --config configs/wpc/1/1.json --stage prepare --tnet-ckpt checks/wpc10_tnet_matrix.ckpt --rotation-mode matrix --projection-backend pytorch3d_scatter
 python main.py --config configs/wpc/1/1.json --stage prepare --tnet-ckpt checks/wpc10_tnet_quaternion.ckpt --rotation-mode quaternion --projection-backend matplotlib_ref
```

The `prepare` stage writes aligned single-view projections to `dataset.pic_path` as:
```text
<sample_name>_part0_1.png
...
<sample_name>_part31_1.png
```

Step 4: train main model for quality regression:
```bash
 python main.py --config configs/wpc/1/1.json --stage train
```





## Acknowledgements
The following repository contributed to the development of this codebase, and we thank the authors for their open-source release:

- [MM-PCQA](https://github.com/zzc-1998/MM-PCQA)
- [3DTA-PCQA](https://github.com/philox12358/3DTA-PCQA)
- [COPP-Net](https://github.com/philox12358/COPP-Net)

## Citation
If you find this code useful for your research, please consider citing our paper:
```
@article{wu2025net,
  title={DA-Net: A Double Alignment Multimodal Learning Network for Point Cloud Quality Assessment},
  author={Wu, Xinqiang and He, Zhouyan and Luo, Ting and Jiang, Gangyi and Zhou, Wujie and Zhu, Linwei and Lin, Weisi},
  journal={IEEE Transactions on Image Processing},
  volume={34},
  pages={8185--8200},
  year={2025},
  publisher={IEEE}
}
```
