# [Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos](https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf)
![](https://github.com/hehefan/P4Transformer/blob/main/imgs/framework.png)

## Introduction
Point cloud videos exhibit irregularities and lack of order along the spatial dimension where points emerge inconsistently across different frames. To capture the dynamics in point cloud videos, point tracking is usually employed. However, as points may flow in and out across frames, computing accurate point trajectories is extremely difficult. 
Moreover, tracking usually relies on point colors and thus may fail to handle colorless point clouds. In this paper, to avoid point tracking, we propose a novel Point 4D Transformer (P4Transformer) network to model raw point cloud videos. Specifically, P4Transformer consists of (i) a point 4D convolution to embed the spatio-temporal local structures presented in a point cloud video and (ii) a transformer to capture the appearance and motion information across the entire video by performing self-attention on the embedded local features. In this fashion, related or similar local areas are merged with attention weight rather than by explicit tracking. 

## Installation

The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 8.3.1, PyTorch (both v1.4.0 and v1.8.1 are supported), CUDA 10.2 and cuDNN v7.6.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
mv modules-pytorch-1.4.0/modules-pytorch-1.8.1 modules
cd modules
python setup.py install
```

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{fan21p4transformer,
  author    = {Hehe Fan and
               Yi Yang and
               Mohan Kankanhalli},
  title     = {Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos},
  booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition, {CVPR}},
  year      = {2021}
}
```

## Related Repos
1. PointNet++ PyTorch implementation: https://github.com/facebookresearch/votenet/tree/master/pointnet2
2. MeteorNet: https://github.com/xingyul/meteornet
3. 3DV: https://github.com/3huo/3DV-Action
4. PSTNet: https://github.com/hehefan/Point-Spatio-Temporal-Convolution
5. Transformer: https://github.com/lucidrains/vit-pytorch
6. PointRNN (TensorFlow implementation): https://github.com/hehefan/PointRNN
7. PointRNN (PyTorch implementation): https://github.com/hehefan/PointRNN-PyTorch
8. Awesome Dynamic Point Cloud / Point Cloud Video / Point Cloud Sequence / 4D Point Cloud Analysis: https://github.com/hehefan/Awesome-Dynamic-Point-Cloud-Analysis
