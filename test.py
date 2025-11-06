# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import torch
import numpy as np
import json
from tqdm import tqdm


def _angles_to_rotation_matrix(pitch, roll):
    # 转换为张量
    pitch_rad = torch.tensor(pitch * (torch.pi / 180.0))
    roll_rad = torch.tensor(roll * (torch.pi / 180.0))

    # Rx(pitch) - 绕 x 轴旋转
    cr, sr = torch.cos(pitch_rad), torch.sin(pitch_rad)
    cp, sp = torch.cos(roll_rad), torch.sin(roll_rad)
    Rx = torch.tensor([
        [1,   0,    0],
        [0,  cp,  -sp],
        [0,  sp,   cp]
    ], dtype=torch.float32)

    # Ry(roll) - 绕 y 轴旋转

    Ry = torch.tensor([
        [cr,  0,  sr],
        [0,  1,   0],
        [-sr,  0,  cr]
    ], dtype=torch.float32)

    R = torch.mm(Ry, Rx)

    return R


print(_angles_to_rotation_matrix(-90, -180))
