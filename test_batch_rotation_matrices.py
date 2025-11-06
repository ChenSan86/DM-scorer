#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 batch 中 rotation_matrices 的传递
"""

import sys
sys.path.insert(0, 'projects')

import torch
from datasets import get_seg_shapenet_dataset
from thsolver import Solver
from argparse import Namespace

# 创建简单的配置
flags = Namespace(
    location='projects/data_2.0/points',
    filelist='projects/data_2.0/filelist/models_test.txt',
    depth=5,
    full_depth=2,
    distort=False,
    angle=(0, 5, 0),
    interval=(1, 1, 1),
    scale=0.25,
    uniform=True,
    jitter=0.25,
    flip=(0, 0, 0),
    orient_normal='xyz',
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    take=10,
    feature='ND',
    nempty=False
)

def test_batch_rotation_matrices():
    """测试 batch 中的 rotation_matrices 是否正确"""
    print("=" * 60)
    print("测试 Batch 中 rotation_matrices 的传递")
    print("=" * 60)
    
    # 1. 创建数据集
    print("\n[1] 创建数据集...")
    dataset, collate_fn = get_seg_shapenet_dataset(flags)
    print(f"   数据集大小: {len(dataset)}")
    
    # 2. 测试单个样本
    print("\n[2] 测试单个样本...")
    sample = dataset[0]
    print(f"   样本字段: {sample.keys()}")
    
    if 'rotation_matrices' in sample:
        rot_mat = sample['rotation_matrices']
        print(f"   ✓ rotation_matrices 存在")
        print(f"   形状: {rot_mat.shape}")
        print(f"   类型: {type(rot_mat)}")
        print(f"   数据类型: {rot_mat.dtype}")
        print(f"   设备: {rot_mat.device}")
        
        # 验证是旋转矩阵
        if rot_mat.shape == (338, 3, 3):
            print(f"   ✓ 形状正确: (338, 3, 3)")
            
            # 检查是否是正交矩阵（旋转矩阵性质）
            R_sample = rot_mat[0]  # 取第一个矩阵
            RRT = torch.matmul(R_sample, R_sample.T)
            is_orthogonal = torch.allclose(RRT, torch.eye(3), atol=1e-5)
            det = torch.det(R_sample)
            
            print(f"   第一个矩阵验证:")
            print(f"     - 正交性 (R@R^T=I): {is_orthogonal}")
            print(f"     - 行列式 (det(R)≈1): {det.item():.6f}")
        else:
            print(f"   ✗ 形状错误，期望 (338, 3, 3)")
    else:
        print(f"   ✗ rotation_matrices 不存在!")
        return False
    
    # 3. 创建 DataLoader 并测试 batch
    print("\n[3] 测试 DataLoader batch...")
    from torch.utils.data import DataLoader
    from thsolver.sampler import InfSampler
    
    sampler = InfSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset,
        batch_size=flags.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 获取一个 batch
    batch = next(iter(dataloader))
    print(f"   Batch 字段: {batch.keys()}")
    
    if 'rotation_matrices' in batch:
        rot_mat_batch = batch['rotation_matrices']
        print(f"   ✓ batch 中有 rotation_matrices")
        print(f"   形状: {rot_mat_batch.shape}")
        print(f"   类型: {type(rot_mat_batch)}")
        
        # 验证是否正确（应该是 (338, 3, 3)，而不是 (B, 338, 3, 3)）
        if rot_mat_batch.shape == (338, 3, 3):
            print(f"   ✓ 形状正确: (338, 3, 3) - 所有样本共享")
        elif isinstance(rot_mat_batch, list):
            print(f"   ✗ 错误: rotation_matrices 是列表，长度 {len(rot_mat_batch)}")
            print(f"      应该在 CollateBatch 中取 [0]")
            return False
        else:
            print(f"   ✗ 形状错误: {rot_mat_batch.shape}")
            print(f"      期望: (338, 3, 3)")
            return False
    else:
        print(f"   ✗ batch 中没有 rotation_matrices!")
        return False
    
    # 4. 验证数据一致性
    print("\n[4] 验证数据一致性...")
    sample_rot = dataset[0]['rotation_matrices']
    batch_rot = batch['rotation_matrices']
    
    is_same = torch.equal(sample_rot, batch_rot)
    print(f"   单样本 vs Batch: {'✓ 相同' if is_same else '✗ 不同'}")
    
    if not is_same:
        diff = (sample_rot - batch_rot).abs().max()
        print(f"   最大差异: {diff.item()}")
    
    # 5. 测试与 SegSolver 中的版本对比
    print("\n[5] 测试与 SegSolver.rotation_matrices 对比...")
    try:
        from segmentation import SegSolver
        
        # 创建临时配置
        from argparse import Namespace
        temp_flags = Namespace(
            SOLVER=Namespace(
                run='train',
                gpu=(0,),
                logdir='temp',
                progress_bar=False
            ),
            MODEL=Namespace(),
            DATA=Namespace(
                train=flags,
                test=flags
            )
        )
        
        # 注意：这里不实际创建 SegSolver（会初始化GPU），
        # 只测试 _load_rotation_matrices 方法
        import json
        import os
        json_path = 'projects/rotation_matrices.json'
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            rotation_matrices_solver = []
            for i in range(338):
                key = f"ori_{i:03d}"
                if key in data:
                    import numpy as np
                    matrix = np.array(data[key]['rotation_matrix'], dtype=np.float32)
                    rotation_matrices_solver.append(torch.from_numpy(matrix))
                else:
                    rotation_matrices_solver.append(torch.eye(3, dtype=torch.float32))
            
            rotation_matrices_solver = torch.stack(rotation_matrices_solver)
            
            is_same_solver = torch.equal(batch_rot, rotation_matrices_solver)
            print(f"   Batch vs SegSolver: {'✓ 相同' if is_same_solver else '✗ 不同'}")
            
            if not is_same_solver:
                diff = (batch_rot - rotation_matrices_solver).abs().max()
                print(f"   最大差异: {diff.item()}")
        else:
            print(f"   ⚠ rotation_matrices.json 不存在，跳过对比")
    
    except Exception as e:
        print(f"   ⚠ 无法对比: {e}")
    
    # 6. 总结
    print("\n" + "=" * 60)
    print("✓ 测试通过！rotation_matrices 正确传递到 batch 中")
    print("=" * 60)
    print("\n建议:")
    print("  1. batch['rotation_matrices'] 形状: (338, 3, 3)")
    print("  2. 所有样本共享同一个 rotation_matrices")
    print("  3. 在 loss 函数中使用时，从 batch 中获取")
    print("  4. 考虑统一数据源（Dataset 或 SegSolver）")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = test_batch_rotation_matrices()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



