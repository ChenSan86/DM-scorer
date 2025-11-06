#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试评估器网络（Scorer Network）的数据流
"""

import sys
import torch
import numpy as np
from argparse import Namespace

# 添加项目路径
sys.path.insert(0, '.')

def test_scorer_network():
    """测试评估器网络"""
    print("="*80)
    print("评估器网络（Scorer Network）数据流测试")
    print("="*80)
    
    # 1. 测试模型创建
    print("\n[1] 测试模型创建...")
    try:
        # 直接导入，避免通过ocnn导入触发其他模块
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ocnn/models'))
        from scorer_net import ScorerNet
        
        model = ScorerNet(
            in_channels=4,
            geo_channels=256,
            rot_channels=128,
            tool_channels=64,
            interp='linear',
            nempty=False
        )
        print("   ✓ ScorerNet 模型创建成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   总参数量: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        
    except Exception as e:
        print(f"   ✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 测试前向传播（模拟数据）
    print("\n[2] 测试前向传播...")
    try:
        import ocnn
        from ocnn.octree import Octree, Points
        
        B = 2  # batch size
        N = 1024  # 每个样本的点数
        
        # 创建模拟点云
        xyz = torch.randn(B * N, 3) * 0.5  # 范围在 [-1, 1]
        normals = torch.randn(B * N, 3)
        normals = normals / normals.norm(dim=1, keepdim=True)
        
        points_list = []
        for i in range(B):
            pts = Points(
                xyz[i*N:(i+1)*N],
                normals[i*N:(i+1)*N]
            )
            points_list.append(pts)
        
        # 构建octree
        octrees = []
        for pts in points_list:
            octree = Octree(depth=5, full_depth=2)
            octree.build_octree(pts)
            octrees.append(octree)
        
        # 合并octree
        merged_octree = ocnn.octree.merge_octrees(octrees)
        merged_octree.construct_all_neigh()
        
        # 合并points
        merged_points = ocnn.octree.merge_points(points_list)
        
        print(f"   Octree构建成功:")
        print(f"     - 节点数: {merged_octree.nnum}")
        print(f"     - 点数: {merged_points.points.shape[0]}")
        
        # 提取特征
        octree_feature = ocnn.modules.InputFeature('ND', nempty=False)
        data = octree_feature(merged_octree)  # [N_nodes, 4]
        print(f"   ✓ 输入特征形状: {data.shape}")
        
        # 构造query_pts
        query_pts = torch.cat([merged_points.points, merged_points.batch_id], dim=1)
        print(f"   ✓ Query points形状: {query_pts.shape}")
        
        # 创建模拟旋转矩阵 [B, 3, 3]
        rotation_matrix = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
        # 添加小扰动
        rotation_matrix = rotation_matrix + torch.randn(B, 3, 3) * 0.1
        # 正交化（简化版）
        U, _, V = torch.svd(rotation_matrix)
        rotation_matrix = torch.matmul(U, V.transpose(1, 2))
        print(f"   ✓ 旋转矩阵形状: {rotation_matrix.shape}")
        
        # 创建刀具参数 [B, 4]
        tool_params = torch.randn(B, 4)
        print(f"   ✓ 刀具参数形状: {tool_params.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            scores = model(
                data,
                merged_octree,
                merged_octree.depth,
                query_pts,
                rotation_matrix,
                tool_params
            )
        
        print(f"   ✓ 输出分数形状: {scores.shape}")
        print(f"   ✓ 分数范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
        
    except Exception as e:
        print(f"   ✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试数据加载器
    print("\n[3] 测试数据加载器...")
    try:
        from datasets import get_seg_shapenet_dataset
        
        # 使用dict而不是Namespace，因为Transform需要**kwargs
        flags = type('Flags', (), {
            'location': 'data_2.0/points',
            'filelist': 'data_2.0/filelist/models_test.txt',
            'depth': 5,
            'full_depth': 2,
            'distort': False,
            'angle': (0, 0, 0),
            'interval': (1, 1, 1),
            'scale': 0.0,
            'uniform': True,
            'jitter': 0.0,
            'flip': (0, 0, 0),
            'orient_normal': 'xyz',
            'batch_size': 2,
            'shuffle': False,
            'num_workers': 0,
            'pin_memory': False,
            'take': 5,
            'feature': 'ND',
            'nempty': False,
            # 添加__dict__方法以支持**展开
            '__dict__': property(lambda self: {
                'depth': 5,
                'full_depth': 2,
                'distort': False,
                'angle': (0, 0, 0),
                'interval': (1, 1, 1),
                'scale': 0.0,
                'uniform': True,
                'jitter': 0.0,
                'flip': (0, 0, 0),
                'orient_normal': 'xyz',
            })
        })()
        
        dataset, collate_fn = get_seg_shapenet_dataset(flags)
        print(f"   ✓ 数据集大小: {len(dataset)}")
        
        # 测试单个样本
        sample = dataset[0]
        print(f"   ✓ 样本字段: {list(sample.keys())}")
        
        # 检查关键字段
        if 'rotation_matrices' in sample:
            print(f"   ✓ rotation_matrices 形状: {sample['rotation_matrices'].shape}")
        else:
            print("   ✗ 缺少 rotation_matrices 字段!")
            return False
        
        if 'labels' in sample:
            print(f"   ✓ labels 形状: {sample['labels'].shape}")
        else:
            print("   ✗ 缺少 labels 字段!")
            return False
        
        # 测试DataLoader
        from torch.utils.data import DataLoader
        from thsolver.sampler import InfSampler
        
        sampler = InfSampler(dataset, shuffle=False)
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print(f"   ✓ Batch 字段: {list(batch.keys())}")
        print(f"   ✓ Batch labels 形状: {batch['labels'].shape}")
        print(f"   ✓ Batch rotation_matrices 形状: {batch['rotation_matrices'].shape}")
        
    except Exception as e:
        print(f"   ✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试 ScorerSolver
    print("\n[4] 测试 ScorerSolver...")
    try:
        from scorer_solver import ScorerSolver
        print("   ✓ ScorerSolver 导入成功")
        
        # 测试采样函数
        B = 8
        rot_indices = torch.randint(0, 338, (B,))
        print(f"   ✓ 采样旋转索引: {rot_indices.tolist()}")
        
        # 测试从labels中提取GT分数
        labels = torch.rand(B, 338)  # 模拟labels
        score_gt = labels[torch.arange(B), rot_indices]
        print(f"   ✓ GT分数形状: {score_gt.shape}")
        
    except Exception as e:
        print(f"   ✗ ScorerSolver 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结
    print("\n" + "="*80)
    print("✓ 所有测试通过！评估器网络准备就绪")
    print("="*80)
    print("\n下一步:")
    print("  1. 运行快速训练测试:")
    print("     cd projects")
    print("     python run_scorer_deepmill.py --gpu 0 --ratios 0.01")
    print("")
    print("  2. 运行完整训练:")
    print("     cd projects")
    print("     python run_scorer_deepmill.py --gpu 0 --ratios 1.0")
    print("="*80)
    
    return True


if __name__ == '__main__':
    try:
        success = test_scorer_network()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

