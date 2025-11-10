#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估器网络推理脚本
用于对给定的点云、姿态和刀具参数进行评分预测
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from plyfile import PlyData
import ocnn
from ocnn.octree import Octree, Points
from ocnn.models.scorer_net import ScorerNet


class ScorerInference:
    """评估器网络推理类"""
    
    def __init__(
        self,
        model_path: str,
        depth: int = 5,
        full_depth: int = 2,
        feature: str = 'ND',
        nempty: bool = False,
        geo_channels: int = 256,
        rot_channels: int = 128,
        tool_channels: int = 64,
        device: str = 'cuda'
    ):
        """
        初始化推理类
        
        Args:
            model_path: 模型权重文件路径（.pth文件）
            depth: 八叉树深度
            full_depth: 全填充深度
            feature: 特征类型（'ND'表示Normal+Depth）
            nempty: 是否包含空节点
            geo_channels: 几何特征维度
            rot_channels: 旋转特征维度
            tool_channels: 刀具特征维度
            device: 设备（'cuda'或'cpu'）
        """
        self.depth = depth
        self.full_depth = full_depth
        self.feature = feature
        self.nempty = nempty
        self.device = device
        
        # 创建模型
        self.model = ScorerNet(
            in_channels=4,
            geo_channels=geo_channels,
            rot_channels=rot_channels,
            tool_channels=tool_channels,
            interp='linear',
            nempty=nempty
        )
        
        # 加载模型权重
        self._load_model(model_path)
        
        # 设置模型为评估模式
        self.model.eval()
        self.model.to(device)
        
        # 创建输入特征提取器
        self.input_feature = ocnn.modules.InputFeature(feature, nempty)
        
        print(f"✓ 模型加载成功: {model_path}")
        print(f"✓ 设备: {device}")
        print(f"✓ 模型参数: depth={depth}, full_depth={full_depth}, feature={feature}")
    
    def _load_model(self, model_path: str):
        """加载模型权重"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict) and 'model_dict' in checkpoint:
            state_dict = checkpoint['model_dict']
        else:
            state_dict = checkpoint
        
        # 加载权重（处理可能的键名不匹配）
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        
        if len(pretrained_dict) == 0:
            raise ValueError("模型权重不匹配，请检查模型架构")
        
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        
        print(f"✓ 成功加载 {len(pretrained_dict)}/{len(model_dict)} 个参数")
    
    def load_point_cloud(self, ply_path: str):
        """
        加载点云文件
        
        Args:
            ply_path: PLY文件路径
            
        Returns:
            points: Points对象
            octree: Octree对象
        """
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"点云文件不存在: {ply_path}")
        
        # 读取PLY文件
        plydata = PlyData.read(ply_path)
        vtx = plydata['vertex']
        
        # 提取点坐标和法线
        xyz = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1).astype(np.float32)
        normals = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1).astype(np.float32)
        
        # 创建Points对象
        points = Points(
            torch.from_numpy(xyz),
            torch.from_numpy(normals)
        )
        
        # 裁剪到[-1, 1]范围（octree要求）
        points.clip(min=-1, max=1)
        
        # 构建octree
        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        octree.construct_all_neigh()
        
        print(f"✓ 点云加载成功: {ply_path}")
        print(f"  - 点数: {len(xyz)}")
        print(f"  - Octree节点数: {octree.nnum}")
        
        return points, octree
    
    def predict(
        self,
        points: Points,
        octree: Octree,
        euler_angles: np.ndarray,
        tool_params: np.ndarray
    ):
        """
        进行推理预测
        
        Args:
            points: Points对象
            octree: Octree对象
            euler_angles: 欧拉角 [B, 2] 或 [2]，(pitch, roll) 单位：度
            tool_params: 刀具参数 [B, 4] 或 [4]
            
        Returns:
            scores: 预测分数 [B]，数值越小表示质量越好
        """
        # 转换为tensor
        if isinstance(euler_angles, np.ndarray):
            euler_angles = torch.from_numpy(euler_angles).float()
        if isinstance(tool_params, np.ndarray):
            tool_params = torch.from_numpy(tool_params).float()
        
        # 处理单样本情况
        if euler_angles.dim() == 1:
            euler_angles = euler_angles.unsqueeze(0)  # [1, 2]
        if tool_params.dim() == 1:
            tool_params = tool_params.unsqueeze(0)  # [1, 4]
        
        B = euler_angles.size(0)
        
        # 移动到GPU
        octree = octree.to(self.device)
        points_xyz = points.points.to(self.device)
        points_normals = points.normals.to(self.device)
        euler_angles = euler_angles.to(self.device)
        tool_params = tool_params.to(self.device)
        
        # 提取输入特征
        data = self.input_feature(octree)  # [N_nodes, C_in]
        
        # 构造查询点 [N_pts, 4]
        # 对于单样本推理，batch_id全为0
        N_pts = points_xyz.size(0)
        batch_id = torch.zeros(N_pts, 1, device=self.device, dtype=torch.float32)
        query_pts = torch.cat([points_xyz, batch_id], dim=1)
        
        # 推理
        with torch.no_grad():
            scores = self.model(
                data,
                octree,
                octree.depth,
                query_pts,
                euler_angles,
                tool_params
            )  # [B]
        
        return scores.cpu().numpy()
    
    def predict_from_file(
        self,
        ply_path: str,
        euler_angles: np.ndarray,
        tool_params: np.ndarray
    ):
        """
        从文件加载点云并推理
        
        Args:
            ply_path: PLY文件路径
            euler_angles: 欧拉角 [B, 2] 或 [2]，(pitch, roll) 单位：度
            tool_params: 刀具参数 [B, 4] 或 [4]
            
        Returns:
            scores: 预测分数 [B]
        """
        points, octree = self.load_point_cloud(ply_path)
        return self.predict(points, octree, euler_angles, tool_params)


def load_euler_angles(euler_file: str = None):
    """
    加载欧拉角对文件
    
    Args:
        euler_file: 欧拉角文件路径，如果为None则使用默认路径
        
    Returns:
        euler_angles: [338, 2] 欧拉角对数组
    """
    if euler_file is None:
        euler_file = os.path.join(
            os.path.dirname(__file__),
            'ori_square_grid_338.txt'
        )
    
    if not os.path.exists(euler_file):
        raise FileNotFoundError(f"欧拉角文件不存在: {euler_file}")
    
    euler_angles = []
    with open(euler_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 解析格式: (pitch, roll)
                line = line.strip('()')
                pitch_str, roll_str = line.split(',')
                pitch = float(pitch_str.strip())
                roll = float(roll_str.strip())
                euler_angles.append([pitch, roll])
    
    euler_angles = np.array(euler_angles, dtype=np.float32)
    print(f"✓ 加载 {len(euler_angles)} 个欧拉角对")
    return euler_angles


def main():
    parser = argparse.ArgumentParser(description='评估器网络推理脚本')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重文件路径（.pth文件）')
    parser.add_argument('--ply', type=str, required=True,
                       help='点云文件路径（.ply文件）')
    parser.add_argument('--euler', type=str, nargs=2, metavar=('PITCH', 'ROLL'),
                       help='欧拉角（pitch, roll），单位：度')
    parser.add_argument('--euler_idx', type=int,
                       help='使用预定义欧拉角索引（0-337）')
    parser.add_argument('--tool_params', type=float, nargs=4, metavar=('T1', 'T2', 'T3', 'T4'),
                       required=True,
                       help='刀具参数（4个浮点数）')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='计算设备（默认：cuda）')
    parser.add_argument('--batch', type=int, default=1,
                       help='批处理大小（默认：1）')
    parser.add_argument('--euler_file', type=str, default=None,
                       help='欧拉角文件路径（默认：使用项目中的ori_square_grid_338.txt）')
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    # 初始化推理类
    print("="*80)
    print("初始化推理模型...")
    print("="*80)
    inferencer = ScorerInference(
        model_path=args.model,
        device=args.device
    )
    
    # 处理欧拉角
    if args.euler is not None:
        euler_angles = np.array([[float(args.euler[0]), float(args.euler[1])]], 
                                dtype=np.float32)
        print(f"使用指定的欧拉角: pitch={args.euler[0]}°, roll={args.euler[1]}°")
    elif args.euler_idx is not None:
        euler_angles_all = load_euler_angles(args.euler_file)
        if args.euler_idx < 0 or args.euler_idx >= len(euler_angles_all):
            raise ValueError(f"欧拉角索引超出范围: {args.euler_idx} (有效范围: 0-{len(euler_angles_all)-1})")
        euler_angles = euler_angles_all[args.euler_idx:args.euler_idx+1]
        print(f"使用预定义欧拉角索引 {args.euler_idx}: pitch={euler_angles[0,0]}°, roll={euler_angles[0,1]}°")
    else:
        raise ValueError("必须指定 --euler 或 --euler_idx")
    
    # 处理刀具参数
    tool_params = np.array([args.tool_params], dtype=np.float32)
    print(f"刀具参数: {tool_params[0]}")
    
    # 批处理
    if args.batch > 1:
        euler_angles = np.repeat(euler_angles, args.batch, axis=0)
        tool_params = np.repeat(tool_params, args.batch, axis=0)
    
    # 推理
    print("\n" + "="*80)
    print("开始推理...")
    print("="*80)
    scores = inferencer.predict_from_file(
        ply_path=args.ply,
        euler_angles=euler_angles,
        tool_params=tool_params
    )
    
    # 输出结果
    print("\n" + "="*80)
    print("推理结果")
    print("="*80)
    for i, score in enumerate(scores):
        print(f"样本 {i+1}: 评分 = {score:.6f}")
    print("="*80)
    
    return scores


if __name__ == '__main__':
    try:
        scores = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

