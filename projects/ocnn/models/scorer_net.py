# --------------------------------------------------------
# Scorer Network for Reachability Evaluation (No Tool Params)
# Copyright (c) 2025
# --------------------------------------------------------

import torch
import torch.nn as nn
from typing import Optional
import ocnn
from ocnn.octree import Octree


class ScorerNet(nn.Module):
    """
    可达性评估网络：给定点云和姿态，预测当前姿态下的不可达点比例（0-1）

    任务:
        - 评估模型点云在该姿态下的可达性
        - 标签：不可达点数量 / 点云总数量 ∈ [0, 1]
          → 数值越大，不可达点越多，可达性越差
          → 数值越小，不可达点越少，可达性越好

    输入:
        - data: [N_nodes, C_in] 八叉树节点特征
        - octree: Octree 对象
        - depth: int
        - query_pts: [N_pts, 4] 查询点 (x, y, z, batch_id)
        - euler_angles: [B, 2] 欧拉角对（pitch, roll）
        - tool_params: [B, 4] 刀具参数（仅为兼容保留，当前版本不使用）

    输出:
        - unreachable_ratio: [B] 不可达点比例（不可达点数量 / 点云数量），范围 [0, 1]
    """

    def __init__(
        self,
        in_channels: int = 4,
        geo_channels: int = 256,      # 几何特征维度
        rot_channels: int = 128,      # 旋转特征维度
        interp: str = 'linear',
        nempty: bool = False,
        use_decoder: bool = False,    # 是否使用解码器（保留接口）
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.geo_channels = geo_channels
        self.rot_channels = rot_channels
        self.nempty = nempty
        self.use_decoder = use_decoder

        # ============ 1. 几何特征提取（使用现有的编码器）============
        self._config_geometry_encoder()

        # 编码器第一层
        self.conv1 = ocnn.modules.OctreeConvBnRelu(
            in_channels, self.encoder_channel[0], nempty=nempty
        )

        # 下采样层
        self.downsample = nn.ModuleList([
            ocnn.modules.OctreeConvBnRelu(
                self.encoder_channel[i], self.encoder_channel[i + 1],
                kernel_size=[2], stride=2, nempty=nempty
            ) for i in range(self.encoder_stages)
        ])

        # 编码器 ResBlock
        self.encoder = nn.ModuleList([
            ocnn.modules.OctreeResBlocks(
                self.encoder_channel[i + 1], self.encoder_channel[i + 1],
                self.encoder_blocks[i], self.bottleneck, nempty, self.resblk
            ) for i in range(self.encoder_stages)
        ])

        # 插值层
        self.octree_interp = ocnn.nn.OctreeInterp(interp, nempty)

        # 几何特征投影到目标维度
        self.geo_proj = nn.Sequential(
            nn.Linear(self.encoder_channel[-1], 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, geo_channels),
            nn.ReLU(inplace=True),
        )

        # ============ 2. 欧拉角特征提取 ============
        self.euler_encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, rot_channels),
            nn.ReLU(inplace=True),
        )

        # ============ 3. 特征融合（几何 + 姿态） ============
        fusion_dim = geo_channels + rot_channels
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        # ============ 4. 不可达比例预测头 ============
        self.score_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出限制到 [0, 1]，对应不可达点比例
        )

        print(f"ScorerNet initialized (ignore tool params):")
        print(f"  - Geometry channels: {geo_channels}")
        print(f"  - Rotation channels: {rot_channels}")
        print(f"  - Fusion dim: {fusion_dim}")

    def _config_geometry_encoder(self):
        """配置几何编码器（参考 UNet）"""
        self.encoder_blocks = [2, 3, 4, 6]              # 4 个 stage
        self.encoder_channel = [32, 32, 64, 128, 256]   # 5 个 channel（包括第一层输出）
        self.encoder_stages = len(self.encoder_blocks)  # 4
        self.bottleneck = 1
        self.resblk = ocnn.modules.OctreeResBlock2

    def encode_geometry(self, data, octree, depth, query_pts):
        """提取几何特征"""
        # 编码器前向
        convd = dict()
        convd[depth] = self.conv1(data, octree, depth)
        for i in range(self.encoder_stages):
            d = depth - i
            conv = self.downsample[i](convd[d], octree, d)
            convd[d - 1] = self.encoder[i](conv, octree, d - 1)

        # 获取最深层特征
        deepest_depth = depth - self.encoder_stages
        deepest_feat = convd[deepest_depth]  # [N_nodes_deep, C]

        # 插值到点云
        point_feat = self.octree_interp(
            deepest_feat, octree, deepest_depth, query_pts
        )  # [N_pts, C]

        # 全局平均池化（按 batch）
        batch_id = query_pts[:, 3].long()
        B = batch_id.max().item() + 1

        geo_feat = self._batch_mean_pool(point_feat, batch_id, B)  # [B, C]

        # 投影到目标维度
        geo_feat = self.geo_proj(geo_feat)  # [B, geo_channels]

        return geo_feat

    @staticmethod
    def _batch_mean_pool(point_feat, batch_id, B):
        """批量平均池化"""
        C = point_feat.size(1)
        sum_feat = torch.zeros(
            B, C, device=point_feat.device, dtype=point_feat.dtype
        )
        sum_feat.index_add_(0, batch_id, point_feat)
        cnt = torch.bincount(batch_id, minlength=B).clamp_min(1).float()
        return sum_feat / cnt.unsqueeze(1).to(point_feat.device)

    def forward(
        self,
        data: torch.Tensor,           # [N_nodes, C_in]
        octree: Octree,
        depth: int,
        query_pts: torch.Tensor,      # [N_pts, 4]
        euler_angles: torch.Tensor,   # [B, 2] 欧拉角对（pitch, roll）
        tool_params: Optional[torch.Tensor] = None,  # 为兼容保留，不使用
    ):
        """
        前向传播

        Returns:
            unreachable_ratio: [B] 不可达点比例（不可达点数 / 点云数量），范围 [0, 1]
        """
        B = euler_angles.size(0)

        # 1. 提取几何特征
        geo_feat = self.encode_geometry(
            data, octree, depth, query_pts
        )  # [B, geo_channels]

        # 2. 提取欧拉角特征
        rot_feat = self.euler_encoder(euler_angles)  # [B, rot_channels]

        # 3. 特征融合（几何 + 姿态，不再使用刀具参数）
        fused_feat = torch.cat(
            [geo_feat, rot_feat], dim=1
        )  # [B, fusion_dim]
        fused_feat = self.fusion_mlp(fused_feat)  # [B, 128]

        # 4. 预测不可达比例
        unreachable_ratio = self.score_head(fused_feat).squeeze(-1)  # [B]

        return unreachable_ratio
