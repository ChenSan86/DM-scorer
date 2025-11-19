# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import os
import torch
import ocnn
import numpy as np
import json
from tqdm import tqdm
from thsolver import Solver

from datasets import (get_seg_shapenet_dataset, get_scannet_dataset,
                      get_kitti_dataset)

torch.multiprocessing.set_sharing_strategy('file_system')


class SegSolver(Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 加载旋转矩阵映射
        self.rotation_matrices = self._load_rotation_matrices()

    def _load_rotation_matrices(self):
        """加载JSON文件中的旋转矩阵到内存"""
        json_path = os.path.join(os.path.dirname(
            __file__), 'rotation_matrices.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            rotation_matrices = []
            for i in range(338):  # 假设有110个旋转矩阵
                key = f"ori_{i:03d}"
                if key in data:
                    matrix = np.array(
                        data[key]['rotation_matrix'], dtype=np.float32)
                    rotation_matrices.append(torch.from_numpy(matrix))
                else:
                    # 如果缺少某个索引，用单位矩阵填充
                    rotation_matrices.append(torch.eye(3, dtype=torch.float32))

            # 堆叠成 (110, 3, 3) 的tensor
            rotation_matrices = torch.stack(rotation_matrices)
            print(f"成功加载 {len(rotation_matrices)} 个旋转矩阵")
            return rotation_matrices
        except Exception as e:
            print(f"加载旋转矩阵失败: {e}")
            # 返回 110 个单位矩阵作为 fallback
            return torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(338, 1, 1)

    # -----------------------------
    # Model / Dataset constructors
    # -----------------------------
    def get_model(self, flags):
        if flags.name.lower() == 'segnet':
            model = ocnn.models.SegNet(
                flags.channel, flags.nout, flags.stages, flags.interp, flags.nempty)
        elif flags.name.lower() == 'unet':
            # 从 YAML 的 MODEL 节点里摘出我们关心的可选开关
            extra_kwargs = {}
            for k in ['use_decoder', 'pyramid_depths', 'tool_fusion',
                      'use_attention_pool', 'use_tanh_head']:
                if hasattr(flags, k):
                    extra_kwargs[k] = getattr(flags, k)
            model = ocnn.models.UNet(
                flags.channel, flags.nout, flags.interp, flags.nempty, **extra_kwargs)
        else:
            raise ValueError('Unknown model name: {}'.format(flags.name))
        return model

    def get_dataset(self, flags):
        if flags.name.lower() == 'shapenet':
            return get_seg_shapenet_dataset(flags)
        elif flags.name.lower() == 'scannet':
            return get_scannet_dataset(flags)
        elif flags.name.lower() == 'kitti':
            return get_kitti_dataset(flags)
        else:
            raise ValueError('Unknown dataset name: {}'.format(flags.name))

    def get_input_feature(self, octree):
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    # -----------------------------
    # Batch processing utilities
    # -----------------------------
    def _to_cuda_float_tensor(self, x):
        """Robust conversion: list / list[np.ndarray] / np.ndarray / tensor -> float32 CUDA tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32, device='cuda')

        import numpy as np
        # 关键：直接用 dtype=np.float32 强制数值化（可处理 "1.23" 之类的字符串）
        try:
            x_np = np.array(x, dtype=np.float32)
        except (TypeError, ValueError):
            # 若内部混有空字符串或多余空格，做一次清洗后再转
            x_np = np.array([[str(v).strip() for v in row]
                            for row in x], dtype=np.float32)

        return torch.from_numpy(x_np).to(device='cuda')

    def process_batch(self, batch, flags):
        def points2octree(points):
            octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
            octree.build_octree(points)
            return octree

        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)
        else:
            points = [pts.cuda(non_blocking=True) for pts in batch['points']]
            octrees = [points2octree(pts) for pts in points]
            octree = ocnn.octree.merge_octrees(octrees)
            octree.construct_all_neigh()
            batch['points'] = ocnn.octree.merge_points(points)
            batch['octree'] = octree
        return batch

    # -----------------------------
    # Forward pass
    # -----------------------------
    def model_forward(self, batch):
        """
        模型前向传播

        Args:
            batch: 数据批次

        Returns:
            score_pred: [B] 预测分数
            score_gt: [B] 真实分数
            features: [B, 338, feature_dim] 每个姿态的特征
        """
        octree, points = batch['octree'], batch['points']
        data = self.get_input_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)

        B = batch['labels'].size(0)

        angles = batch['angles']  # 每个样本有 338 个姿态
        tool_params = batch['tool_params']

        # 获取前向预测分数
        score_pred = self.model.forward(data, octree, octree.depth, query_pts, angles, tool_params)

        # 获取对应的 GT 分数
        score_gt = batch['labels']  # [B]

        # 获取每个姿态的特征
        features = self.model.get_features(data, octree, octree.depth, query_pts, angles)

        return score_pred, score_gt, features

    @staticmethod
    def _angles_to_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
        """
        Rx(big)@Ry(small)
        """
        assert angles.dim() == 2 and angles.size(1) in (
            2, 3), "angles must be (B,2) or (B,3)"
        device = angles.device
        pitch = angles[:, 0] * (torch.pi / 180.0)  # 小
        roll = angles[:, 1] * (torch.pi / 180.0)  # 大

        cp, sp = torch.cos(pitch), torch.sin(pitch)  # 小
        cr, sr = torch.cos(roll), torch.sin(roll)

        Rx = torch.zeros(angles.size(0), 3, 3,
                         device=device, dtype=angles.dtype)
        Rx[:, 0, 0] = 1
        Rx[:, 1, 1] = cr
        Rx[:, 1, 2] = -sr
        Rx[:, 2, 1] = sr
        Rx[:, 2, 2] = cr

        Ry = torch.zeros_like(Rx)
        Ry[:, 0, 0] = cp
        Ry[:, 0, 2] = sp
        Ry[:, 1, 1] = 1
        Ry[:, 2, 0] = -sp
        Ry[:, 2, 2] = cp

        R = Rx @ Ry

        return R

    # -----------------------------
    # Loss & Metrics (Angles -> R)
    # -----------------------------
    @staticmethod
    def _six_dim_to_rotation_matrix(six_dim_vector: torch.Tensor) -> torch.Tensor:
        """(N,6) -> (N,3,3) using Gram–Schmidt; numerically stable."""
        x = six_dim_vector[:, 0:3]
        y = six_dim_vector[:, 3:6]

        # 防止全零向量导致normalize产生NaN
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x = torch.where(x_norm > 1e-8, x / x_norm,
                        torch.tensor([1.0, 0.0, 0.0], device=x.device))

        y = y - torch.sum(x * y, dim=1, keepdim=True) * x

        y_norm = torch.norm(y, dim=1, keepdim=True)
        y = torch.where(y_norm > 1e-8, y / y_norm,
                        torch.tensor([0.0, 1.0, 0.0], device=y.device))

        z = torch.cross(x, y, dim=1)
        R = torch.stack([x, y, z], dim=-1)  # (N,3,3)
        return R

    def _label_to_rotation_matrix(self, label: torch.Tensor) -> torch.Tensor:
        """
        将 GT label 转为旋转矩阵，兼容多种格式：
          - (B,2) / (B,3): 角度（度），先 Rx 再 Ry（yaw 默认为 0）
          - (B,6): 6D 表示，走 Gram–Schmidt
        其他形状将抛出 ValueError。
        """
        if not isinstance(label, torch.Tensor) or label.dim() != 2:
            raise ValueError("GT label must be a (B,2/3/6) tensor")

        if label.size(1) in (2, 3):
            return self._angles_to_rotation_matrix(label, yaw_zero=(label.size(1) == 2))
        elif label.size(1) == 6:
            return self._six_dim_to_rotation_matrix(label)
        else:
            raise ValueError(
                f"Unsupported GT label shape: {tuple(label.shape)}")

    def _maybe_label_is_score_table(self, label: torch.Tensor) -> bool:
        """
        粗判 label 是否像打分表 (B, K)：
          - 是 2 维
          - 列数既不是 2、3、6
        """
        return isinstance(label, torch.Tensor) and label.dim() == 2 and label.size(1) not in (2, 3, 6)

    def _safe_metric_default(self) -> float:
        """当没有 GT 角度/6D（例如只有打分表）时，给一个合理的默认指标值。"""
        return 0.0
    def contrastive_loss(F_i, F_j, gt_i, gt_j):
        """简化版对比损失"""
        # 计算标签差异作为权重
        w_ij = torch.abs(gt_i - gt_j)  # 权重：标签差异

        # 计算特征之间的欧氏距离
        dist = torch.norm(F_i - F_j, p=2, dim=1)

        # 计算对比损失
        loss = dist * w_ij  # 权重乘以距离
        return loss.mean()  # 平均损失


    def loss_function(self, logit, label):
        """
        Frobenius loss between predicted and gt rotation matrices.
        预测：angles (B,2) -> R
        GT: 兼容 (B,2)/(B,3)/(B,6)。若是其他形状（如打分表），这里将抛错。
        """
        R_pred = self._angles_to_rotation_matrix(logit)

        # 若 label 是打分表，这个 loss 不适用；建议仅在 label 是角度/6D 时使用。
        if self._maybe_label_is_score_table(label):
            raise ValueError(
                "loss_function expects GT angles or 6D, but got a score table. "
                "Use loss_function_pro/promax instead."
            )

        R_gt = self._label_to_rotation_matrix(label)
        diff = R_pred - R_gt
        loss = diff.pow(2).mean()
        return loss

    def loss_function_pro(self, logit, label):
        """
        新的loss函数（用于打分表）：
        1. 将logit转换为旋转矩阵
        2. 与预定义旋转矩阵比较，使用软分配
        3. 返回label中加权平均的浮点数作为loss
        说明：label 通常为 (B, K) 的打分表（分数越低越好）
        """

        # 预测角度 -> 旋转矩阵
        R_pred = self._angles_to_rotation_matrix(logit)  # (B, 3, 3)

        # 确保旋转矩阵在正确的设备上
        if self.rotation_matrices.device != R_pred.device:
            self.rotation_matrices = self.rotation_matrices.to(R_pred.device)

        # 计算与所有预定义旋转矩阵的测地距离
        R_pred_expanded = R_pred.unsqueeze(1)  # (B, 1, 3, 3)
        rotation_matrices_expanded = self.rotation_matrices.unsqueeze(
            0)  # (1, K, 3, 3)

        R_pred_transposed = R_pred_expanded.transpose(-1, -2)  # (B, 1, 3, 3)
        R_diff = torch.matmul(
            R_pred_transposed, rotation_matrices_expanded)  # (B, K, 3, 3)

        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(dim=-1)  # (B, K)
        cos_angle = torch.clamp((trace - 1) / 2, -0.999999, 0.999999)
        distances = torch.acos(cos_angle)  # (B, K)
        distances = torch.where(torch.isnan(distances),
                                torch.zeros_like(distances), distances)

        temperature = 0.1
        weights = torch.softmax(-distances / temperature, dim=1)  # (B, K)

        # label 为打分表 (B, K)，分数越低越好
        loss_values = torch.sum(weights * label, dim=1)  # (B,)

        final_loss = loss_values.mean()
        if torch.isnan(final_loss):
            print("错误：最终loss为NaN，返回默认值")
            return torch.tensor(1.0, device=loss_values.device, requires_grad=True)
        return final_loss

    def loss_function_promax(self, logit, label):
        """
        改进的loss函数（打分表，允许预测矩阵绕 z 轴自由旋转）：
        1. 将logit转换为旋转矩阵
        2. 允许预测的旋转矩阵绕z轴任意旋转（离散采样）
        3. 与预定义旋转矩阵比较，使用软分配
        4. 返回label中加权平均的浮点数作为loss
        """
        R_pred = self._angles_to_rotation_matrix(logit)  # (B, 3, 3)

        # 设备同步
        if self.rotation_matrices.device != R_pred.device:
            self.rotation_matrices = self.rotation_matrices.to(R_pred.device)

        # 生成 z 轴旋转矩阵（每 30° 采样一次，共 12 个角度）
        angles = torch.linspace(0, 2 * np.pi, 13, device=R_pred.device)[:-1]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        R_z = torch.zeros(len(angles), 3, 3, device=R_pred.device)
        R_z[:, 0, 0] = cos_angles
        R_z[:, 0, 1] = -sin_angles
        R_z[:, 1, 0] = sin_angles
        R_z[:, 1, 1] = cos_angles
        R_z[:, 2, 2] = 1.0

        # 对每个预测矩阵应用所有 z 轴旋转 (B, 12, 3, 3)
        R_pred_rotated = torch.matmul(R_pred.unsqueeze(1), R_z.unsqueeze(0))

        # 计算测地距离 (B, 12, K)
        B, num_rotations = R_pred_rotated.shape[:2]
        K = self.rotation_matrices.shape[0]

        R_pred_flat = R_pred_rotated.view(-1, 3, 3)  # (B*12, 3, 3)
        R_diff = torch.matmul(R_pred_flat.unsqueeze(1).transpose(-1, -2),
                              self.rotation_matrices.unsqueeze(0))            # (B*12, K, 3, 3)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-
                               1).sum(dim=-1)          # (B*12, K)
        cos_angle = torch.clamp((trace - 1) / 2, -0.999999, 0.999999)
        # (B*12, K)
        distances_flat = torch.acos(cos_angle)
        distances = distances_flat.view(
            B, num_rotations, K)                   # (B, 12, K)

        # 对每个预定义矩阵取最小距离（考虑 z 轴旋转）
        min_distances, _ = distances.min(dim=1)  # (B, K)
        min_distances = torch.where(torch.isnan(min_distances),
                                    torch.zeros_like(min_distances), min_distances)

        temperature = 0.1
        weights = torch.softmax(-min_distances / temperature, dim=1)  # (B, K)

        # label 为打分表 (B, K)
        loss_values = torch.sum(weights * label, dim=1)  # (B,)
        final_loss = loss_values.mean()
        if torch.isnan(final_loss):
            print("错误：loss_function_promax最终loss为NaN，返回默认值")
            return torch.tensor(1.0, device=loss_values.device, requires_grad=True)
        return final_loss

    def mean(self, logit, label):
        R_pred = self._angles_to_rotation_matrix(logit)

        # label 若是打分表，无法得到 GT 矩阵，返回默认值
        if self._maybe_label_is_score_table(label):
            return self._safe_metric_default()

        R_gt = self._label_to_rotation_matrix(label)
        R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        angle_error = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        return angle_error.mean().item()

    def max(self, logit, label):
        R_pred = self._angles_to_rotation_matrix(logit)
        if self._maybe_label_is_score_table(label):
            return self._safe_metric_default()
        R_gt = self._label_to_rotation_matrix(label)
        R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        angle_error = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        return angle_error.max().item()

    def std_score(self, logit, label):
        R_pred = self._angles_to_rotation_matrix(logit)
        if self._maybe_label_is_score_table(label):
            return self._safe_metric_default()
        R_gt = self._label_to_rotation_matrix(label)
        R_diff = torch.matmul(R_pred.transpose(1, 2), R_gt)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-1).sum(-1)
        angle_error = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        # 关键改动：unbiased=False，避免只有 1 个样本时的告警
        return angle_error.std(unbiased=False).item()

    def model_score(self, logit, score):
        """
        新的accuracy指标（打分表场景）：
        根据预测出的旋转矩阵，找到离得最近的那个预定义旋转矩阵，
        将旋转矩阵对应的索引值作为模型评分
        """
        R_pred = self._angles_to_rotation_matrix(logit)
        # 设备同步
        if self.rotation_matrices.device != R_pred.device:
            self.rotation_matrices = self.rotation_matrices.to(R_pred.device)

        # 转换 score 为 tensor
        if score is None:
            return 0.0
        if isinstance(score, list):
            try:
                score = np.stack(score, axis=0)
            except Exception:
                score = np.array(score)
        if isinstance(score, np.ndarray):
            score = torch.from_numpy(score)
        if not isinstance(score, torch.Tensor):
            return 0.0
        score = score.to(dtype=torch.float32, device=R_pred.device)

        # 规范 shape: (K,) -> (1,K)
        if score.dim() == 1:
            score = score.unsqueeze(0)

        K_defined = self.rotation_matrices.shape[0]
        # 补齐或裁剪列
        if score.shape[1] < K_defined:
            pad = torch.zeros(
                score.shape[0], K_defined - score.shape[1], device=score.device)
            score = torch.cat([score, pad], dim=1)
        elif score.shape[1] > K_defined:
            score = score[:, :K_defined]

        # 广播到 batch
        if score.shape[0] == 1 and R_pred.shape[0] > 1:
            score = score.repeat(R_pred.shape[0], 1)

        # 测地距离
        R_diff = torch.matmul(R_pred.unsqueeze(1).transpose(-1, -2),
                              self.rotation_matrices.unsqueeze(0))         # (B, K, 3, 3)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-
                               1).sum(dim=-1)       # (B, K)
        cos_angle = torch.clamp((trace - 1) / 2, -0.999999, 0.999999)
        # (B, K)
        distances = torch.acos(cos_angle)
        closest = torch.argmin(
            distances, dim=1)                             # (B,)
        closest = torch.clamp(closest, 0, score.shape[1] - 1)
        picked = score[torch.arange(
            score.shape[0], device=score.device), closest]
        return picked.mean().item()

    def model_score_pro(self, logit, score):
        """
        改进的accuracy指标（打分表场景）：
        允许预测的旋转矩阵绕 z 轴任意旋转，然后找到离得最近的预定义旋转矩阵，
        将旋转矩阵对应的索引值作为模型评分
        """
        R_pred = self._angles_to_rotation_matrix(logit)
        if self.rotation_matrices.device != R_pred.device:
            self.rotation_matrices = self.rotation_matrices.to(R_pred.device)

        # 转换 score 为 tensor（同上）
        if score is None:
            return 0.0
        if isinstance(score, list):
            try:
                score = np.stack(score, axis=0)
            except Exception:
                score = np.array(score)
        if isinstance(score, np.ndarray):
            score = torch.from_numpy(score)
        if not isinstance(score, torch.Tensor):
            return 0.0
        score = score.to(dtype=torch.float32, device=R_pred.device)
        if score.dim() == 1:
            score = score.unsqueeze(0)

        K_defined = self.rotation_matrices.shape[0]
        if score.shape[1] < K_defined:
            pad = torch.zeros(
                score.shape[0], K_defined - score.shape[1], device=score.device)
            score = torch.cat([score, pad], dim=1)
        elif score.shape[1] > K_defined:
            score = score[:, :K_defined]
        if score.shape[0] == 1 and R_pred.shape[0] > 1:
            score = score.repeat(R_pred.shape[0], 1)

        # 生成 z 轴旋转（12 个）
        angles = torch.linspace(0, 2 * np.pi, 13, device=R_pred.device)[:-1]
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        R_z = torch.zeros(len(angles), 3, 3, device=R_pred.device)
        R_z[:, 0, 0] = cos_angles
        R_z[:, 0, 1] = -sin_angles
        R_z[:, 1, 0] = sin_angles
        R_z[:, 1, 1] = cos_angles
        R_z[:, 2, 2] = 1.0

        R_pred_rotated = torch.matmul(R_pred.unsqueeze(
            1), R_z.unsqueeze(0))   # (B, 12, 3, 3)
        B, num_rotations = R_pred_rotated.shape[:2]
        K = self.rotation_matrices.shape[0]
        # (B*12, 3, 3)
        R_pred_flat = R_pred_rotated.view(-1, 3, 3)

        R_diff = torch.matmul(R_pred_flat.unsqueeze(1).transpose(-1, -2),
                              self.rotation_matrices.unsqueeze(0))              # (B*12, K, 3, 3)
        trace = torch.diagonal(R_diff, dim1=-2, dim2=-
                               1).sum(dim=-1)            # (B*12, K)
        cos_angle = torch.clamp((trace - 1) / 2, -0.999999, 0.999999)
        # (B*12, K)
        distances_flat = torch.acos(cos_angle)
        distances = distances_flat.view(
            B, num_rotations, K)                     # (B, 12, K)

        min_distances, _ = distances.min(
            dim=1)                                  # (B, K)
        closest = torch.argmin(
            min_distances, dim=1)                             # (B,)
        closest = torch.clamp(closest, 0, score.shape[1] - 1)
        picked = score[torch.arange(
            score.shape[0], device=score.device), closest]
        return picked.mean().item()

    # -----------------------------
    # Train / Test / Eval loops
    # -----------------------------
    def train_step(self, batch):
        # 注意：大量打印会拖慢训练，按需开启
        
        print("--------------------------------")
        print("train_step")
        print("--------------------------------")
        print("batch: ", batch)
        print("--------------------------------")
        
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        logit, label = self.model_forward(batch)

        # 若 label 是打分表，优先用 pro 族损失
        loss = self.loss_function_pro(logit, label)

        mean = self.mean(logit, label)
        maxe = self.max(logit, label)
        stdv = self.std_score(logit, label)
        model_score = self.model_score(logit, label)
        model_score_pro = self.model_score_pro(logit, label)

        device = loss.device
        return {
            'train/loss': loss,
            'train/mean_error': torch.tensor(mean, dtype=torch.float32, device=device),
            'train/max_error': torch.tensor(maxe, dtype=torch.float32, device=device),
            'train/standard_deviation': torch.tensor(stdv, dtype=torch.float32, device=device),
            'train/model_score': torch.tensor(model_score, dtype=torch.float32, device=device),
            'train/model_score_pro': torch.tensor(model_score_pro, dtype=torch.float32, device=device),
        }

    def test_step(self, batch):
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            logit, label = self.model_forward(batch)

            loss = self.loss_function_pro(logit, label)
            mean = self.mean(logit, label)
            maxe = self.max(logit, label)
            stdv = self.std_score(logit, label)
            model_score = self.model_score(logit, label)
            model_score_pro = self.model_score_pro(logit, label)

        device = loss.device
        names = ['test/loss', 'test/mean_error', 'test/max_error',
                 'test/standard_deviation', 'test/model_score', 'test/model_score_pro']
        tensors = [
            loss,
            torch.tensor(mean, dtype=torch.float32, device=device),
            torch.tensor(maxe, dtype=torch.float32, device=device),
            torch.tensor(stdv, dtype=torch.float32, device=device),
            torch.tensor(model_score, dtype=torch.float32, device=device),
            torch.tensor(model_score_pro, dtype=torch.float32, device=device),
        ]
        return dict(zip(names, tensors))

    def eval_step(self, batch):
        """
        Evaluation-time export of predictions. Saves per-sample angles (deg) and 3x3 matrices.
        """
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            # logit: (B,2) angles in degrees
            logit, label = self.model_forward(batch)
            R_pred = self._angles_to_rotation_matrix(
                logit).cpu().numpy()   # (B,3,3)
            angles_pred = logit.detach().cpu().numpy()                       # (B,2)

        filenames = batch['filename']  # list of strings length B
        for i, fname in enumerate(filenames):
            # record last prediction; solver may call multiple epochs
            self.eval_rst[fname] = {
                'angles_deg': angles_pred[i],  # [pitch, roll] in degrees
                'R': R_pred[i],
            }

            # Save on the last eval epoch
            if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                # logs/.../<original>.eval.npz
                full_filename = os.path.join(
                    self.logdir, fname[:-4] + '.eval.npz')
                curr_folder = os.path.dirname(full_filename)
                if not os.path.exists(curr_folder):
                    os.makedirs(curr_folder)
                np.savez(
                    full_filename,
                    angles_deg=self.eval_rst[fname]['angles_deg'],
                    R=self.eval_rst[fname]['R'],
                )

    def result_callback(self, avg_tracker, epoch):
        """
        Print concise pose metrics aggregated by the tracker.
        Accept both Python floats and torch.Tensors.
        """
        avg = avg_tracker.average()

        def _to_float(x, default=0.0):
            if x is None:
                return default
            if isinstance(x, torch.Tensor):
                return x.detach().item()
            try:
                return float(x)
            except Exception:
                return default

        loss = _to_float(avg.get('test/loss'))
        mean_err = _to_float(avg.get('test/mean_error'))
        max_err = _to_float(avg.get('test/max_error'))
        std_err = _to_float(avg.get('test/standard_deviation'))
        model_score = _to_float(avg.get('test/model_score'))
        model_score_pro = _to_float(avg.get('test/model_score_pro'))

        tqdm.write(f'=> Epoch: {epoch} | '
                   f'test/loss: {loss:.6f} | '
                   f'mean(rad): {mean_err:.6f} | '
                   f'max(rad): {max_err:.6f} | '
                   f'std(rad): {std_err:.6f} | '
                   f'model_score: {model_score:.6f} | '
                   f'model_score_pro: {model_score_pro:.6f}')


if __name__ == "__main__":
    SegSolver.main()
