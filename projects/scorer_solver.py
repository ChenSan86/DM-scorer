# --------------------------------------------------------
# Scorer Solver for Training Quality Evaluation Network
# Copyright (c) 2025
# --------------------------------------------------------

import os
import torch
import ocnn
import numpy as np
from tqdm import tqdm
from thsolver import Solver

from datasets import get_seg_shapenet_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


class ScorerSolver(Solver):
    """评估器网络训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ==================== Model / Dataset ====================
    def get_model(self, flags):
        """创建评估器模型"""
        from ocnn.models.scorer_net import ScorerNet
        model = ScorerNet(
            in_channels=flags.channel,
            geo_channels=getattr(flags, 'geo_channels', 256),
            rot_channels=getattr(flags, 'rot_channels', 128),
            tool_channels=getattr(flags, 'tool_channels', 64),
            interp=flags.interp,
            nempty=flags.nempty
        )
        return model

    def get_dataset(self, flags):
        """获取数据集"""
        return get_seg_shapenet_dataset(flags)

    def get_input_feature(self, octree):
        """从octree提取输入特征"""
        flags = self.FLAGS.MODEL
        octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
        data = octree_feature(octree)
        return data

    # ==================== Batch Processing ====================
    def process_batch(self, batch, flags):
        """处理batch，移到GPU"""
        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
            batch['points'] = batch['points'].cuda(non_blocking=True)

        # 转换其他字段
        if 'labels' in batch:
            batch['labels'] = batch['labels'].cuda(non_blocking=True)
        if 'rotation_matrices' in batch:
            batch['rotation_matrices'] = batch['rotation_matrices'].cuda(
                non_blocking=True)

        return batch

    def _to_cuda_float_tensor(self, x):
        """转换为CUDA float tensor"""
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32, device='cuda')

        import numpy as np
        try:
            x_np = np.array(x, dtype=np.float32)
        except (TypeError, ValueError):
            x_np = np.array([[str(v).strip() for v in row]
                            for row in x], dtype=np.float32)

        return torch.from_numpy(x_np).to(device='cuda')

    # ==================== Training Strategy ====================
    def sample_rotation_indices(self, B, strategy='uniform'):
        """
        为每个样本采样一个旋转矩阵索引

        Args:
            B: batch size
            strategy: 采样策略
                - 'uniform': 均匀随机采样
                - 'hard': 困难样本挖掘（高分数区域）
                - 'mixed': 混合采样

        Returns:
            indices: [B] 旋转矩阵索引
        """
        if strategy == 'uniform':
            # 均匀随机采样
            indices = torch.randint(0, 338, (B,), device='cuda')
        elif strategy == 'hard':
            # 困难样本挖掘：从分数较高的前50%中采样
            # TODO: 需要根据当前模型预测实现
            indices = torch.randint(0, 338, (B,), device='cuda')
        else:  # mixed
            # 80%均匀，20%从前半部分采样
            num_uniform = int(0.8 * B)
            uniform_indices = torch.randint(
                0, 338, (num_uniform,), device='cuda')
            hard_indices = torch.randint(
                0, 169, (B - num_uniform,), device='cuda')
            indices = torch.cat([uniform_indices, hard_indices])
            indices = indices[torch.randperm(B, device='cuda')]

        return indices

    def model_forward(self, batch, rot_indices=None):
        """
        模型前向传播

        Args:
            batch: 数据批次
            rot_indices: [B] 指定的旋转矩阵索引（可选，用于推理）

        Returns:
            score_pred: [B] 预测分数
            score_gt: [B] 真实分数
            rot_indices: [B] 使用的旋转索引
        """
        octree, points = batch['octree'], batch['points']
        data = self.get_input_feature(octree)
        query_pts = torch.cat([points.points, points.batch_id], dim=1)

        B = batch['labels'].size(0)

        # 采样旋转矩阵索引
        if rot_indices is None:
            rot_indices = self.sample_rotation_indices(B, strategy='uniform')

        # 获取对应的旋转矩阵
        rotation_matrices = batch['rotation_matrices']  # [338, 3, 3]
        selected_rotations = rotation_matrices[rot_indices]  # [B, 3, 3]

        # 获取刀具参数
        tool_params = self._to_cuda_float_tensor(
            batch['tool_params'])  # [B, 4]

        # 前向传播
        score_pred = self.model.forward(
            data, octree, octree.depth, query_pts,
            selected_rotations, tool_params
        )  # [B]

        # 获取对应的GT分数
        labels = batch['labels']  # [B, 338]
        score_gt = labels[torch.arange(B, device='cuda'), rot_indices]  # [B]

        return score_pred, score_gt, rot_indices

    # ==================== Loss & Metrics ====================
    def loss_function(self, score_pred, score_gt):
        """MSE loss"""
        loss = torch.nn.functional.mse_loss(score_pred, score_gt)
        return loss

    def loss_function_huber(self, score_pred, score_gt, delta=1.0):
        """Huber loss（对outlier更鲁棒）"""
        loss = torch.nn.functional.huber_loss(
            score_pred, score_gt, delta=delta)
        return loss

    def mae(self, score_pred, score_gt):
        """平均绝对误差"""
        return torch.abs(score_pred - score_gt).mean().item()

    def rmse(self, score_pred, score_gt):
        """均方根误差"""
        return torch.sqrt(torch.mean((score_pred - score_gt) ** 2)).item()

    def relative_error(self, score_pred, score_gt):
        """相对误差（百分比）"""
        rel_err = torch.abs(score_pred - score_gt) / (score_gt.abs() + 1e-6)
        return (rel_err.mean() * 100).item()

    # ==================== Train / Test Steps ====================
    def train_step(self, batch):
        """训练步骤"""
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        score_pred, score_gt, _ = self.model_forward(batch)

        # 计算损失
        loss = self.loss_function(score_pred, score_gt)

        # 计算指标
        mae = self.mae(score_pred, score_gt)
        rmse_val = self.rmse(score_pred, score_gt)
        rel_err = self.relative_error(score_pred, score_gt)

        device = loss.device
        return {
            'train/loss': loss,
            'train/mae': torch.tensor(mae, dtype=torch.float32, device=device),
            'train/rmse': torch.tensor(rmse_val, dtype=torch.float32, device=device),
            'train/rel_error': torch.tensor(rel_err, dtype=torch.float32, device=device),
        }

    def test_step(self, batch):
        """测试步骤"""
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        with torch.no_grad():
            score_pred, score_gt, _ = self.model_forward(batch)

            loss = self.loss_function(score_pred, score_gt)
            mae = self.mae(score_pred, score_gt)
            rmse_val = self.rmse(score_pred, score_gt)
            rel_err = self.relative_error(score_pred, score_gt)

        device = loss.device
        names = ['test/loss', 'test/mae', 'test/rmse', 'test/rel_error']
        tensors = [
            loss,
            torch.tensor(mae, dtype=torch.float32, device=device),
            torch.tensor(rmse_val, dtype=torch.float32, device=device),
            torch.tensor(rel_err, dtype=torch.float32, device=device),
        ]
        return dict(zip(names, tensors))

    def eval_step(self, batch):
        """
        评估步骤：对每个样本测试所有338个姿态
        """
        batch = self.process_batch(batch, self.FLAGS.DATA.test)
        B = batch['labels'].size(0)

        with torch.no_grad():
            all_scores = []

            # 遍历所有338个姿态
            for rot_idx in range(338):
                rot_indices = torch.full(
                    (B,), rot_idx, device='cuda', dtype=torch.long)
                score_pred, _, _ = self.model_forward(batch, rot_indices)
                all_scores.append(score_pred.cpu().numpy())

            all_scores = np.stack(all_scores, axis=1)  # [B, 338]
            labels_gt = batch['labels'].cpu().numpy()  # [B, 338]

            # 保存结果
            filenames = batch['filename']
            for i, fname in enumerate(filenames):
                self.eval_rst[fname] = {
                    'scores_pred': all_scores[i],  # [338]
                    'scores_gt': labels_gt[i],     # [338]
                }

                # 最后一个epoch保存
                if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                    full_filename = os.path.join(
                        self.logdir, fname[:-4] + '.scorer_eval.npz'
                    )
                    curr_folder = os.path.dirname(full_filename)
                    if not os.path.exists(curr_folder):
                        os.makedirs(curr_folder)
                    np.savez(
                        full_filename,
                        scores_pred=self.eval_rst[fname]['scores_pred'],
                        scores_gt=self.eval_rst[fname]['scores_gt'],
                    )

    def result_callback(self, avg_tracker, epoch):
        """打印结果回调"""
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
        mae = _to_float(avg.get('test/mae'))
        rmse = _to_float(avg.get('test/rmse'))
        rel_err = _to_float(avg.get('test/rel_error'))

        tqdm.write(f'=> Epoch: {epoch} | '
                   f'loss: {loss:.6f} | '
                   f'MAE: {mae:.6f} | '
                   f'RMSE: {rmse:.6f} | '
                   f'Rel.Err: {rel_err:.2f}%')


if __name__ == "__main__":
    ScorerSolver.main()
