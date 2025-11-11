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

        # ============ 自动保存最优权重（与thsolver/solver.py风格一致） ============
        # 优先使用配置项 FLAGS.SOLVER.best_val（如："min:mae" 或 "max:acc"）。
        # 若未配置，则在本类中默认使用 test/loss 最小化作为最优标准。
        self._best_mode = 'min'   # 'min' 或 'max'
        self._best_key = 'test/loss'
        best_val_cfg = getattr(self.FLAGS.SOLVER, 'best_val', '')
        if isinstance(best_val_cfg, str) and ':' in best_val_cfg:
            mode, key = best_val_cfg.split(':', 1)
            mode = mode.strip().lower()
            key = key.strip()
            # 兼容配置里写成 'test/mae' 的情况
            if key.startswith('test/'):
                metric_name = key
            else:
                metric_name = f'test/{key}'
            if mode in ('min', 'max'):
                self._best_mode = mode
                self._best_key = metric_name

        self._best_value = float('inf') if self._best_mode == 'min' else -float('inf')
        # 文件名包含指标，便于区分
        safe_key = self._best_key.replace('/', '_')
        self._best_ckpt_path = os.path.join(self.FLAGS.SOLVER.logdir, f'best_{safe_key}.pth')
        # ======================================================================

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
        """处理batch，移到GPU。稳健处理 list/ndarray/tensor 情况。"""
        # Octree 与 Points
        if 'octree' in batch:
            batch['octree'] = batch['octree'].cuda(non_blocking=True)
        if 'points' in batch:
            batch['points'] = batch['points'].cuda(non_blocking=True)

        # 统一转换到 CUDA 的小工具
        def _to_cuda(x, dtype=torch.float32):
            import numpy as np
            if isinstance(x, torch.Tensor):
                return x.to(device='cuda', dtype=dtype, non_blocking=True)
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device='cuda', dtype=dtype, non_blocking=True)
            if isinstance(x, list):
                try:
                    tx = torch.tensor(x, dtype=dtype)
                except Exception:
                    # 退而求其次：逐元素转 tensor 再 stack
                    tx = torch.stack([torch.as_tensor(e) for e in x], dim=0).to(dtype=dtype)
                return tx.to(device='cuda', non_blocking=True)
            return x

        # labels: 期望 [B, 338]，用于数值回归
        if 'labels' in batch:
            batch['labels'] = _to_cuda(batch['labels'], dtype=torch.float32)

        # euler_angles: 数据集提供 [338, 2]，但 DataLoader 可能 collate 成 [B, 338, 2] 或 list
        if 'angles' in batch:
            batch['angles'] = _to_cuda(batch['angles'], dtype=torch.float32)
        if 'tool_params' in batch:
            batch['tool_params'] = _to_cuda(batch['tool_params'], dtype=torch.float32)

        return batch

    def _to_cuda_float_tensor(self, x):
        """转换为CUDA float32 tensor（稳健版，支持 list/ndarray/tensor）。"""
        import numpy as np
        if isinstance(x, torch.Tensor):
            return x.to(device='cuda', dtype=torch.float32, non_blocking=True)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device='cuda', dtype=torch.float32, non_blocking=True)
        if isinstance(x, list):
            if len(x) == 0:
                return torch.empty(0, device='cuda', dtype=torch.float32)
            try:
                t = torch.tensor(x, dtype=torch.float32)
            except Exception:
                # 逐元素兜底
                t = torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in x], dim=0)
            return t.to(device='cuda', non_blocking=True)
        raise TypeError(f'Unsupported type for CUDA conversion: {type(x)}')

    # ==================== 最优权重保存辅助 ====================
    def _is_better(self, cur):
        if cur is None:
            return False
        return (cur < self._best_value) if self._best_mode == 'min' else (cur > self._best_value)

    def _get_model_state(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return model.state_dict()

    def _save_best(self, epoch):
        # 仅主进程保存
        if not getattr(self, 'is_master', True):
            return
        os.makedirs(os.path.dirname(self._best_ckpt_path), exist_ok=True)
        torch.save(self._get_model_state(), self._best_ckpt_path)
        tqdm.write(f"=> Saved best model to {self._best_ckpt_path} | epoch {epoch} | {self._best_key}={self._best_value:.6f}")

   

    def model_forward(self, batch):
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

        angles =  batch['angles'] 

        # 获取刀具参数
        tool_params = batch['tool_params']

        # 前向传播
        score_pred = self.model.forward(
            data, octree, octree.depth, query_pts,
            angles, tool_params
        )  # [B]

        # 获取对应的GT分数
       
        score_gt = batch['labels']# [B]

        return score_pred, score_gt

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
#TODO: verify
    def relative_error(self, score_pred, score_gt):
        """相对误差（百分比）"""
        rel_err = torch.abs(score_pred - score_gt) / (score_gt.abs() + 1e-6)
        return (rel_err.mean()).item()

    # ==================== Train / Test Steps ====================
    def train_step(self, batch):
        print("train step")
        print("="*100)
        print(batch)
        print("="*100)

        """训练步骤"""
        batch = self.process_batch(batch, self.FLAGS.DATA.train)
        score_pred, score_gt= self.model_forward(batch)

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
            score_pred, score_gt= self.model_forward(batch)

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
            score_pred, scores_gt = self.model_forward(batch)

            

            # 保存结果
            filenames = batch['id_names']
            for i, fname in enumerate(filenames):
                self.eval_rst[fname] = {
                    'scores_pred': score_pred.cpu().numpy()[i],  
                    'scores_gt': scores_gt.cpu().numpy()[i],     
                    'tool_params': batch['tool_params'].cpu().numpy()[i],
                    'angles': batch['angles'].cpu().numpy()[i],
                    'id_name': fname
                }

                # 最后一个epoch保存
                if self.FLAGS.SOLVER.eval_epoch - 1 == batch['epoch']:
                    full_filename = os.path.join(
                        self.eval_output_dir, fname + '.scorer_eval.npz'
                    )
                    curr_folder = os.path.dirname(full_filename)
                    if not os.path.exists(curr_folder):
                        os.makedirs(curr_folder)
                    np.savez(
                        full_filename,
                        scores_pred=self.eval_rst[fname]['scores_pred'],
                        scores_gt=self.eval_rst[fname]['scores_gt'],
                        tool_params=self.eval_rst[fname]['tool_params'],
                        angles=self.eval_rst[fname]['angles'],
                        id_name=self.eval_rst[fname]['id_name']
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

        # 评估并自动保存当前最优模型（与基类保存机制互补；可共存）
        cur_metric = avg.get(self._best_key)
        cur_metric = _to_float(cur_metric, default=None)
        if cur_metric is None:
            # 若未能从tracker中取到配置的指标，则回退用 test/loss
            cur_metric = loss

        if self._is_better(cur_metric):
            self._best_value = cur_metric
            self._save_best(epoch)


if __name__ == "__main__":
    ScorerSolver.main()
