# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='unet_d5', help='log alias')
parser.add_argument('--gpu', type=str, default='0', help='CUDA visible devices')
parser.add_argument('--depth', type=int, default=5, help='octree depth')
parser.add_argument('--model', type=str, default='unet', help='segnet | unet')
parser.add_argument('--mode', type=str, default='randinit')  # kept for compatibility
parser.add_argument('--ckpt', type=str, default='', help='checkpoint path or empty')
parser.add_argument('--ratios', type=float, default=[1.0], nargs='*', help='train ratios')
# 便捷开关：是否用 Encoder-only(单层)
parser.add_argument('--encoder_only', action='store_true', help='override MODEL to encoder-only single layer')

args = parser.parse_args()

alias = args.alias
gpu = args.gpu
ratios = args.ratios

module = 'segmentation.py'
config_path = 'configs/seg_deepmill.yaml'  # 正确读取 YAML 的关键：仅传 --config
script_base = ['python', module, '--config', config_path]

data_root = 'data_2.0'
log_root = 'logs/seg_deepmill'

categories = ['models']
names = ['models']
seg_num = [2]      # 原项目字段保留（不会影响姿态训练）
train_num = [4451]
test_num = [1112]
max_epoches = [1500]

def build_cmd_list(
    logdir: str, max_epoch: int, milestone1: int, milestone2: int,
    take: int, cat: str, depth: int, test_every_epoch: int,
    encoder_only: bool
):
    """返回传给 subprocess.run 的参数列表（list[str]）"""
    # 与 YAML 合作：只覆写必要项，其余从 YAML 读取
    cmd = script_base + [

        'SOLVER.logdir', logdir,
        'SOLVER.max_epoch', str(max_epoch),
        # 里程碑记得升序，并作为一个整体字符串传入
        'SOLVER.milestones', f'({min(milestone1, milestone2)},{max(milestone1, milestone2)})',
        'SOLVER.test_every_epoch', str(test_every_epoch),
        'SOLVER.ckpt', (args.ckpt if args.ckpt != '' else "''"),
        'DATA.train.depth', str(depth),
        'DATA.train.filelist', f'{data_root}/filelist/{cat}_train_val.txt',
        'DATA.train.take', str(take),
        'DATA.test.depth', str(depth),
        'DATA.test.filelist', f'{data_root}/filelist/{cat}_test.txt',
        # 使用 UNet + 6D 输出（姿态）
        'MODEL.name', 'unet',             # 建议为 'unet'
        'MODEL.nout', '6',
        # 验证指标：最小化测试集平均角误差
        'SOLVER.best_val', 'min:loss',
        # 频道与插值方式（与你 YAML 一致即可；如 YAML 已设置可省略）
        'MODEL.channel', '4',
        'MODEL.interp', 'linear',
    ]
    

    if encoder_only:
        cmd += [
            'MODEL.use_decoder', 'false',
            'MODEL.pyramid_levels', '(0)',   # 只用最深编码层
            'MODEL.tool_fusion', 'concat',
            'MODEL.use_attention_pool', 'false',
            'MODEL.use_tanh_head', 'true',
        ]

    # 可保留（对 UNet 姿态头无影响，留着不报错）
    # cmd += ['MODEL.stages', str(depth - 2)]
    # 旧分割字段，兼容留存
    cmd += ['LOSS.num_class', str(seg_num[0])]

    return cmd


def main():
    test_every_epoch = 10  
    for i in range(len(ratios)):
        for k in range(len(categories)):
            ratio, cat = ratios[i], categories[k]
            mul = 2 if ratio < 0.1 else 1
            max_epoch = int(max_epoches[k] * ratio * mul)
            # 用升序里程碑：25% 和 50% 处
            milestone2 = int(0.25 * max_epoch)
            milestone1 = int(0.50 * max_epoch)
            take = int(math.ceil(train_num[k] * ratio))
            logdir = os.path.join(log_root, f'{alias}/{cat}_{names[k]}/ratio_{ratio:.2f}')

            cmd_list = build_cmd_list(
                logdir=logdir,
                max_epoch=max_epoch,
                milestone1=milestone1,
                milestone2=milestone2,
                take=take,
                cat=cat,
                depth=args.depth,
                test_every_epoch=test_every_epoch,
                encoder_only=args.encoder_only,
            )

            print('\n>>> Launch command (list form):\n', cmd_list, '\n')
            # 关键：list 形式，不再拼接字符串，避免转义/逗号/空格问题
            subprocess.run(cmd_list, check=False)

    # 训练完成后做简要汇总（读取 test/mean_error 等）
    summary = []
    summary.append('names, ' + ', '.join(names) + ', mean_error(rad), max_error(rad), std(rad)')
    summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
    summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

    for i in range(len(ratios) - 1, -1, -1):
        means, maxes, stds = [None]*len(categories), [None]*len(categories), [None]*len(categories)
        for j in range(len(categories)):
            filename = f'{log_root}/{alias}/{categories[j]}_{names[j]}/ratio_{ratios[i]:.2f}/log.csv'
            if not os.path.exists(filename):
                print(f'[WARN] log not found, skip: {filename}')
                continue
            try:
                with open(filename, newline='') as fid:
                    lines = fid.readlines()
                if not lines:
                    print(f'[WARN] log empty, skip: {filename}')
                    continue
                last_line = lines[-1]

                def _grab(tag: str, line: str, default=np.nan):
                    pos = line.find(tag)
                    if pos < 0:
                        return default
                    s = line[pos + len(tag):].strip().split(',')[0].split()[0]
                    try:
                        return float(s)
                    except Exception:
                        return default

                means[j] = _grab('test/mean_error:', last_line)
                maxes[j] = _grab('test/max_error:', last_line)
                stds[j]  = _grab('test/standard_deviation:', last_line)
            except Exception as e:
                print(f'[WARN] parse log failed: {filename}, err={e}')

        arr_means = np.array([x for x in means if x is not None], dtype=float)
        if arr_means.size == 0:
            continue
        Cm = np.nanmean(arr_means)
        Im = np.nansum(arr_means * np.array(test_num[:arr_means.size])) / np.sum(np.array(test_num[:arr_means.size]))
        row = ['{:.4f}'.format(x) for x in arr_means] + ['{:.4f}'.format(Cm), '{:.4f}'.format(Im)]
        summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(row))

    os.makedirs(f'{log_root}/{alias}', exist_ok=True)
    out_csv = f'{log_root}/{alias}/summaries.csv'
    with open(out_csv, 'w') as fid:
        summ = '\n'.join(summary)
        fid.write(summ)
        print('\n' + summ + '\n')

if __name__ == '__main__':
    main()

#dm-decoder-pool-mlp-experiment2/projects/logs/seg_deepmill/unet_d5/models_models/ratio_1.00/checkpoints