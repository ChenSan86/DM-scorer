# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# (Adjusted helper script for DeepMill segmentation run)
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='unet_d5')
# 改为列表：gpu 可多卡 [0] / [0,1] / [2] 等
parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='GPU ids, e.g. 0 or 0 1')
parser.add_argument('--depth', type=int, default=5)
parser.add_argument('--model', type=str, default='unet')
parser.add_argument('--mode', type=str, default='randinit')
# 用空字符串表示“无 ckpt”；如果为空就不传该键
parser.add_argument('--ckpt', type=str, default='')
# 预测比例：支持多个，比如 --ratios 1 0.5 0.1
parser.add_argument('--ratios', type=float, default=[1.0], nargs='*')

args = parser.parse_args()
alias = args.alias
gpus = args.gpu                  # list[int]
mode = args.mode
ratios = args.ratios

module = 'segmentation.py'
data = 'data_2.0'
logdir = 'logs/seg_deepmill'

categories = ['models']
names = ['models']
seg_num = [2]
train_num = [4471]
test_num = [1118]
max_epoches = [1500]

for i in range(len(ratios)):
    for k in range(len(categories)):
        ratio, cat = ratios[i], categories[k]

        mul = 2 if ratio < 0.1 else 1   # <10% 数据时加长训练
        max_epoch = max(1, int(max_epoches[k] * ratio * mul))
        milestone1, milestone2 = int(0.5 * max_epoch), int(0.25 * max_epoch)
        test_every_epoch = 50
        take = max(1, int(math.ceil(train_num[k] * ratio)))

        logs = os.path.join(logdir, f'{alias}/{cat}_{names[k]}/ratio_{ratio:.2f}')
        os.makedirs(logs, exist_ok=True)  # 确保目录存在

        # yacs 对列表的字符串形式兼容良好，这里显式传 list
        gpu_str = str(gpus)  # 例如 [0] 或 [0, 1]
        # milestones 保持逗号分隔（tuple）或改为列表都行；这里改成列表更稳
        milestones_str = f'[{milestone1}, {milestone2}]'

        cmds = [
            "python", module,
            "--config", "configs/seg_deepmill.yaml",
            "SOLVER.gpu", gpu_str,
            "SOLVER.logdir", logs,
            "SOLVER.max_epoch", str(max_epoch),
            "SOLVER.milestones", milestones_str,
            "SOLVER.test_every_epoch", str(test_every_epoch),
            # 只在 ckpt 非空时传入
            # "SOLVER.ckpt", args.ckpt,
            "DATA.train.depth", str(args.depth),
            "DATA.train.filelist", f"{data}/filelist/{cat}_train_val.txt",
            "DATA.train.take", str(take),
            "DATA.test.depth", str(args.depth),
            "DATA.test.filelist", f"{data}/filelist/{cat}_test.txt",
            "MODEL.stages", str(args.depth - 2),
            "MODEL.nout", str(seg_num[k]),
            "MODEL.name", args.model,
            "LOSS.num_class", str(seg_num[k]),
        ]
        if args.ckpt.strip():
            cmds.insert( cmds.index("DATA.train.depth"), "SOLVER.ckpt")
            cmds.insert( cmds.index("DATA.train.depth")+1, args.ckpt.strip())

        print('\n$ ' + ' '.join(cmds) + '\n')
        # 如果训练失败希望立刻抛错，可加 check=True
        subprocess.run(cmds, check=False)

# -------- 汇总阶段（更健壮：不存在就跳过） --------
summary = []
summary.append('names, ' + ', '.join(names) + ', C.mIoU, I.mIoU')
summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))

for i in range(len(ratios)-1, -1, -1):
    ious = [None] * len(categories)
    for j in range(len(categories)):
        filename = f'{logdir}/{alias}/{categories[j]}_{names[j]}/ratio_{ratios[i]:.2f}/log.csv'
        if not os.path.exists(filename):
            print(f'[WARN] log not found, skip: {filename}')
            continue
        with open(filename, newline='') as fid:
            lines = fid.readlines()
        # 兜底：找最后一行中 test/mIoU 的位置
        last_line = lines[-1]
        pos = last_line.find('test/mIoU:')
        if pos < 0:
            print(f'[WARN] no "test/mIoU:" in last line, skip: {filename}')
            continue
        try:
            # 这里用更稳的切分方式
            tail = last_line[pos+len('test/mIoU:'):].strip()
            # 取第一个能转成 float 的 token
            token = tail.split()[0].strip(',;')
            ious[j] = float(token)
        except Exception as e:
            print(f'[WARN] parse mIoU failed for {filename}: {e}')

    # 若本轮一个都没取到，跳过
    valid = [x for x in ious if isinstance(x, (int, float))]
    if not valid:
        summary.append(f'Ratio:{ratios[i]:.2f}, (no results)')
        continue

    # 计算 CmIoU / ImIoU（只对读到的项计算）
    vals = np.array([x if x is not None else 0.0 for x in ious], dtype=float)
    mask = np.array([x is not None for x in ious], dtype=bool)
    CmIoU = vals[mask].mean()
    ImIoU = (vals[mask] * np.array([test_num[t] for t in range(len(categories))])[mask]).sum() / \
            (np.array([test_num[t] for t in range(len(categories))])[mask].sum())

    ious_str = [f'{x:.3f}' if x is not None else 'NA' for x in ious] + \
               [f'{CmIoU:.3f}', f'{ImIoU:.3f}']
    summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(ious_str))

sum_file = f'{logdir}/{alias}/summaries.csv'
os.makedirs(os.path.dirname(sum_file), exist_ok=True)
with open(sum_file, 'w') as fid:
    summ = '\n'.join(summary)
    fid.write(summ)
    print('\n' + summ + '\n')
