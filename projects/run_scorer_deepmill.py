#!/usr/bin/env python
# --------------------------------------------------------
# Run Scorer Network Training
# Copyright (c) 2025
# --------------------------------------------------------

import os
import math
import argparse
import numpy as np
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--alias', type=str, default='scorer_baseline', help='log alias')
parser.add_argument('--gpu', type=str, default='3', help='CUDA visible devices')
parser.add_argument('--depth', type=int, default=5, help='octree depth')
parser.add_argument('--ckpt', type=str, default='/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/score_baseline_1500_epoch_ora_angle/models_models/ratio_1.00/best_test_mae.pth', help='checkpoint path')
parser.add_argument('--ratios', type=float, default=[1.0], nargs='*', help='train ratios')

args = parser.parse_args()

alias = args.alias
gpu = args.gpu
ratios = args.ratios

module = 'scorer_solver.py'
config_path = 'configs/scorer_deepmill.yaml'
script_base = ['python', module, '--config', config_path]

data_root = 'data_2.0'
log_root = 'logs/scorer_deepmill'

categories = ['models']
names = ['models']
train_num = [4451]
test_num = [1112]
max_epoches = [1500]


def build_cmd_list(
    logdir: str, max_epoch: int, milestone1: int, milestone2: int,
    take: int, cat: str, depth: int, test_every_epoch: int
):
    """返回传给 subprocess.run 的参数列表（list[str]）"""
    cmd = script_base + [
        'SOLVER.logdir', logdir,
        'SOLVER.max_epoch', str(max_epoch),
        'SOLVER.milestones', f'({min(milestone1, milestone2)},{max(milestone1, milestone2)})',
        'SOLVER.test_every_epoch', str(test_every_epoch),
        'SOLVER.ckpt', (args.ckpt if args.ckpt != '' else "''"),
        'DATA.train.depth', str(depth),
        'DATA.train.filelist', f'{data_root}/filelist/{cat}_train_val.txt',
        'DATA.train.take', str(take),
        'DATA.test.depth', str(depth),
        'DATA.test.filelist', f'{data_root}/filelist/test.txt',
    ]
    
    return cmd


def main():
    test_every_epoch = 10
    for i in range(len(ratios)):
        for k in range(len(categories)):
            ratio, cat = ratios[i], categories[k]
            mul = 2 if ratio < 0.1 else 1
            max_epoch = int(max_epoches[k] * ratio * mul)
            
            # 里程碑：40%, 70%
            milestone1 = int(0.40 * max_epoch)
            milestone2 = int(0.70 * max_epoch)
            
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
            )
            
            print('\n' + '='*80)
            print(f'Training Scorer Network - Ratio: {ratio:.2f}')
            print('='*80)
            print('Command:', ' '.join(cmd_list))
            print('='*80 + '\n')
            
            subprocess.run(cmd_list, check=False)
    
    # 训练完成后做简要汇总
    summary = []
    summary.append('names, ' + ', '.join(names) + ', MAE, RMSE, Rel.Err(%)')
    summary.append('train_num, ' + ', '.join([str(x) for x in train_num]))
    summary.append('test_num, ' + ', '.join([str(x) for x in test_num]))
    
    for i in range(len(ratios) - 1, -1, -1):
        maes, rmses, rel_errs = [None]*len(categories), [None]*len(categories), [None]*len(categories)
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
                
                maes[j] = _grab('test/mae:', last_line)
                rmses[j] = _grab('test/rmse:', last_line)
                rel_errs[j] = _grab('test/rel_error:', last_line)
            except Exception as e:
                print(f'[WARN] parse log failed: {filename}, err={e}')
        
        arr_maes = np.array([x for x in maes if x is not None], dtype=float)
        if arr_maes.size == 0:
            continue
        
        Cm = np.nanmean(arr_maes)
        Im = np.nansum(arr_maes * np.array(test_num[:arr_maes.size])) / np.sum(np.array(test_num[:arr_maes.size]))
        
        row_vals = ['{:.6f}'.format(x) if x is not None else 'N/A' for x in maes]
        row_vals += ['{:.6f}'.format(Cm), '{:.6f}'.format(Im)]
        summary.append('Ratio:{:.2f}, '.format(ratios[i]) + ', '.join(row_vals))
    
    os.makedirs(f'{log_root}/{alias}', exist_ok=True)
    out_csv = f'{log_root}/{alias}/summaries.csv'
    with open(out_csv, 'w') as fid:
        summ = '\n'.join(summary)
        fid.write(summ)
        print('\n' + '='*80)
        print('Training Summary:')
        print('='*80)
        print(summ)
        print('='*80)
        print(f'\nSummary saved to: {out_csv}\n')


if __name__ == '__main__':
    main()

