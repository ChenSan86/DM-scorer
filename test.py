import os
import glob
import numpy as np
# 解析/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/3370_60_b79.2.scorer_eval.npz
# 怎么打开这个文件并读取里面的内容
all_scores_pred = []
all_scores_gt = []
src_dir = "/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/"

def load_npz_data(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:

        scores_pred = data["scores_pred"]   # numpy.ndarray
        scores_gt = data["scores_gt"]
        tool_params = data["tool_params"]
        angles = data["angles"]
        id_name = data["id_name"]

        all_scores_pred.append(scores_pred)
        all_scores_gt.append(scores_gt)

        print(f"id_name: {id_name}")
        print("scores_pred", scores_pred)
        print("scores_gt", scores_gt)
        print("tool_params:", tool_params)
        print("angles", angles)
        print("-----")

for p in glob.glob("/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/*.scorer_eval.npz"):
   load_npz_data(p)