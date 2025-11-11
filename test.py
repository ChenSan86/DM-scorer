import os
import glob
import numpy as np
# 解析/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/3370_60_b79.2.scorer_eval.npz
# 怎么打开这个文件并读取里面的内容
all_scores_pred = []
all_scores_gt = []

npz_path = "/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/0000_60_b79.2.scorer_eval.npz"
assert os.path.isfile(npz_path), f"文件不存在: {npz_path}"

with np.load(npz_path, allow_pickle=False) as data:
    # 查看可用键
    print("keys:", data.files)  # 例如: ['scores_pred', 'scores_gt', 'tool_params', 'angles', 'id_name']

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

# 如果要批量读取一个目录下所有 .scorer_eval.npz：
# for p in glob.glob("/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/output/*.scorer_eval.npz"):
#     with np.load(p, allow_pickle=False) as d:
#         all_scores_pred.append(d["scores_pred"])
#         all_scores_gt.append(d["scores_gt"])