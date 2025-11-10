import numpy as np

import os

#解析文件/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/scorer_baseline/models_models/ratio_1.00/models/3620_collision_detection.scorer_eval.npz
#projects/logs/scorer_deepmill/scorer_baseline/models_models/ratio_1.00/models/5331_collision_detection.scorer_eval.npz
data = np.load('/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/projects/logs/scorer_deepmill/scorer_baseline/models_models/ratio_1.00/models/2058_collision_detection.scorer_eval.npz')
#打印整个文件
print(data.files)
#打印各个数组的形状
for key in data.files:
    print(f"{key}: {data[key].shape}")
#打印各个数组的内容
for key in data.files:
    print(f"{key}内容: {data[key]}")
    