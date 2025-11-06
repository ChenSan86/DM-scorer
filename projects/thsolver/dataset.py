# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm
import json


def read_file(filename):
    points = np.fromfile(filename, dtype=np.uint8)
    return torch.from_numpy(points)   # convert it to torch.tensor


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, filelist, transform, read_file=read_file,
                 in_memory=False, take: int = -1):
        super(Dataset, self).__init__()
        self.root = root
        self.filelist = filelist
        self.transform = transform
        self.in_memory = in_memory
        self.read_file = read_file
        self.take = take  # 训练/测试时选取的样本数量
        self.rotation_matrices = self._load_rotation_matrices()  # 加载旋转矩阵

        self.filenames, self.labels, self.tool_params = self.load_filenames()  # 加载文件名、标签和刀具参数
        if self.in_memory:
            print('Load files into memory from ' + self.filelist)  # 打印加载信息
            self.samples = [self.read_file(os.path.join(self.root, f))
                            for f in tqdm(self.filenames, ncols=80, leave=False)]  # 预加载所有样本到内存

    def __len__(self):
        return len(self.filenames)  # 返回样本数量

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

    def __getitem__(self, idx):
        sample = (self.samples[idx] if self.in_memory else
                  self.read_file(os.path.join(self.root, self.filenames[idx])))  # 获取单个样本
        output = self.transform(sample, idx)  # 数据增强和octree构建
        output['label'] = self.labels[idx]  # 添加标签
        output['filename'] = self.filenames[idx]  # 添加文件名
        output['rotation_matrices'] = self.rotation_matrices  # 添加旋转矩阵
        filename = self.filenames[idx]
        basename = os.path.basename(filename)

        # 去掉后缀和_collision_detection
        model_name = basename.replace('_collision_detection.ply', '')
        label = read_six_dim_vector(model_name)
        # 这里确保刀具参数添加到输出中
        output['labels'] = np.array(label).astype(np.float32)
        output['tool_params'] = self.tool_params[idx]  # 假设在加载数据时已经填充
        return output  # 返回样本字典

    def load_filenames(self):
        filenames, labels, tool_params = [], [], []  # 初始化列表
        with open(self.filelist) as fid:
            lines = fid.readlines()  # 读取所有行
        for line in lines:
            tokens = line.split()  # 按空格分割
            filename = tokens[0].replace('\\', '/')  # 获取文件名并规范路径分隔符

            # 获取tool_params中的后4位
            if len(tokens) >= 2:
                label = tokens[1]  # 获取标签
                # 读取tool_params中的后4位并进行处理，假设tool_params是4维向量
                tool_param = tokens[-4:]  # 获取最后4位作为刀具参数
            else:
                label = 0  # 默认���签为0

            filenames.append(filename)  # 添加文件名
            labels.append(int(label))  # 添加标签
            tool_params.append(tool_param)  # 添加刀具参数

        num = len(filenames)  # 样本总数
        if self.take > num or self.take < 1:
            self.take = num  # 修正take参数
        result = (filenames[:self.take],
                  labels[:self.take], tool_params[:self.take])
        return result  # 返回指定数量的文件名、标签和刀具参数


def read_six_dim_vector(model_id, result_dir='/home/xinguanze/project/ex_4_dataset_making/result'):
    """
    根据模型ID，从result文件夹中读取338维向量，保留原始数据（不归一化）
    :param model_id: 模型ID（如 00181080_1b7d16dab26af7058f098574_trimesh_000）
    :param result_dir: result文件夹目录
    :return: 原始338维向量（list[float]），未找到则返回None
    """
    import numpy as np

    result_file = os.path.join(result_dir, f"{model_id}.txt")

    if not os.path.exists(result_file):
        return None

    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 读取338行浮点数
        vector = []
        for line in lines[:338]:  # 只取前338行
            line = line.strip()
            if line:
                try:
                    float_val = float(line)
                    vector.append(float_val)
                except ValueError:
                    return None

        # 如果不足338个，用0填充
        while len(vector) < 338:
            vector.append(0.0)

        # 转换为numpy数组并检查异常值
        arr = np.array(vector[:338], dtype=np.float64)

        # 检查NaN或Inf，替换为0
        arr = np.where(np.isnan(arr) | np.isinf(arr), 0.0, arr)

        # 直接返回原始数据，不进行归一化
        return arr.tolist()

    except Exception as e:
        print(f"读取文件 {result_file} 失败: {e}")
        return None
