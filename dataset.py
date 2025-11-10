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

        self.filenames, self.labels, self.tool_params ,self.angles,self.id_names,self.scores= self.load_filenames()  # 加载文件名、标签和刀具参数
        if self.in_memory:
            print('Load files into memory from ' + self.filelist)  # 打印加载信息
            self.samples = [self.read_file(os.path.join(self.root, f))
                            for f in tqdm(self.filenames, ncols=80, leave=False)]  # 预加载所有样本到内存

    def __len__(self):
        return len(self.filenames)  # 返回样本数量

    
    def __getitem__(self, idx):
        sample = (self.samples[idx] if self.in_memory else
                  self.read_file(os.path.join(self.root, self.filenames[idx])))  # 获取单个样本
        output = self.transform(sample, idx)  # 数据增强和octree构建
        output['label'] = self.labels[idx]  # 添加标签
        output['filename'] = self.filenames[idx]  # 添加文件名
        filename = self.filenames[idx]
        basename = os.path.basename(filename)
        model_name = basename.replace('_collision_detection.ply', '')
        tool_vals = torch.tensor(self.tool_params[idx], dtype=torch.float32)  # 列表/字符串
        angles = torch.tensor(self.angles[idx], dtype=torch.float32)  # 列表/字符串
        id_names = self.id_names[idx]
        scores = torch.tensor(self.scores[idx], dtype=torch.float32)
        output['tool_params'] = torch.tensor(tool_vals, dtype=torch.float32)  # [4]
        output['angles'] = angles  # [2]
        output['id_names'] = id_names
        output['labels'] = scores
        return output  # 返回样本字典

    def load_filenames(self):
        filenames, labels, tool_params,angles,id_names,scores= [], [], [] ,[] ,[] ,[] 
        with open(self.filelist) as fid:
            lines = fid.readlines()  # 读取所有行
        for line in lines:
            tokens = line.split()  # 按空格分割
            filename = tokens[0].replace('\\', '/')  # 获取文件名并规范路径分隔符

            # 获取tool_params中的后4位
            if len(tokens) >= 2:
                label = tokens[1]  # 获取标签
                # 读取tool_params中的后4位并进行处理，假设tool_params是4维向量
                tool_param = tokens[2:6]  # 获取最后4位作为刀具参数
                #专程double
                tool_param = [float(tokens[2]), float(tokens[3]), float(tokens[4]), float(tokens[5])]
            else:
                label = 0  # 默认���签为0
            #转乘double
            angle = [float(tokens[6]), float(tokens[7])]
            id_name = tokens[9]
            score = float(tokens[8])


            filenames.append(filename)  # 添加文件名
            labels.append(int(label))  # 添加标签
            tool_params.append(tool_param)  # 添加刀具参数
            angles.append(angle)
            id_names.append(id_name)
            scores.append(score)
            
        num = len(filenames)  # 样本总数
        if self.take > num or self.take < 1:
            self.take = num  # 修正take参数
        result = (filenames[:self.take],
                  labels[:self.take], tool_params[:self.take],angles[:self.take],id_names[:self.take],scores[:self.take])
        return result  # 返回指定数量的文件名、标签和刀具参数

if __name__ == '__main__':
    dataset = Dataset(root='.', filelist='data_filelist/test.txt', transform=None, in_memory=False, take=4)
    print(len(dataset))
    print(dataset.__getitem__(0))
    
'''/home/group1/xinguanze/project/deepmill_scorer/DM-scorer/dataset.py:57: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  output['tool_params'] = torch.tensor(tool_vals, dtype=torch.float32)  # [4]
{'label': 0, 'filename': 'models/0000_collision_detection.ply', 'tool_params': tensor([ 1.7185, 16.1724, 32.9369, 17.6935]), 'angles': tensor([ 60.0000, -79.2000]), 'id_names': '0000_60_b79.2', 'labels': tensor(0.2835)}'''
    