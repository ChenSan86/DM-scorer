# Batch 数据加工打包流程详解

## 一、概述

Batch 数据从原始文件到模型输入要经过 **5个关键阶段**：

```
原始文件 → Dataset.__getitem__ → Transform → CollateBatch → process_batch → 模型输入
   .ply         单样本加载        数据增强      批量打包      八叉树构建     forward
```

---

## 二、完整数据流详解

### 阶段 0: 数据集配置 (get_seg_shapenet_dataset)

**位置**: `projects/datasets/seg_shapenet.py`

```python
def get_seg_shapenet_dataset(flags):
    # 1. 创建Transform（数据增强）
    transform = ShapeNetTransform(flags)
    
    # 2. 创建文件读取器
    read_ply = ReadPly(has_normal=True, has_label=True)
    
    # 3. 创建Collate函数（批量打包）
    collate_batch = CollateBatch(merge_points=True)
    
    # 4. 创建Dataset
    dataset = Dataset(
        flags.location,      # 'data_2.0/points'
        flags.filelist,      # 'data_2.0/filelist/models_train_val.txt'
        transform,
        read_file=read_ply,
        take=flags.take
    )
    
    return dataset, collate_batch
```

**配置到DataLoader** (`thsolver/solver.py`):

```python
def get_dataloader(self, flags):
    # 获取dataset和collate_fn
    dataset, collate_fn = self.get_dataset(flags)
    
    # 创建采样器
    if self.world_size > 1:
        sampler = DistributedInfSampler(dataset, shuffle=flags.shuffle)
    else:
        sampler = InfSampler(dataset, shuffle=flags.shuffle)
    
    # 创建DataLoader（关键：collate_fn参数）
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=flags.batch_size,     # 例如: 8
        num_workers=flags.num_workers,   # 例如: 0
        sampler=sampler, 
        collate_fn=collate_fn,          # ← 这里指定批量打包函数！
        pin_memory=flags.pin_memory
    )
    return data_loader
```

---

### 阶段 1: 单样本加载 (Dataset.__getitem__)

**位置**: `projects/thsolver/dataset.py`

```python
class Dataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        # 1. 读取点云文件（通过ReadPly）
        sample = self.read_file(os.path.join(self.root, self.filenames[idx]))
        # sample = {
        #     'points': np.array([N, 3]),      # 点云坐标
        #     'normals': np.array([N, 3]),     # 法线
        #     'labels': np.array([338])        # 打分表（从result文件夹读取）
        # }
        
        # 2. 数据增强和octree构建（通过Transform）
        output = self.transform(sample, idx)
        # output = {
        #     'points': Points对象,
        #     'inbox_mask': tensor([N]),
        #     'octree': Octree对象
        # }
        
        # 3. 添加额外字段
        output['label'] = self.labels[idx]           # 类别ID（通常为0）
        output['filename'] = self.filenames[idx]     # 文件名
        output['labels'] = self.labels[idx]          # 打分表（会被覆盖）
        output['tool_params'] = self.tool_params[idx] # 刀具参数（字符串列表）
        
        # 4. 重新读取打分表（覆盖）
        filename = self.filenames[idx]
        basename = os.path.basename(filename)
        model_name = basename.replace('_collision_detection.ply', '')
        label = read_six_dim_vector(model_name)  # 从result文件夹读取
        output['labels'] = np.array(label).astype(np.float32)
        
        return output
```

**输出示例**（单个样本）:
```python
{
    'points': <Points对象>,                    # 点云 + 法线
    'inbox_mask': tensor([True, True, ...]),   # [N] 是否在[-1,1]内
    'octree': <Octree对象>,                    # 八叉树结构
    'label': 0,                                # 类别ID
    'filename': 'models/xxx.ply',              # 文件名
    'labels': array([338维打分表], float32),    # 打分表
    'tool_params': ['-0.18', '-0.56', ...]    # 4维刀具参数（字符串）
}
```

---

#### 1.1 ReadPly: 读取原始文件

**位置**: `projects/datasets/utils.py`

```python
class ReadPly:
    def __init__(self, has_normal=True, has_label=True):
        self.has_normal = has_normal
        self.has_label = has_label
    
    def __call__(self, filename):
        # 1. 读取PLY文件
        plydata = PlyData.read(filename)
        vtx = plydata['vertex']
        
        output = {}
        
        # 2. 提取点坐标
        points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
        output['points'] = points.astype(np.float32)  # [N, 3]
        
        # 3. 提取法线
        if self.has_normal:
            normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
            output['normals'] = normal.astype(np.float32)  # [N, 3]
        
        # 4. 读取打分表（从result文件夹）
        if self.has_label:
            basename = os.path.basename(filename)
            model_name = basename.replace('_collision_detection.ply', '')
            label = read_six_dim_vector(model_name)
            output['labels'] = np.array(label).astype(np.float32)  # [338]
        
        return output
```

**read_six_dim_vector 函数**:
```python
def read_six_dim_vector(model_id, result_dir='/home/.../result'):
    """从result文件夹读取338维打分表并min-max归一化"""
    result_file = os.path.join(result_dir, f"{model_id}.txt")
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    # 读取338行浮点数
    vector = []
    for line in lines[:338]:
        vector.append(float(line.strip()))
    
    # min-max归一化到[0,1]
    arr = np.array(vector, dtype=np.float64)
    min_val, max_val = np.min(arr), np.max(arr)
    
    if max_val - min_val == 0:
        normalized = np.full_like(arr, 0.5)
    else:
        normalized = (arr - min_val) / (max_val - min_val)
    
    return normalized.tolist()  # [338]
```

---

#### 1.2 Transform: 数据增强 + Octree构建

**位置**: `projects/ocnn/dataset.py` + `projects/datasets/seg_shapenet.py`

```python
class Transform:
    def __call__(self, sample, idx):
        # 1. 预处理：numpy → Points对象
        output = self.preprocess(sample, idx)
        
        # 2. 数据增强：旋转、缩放、抖动等
        output = self.transform(output, idx)
        
        # 3. 构建Octree
        output['octree'] = self.points2octree(output['points'])
        
        return output
```

**详细流程**:

##### Step 1: preprocess (ShapeNetTransform)
```python
def preprocess(self, sample, idx):
    # numpy数组 → torch.Tensor
    xyz = torch.from_numpy(sample['points']).float()     # [N, 3]
    normal = torch.from_numpy(sample['normals']).float() # [N, 3]
    labels = torch.from_numpy(sample['labels']).float()  # [338]
    
    # 封装为Points对象
    points = Points(xyz, normal)
    
    return {'points': points}
```

##### Step 2: transform（数据增强）
```python
def transform(self, sample, idx):
    points = sample['points']
    
    # 如果启用数据增强（distort=True）
    if self.distort:
        # 生成随机参数
        rng_angle, rng_scale, rng_jitter, rnd_flip = self.rnd_parameters()
        
        # 1. 翻转
        points.flip(rnd_flip)  # 'x', 'y', 'z'的组合
        
        # 2. 旋转
        points.rotate(rng_angle)  # [angle_x, angle_y, angle_z] in radians
        
        # 3. 平移（抖动）
        points.translate(rng_jitter)  # [jitter_x, jitter_y, jitter_z]
        
        # 4. 缩放
        points.scale(rng_scale)  # [scale_x, scale_y, scale_z]
    
    # 法线方向对齐（如果指定）
    if self.orient_normal:
        points.orient_normal(self.orient_normal)  # 'xyz'
    
    # 裁剪到[-1, 1]范围（重要！）
    inbox_mask = points.clip(min=-1, max=1)
    
    sample.update({'points': points, 'inbox_mask': inbox_mask})
    return sample
```

**数据增强参数** (`seg_deepmill.yaml`):
```yaml
DATA:
  train:
    distort: True
    angle: (0, 5, 0)      # y轴旋转 ±5°
    interval: (1, 1, 1)   # 旋转步长
    scale: 0.25           # 缩放范围 [0.75, 1.25]
    jitter: 0.25          # 平移范围 [-0.25, 0.25]
    uniform: True         # 均匀缩放
```

##### Step 3: points2octree（构建八叉树）
```python
def points2octree(self, points):
    # 创建Octree对象
    octree = Octree(self.depth, self.full_depth)
    # depth=5: 最大深度，2^5=32分辨率
    # full_depth=2: 前2层全填充
    
    # 从Points对象构建八叉树
    octree.build_octree(points)
    
    return octree
```

**Octree构建原理**:
```
空间范围: [-1, 1]³

depth=0: 1个节点（整个空间）
depth=1: 8个节点（2×2×2划分）
depth=2: 64个节点（4×4×4划分，全填充）
depth=3: 按点云分布稀疏填充
depth=4: 按点云分布稀疏填充
depth=5: 按点云分布稀疏填充（最大32×32×32分辨率）

只有包含点云的体素才会创建节点 → 稀疏表示
```

---

### 阶段 2: 批量打包 (CollateBatch.__call__)

**位置**: `projects/ocnn/dataset.py`

**触发时机**: DataLoader内部自动调用 `collate_fn(batch_list)`

**输入**: 长度为 `batch_size` 的列表，每个元素是 `Dataset.__getitem__` 的输出

```python
class CollateBatch:
    def __init__(self, merge_points=False):
        self.merge_points = merge_points  # 是否合并Points对象
    
    def __call__(self, batch):
        """
        输入: batch = [sample_0, sample_1, ..., sample_{B-1}]
        每个sample是一个dict，包含：
            'points', 'inbox_mask', 'octree', 'label', 
            'filename', 'labels', 'tool_params'
        """
        outputs = {}
        
        # 1. 遍历所有字段
        for key in batch[0].keys():
            # 收集该字段的所有值
            outputs[key] = [b[key] for b in batch]
            
            # 2. 特殊处理：合并octree
            if 'octree' in key:
                octree = ocnn.octree.merge_octrees(outputs[key])
                octree.construct_all_neigh()  # 构建邻域索引
                outputs[key] = octree
            
            # 3. 特殊处理：合并points
            if 'points' in key and self.merge_points:
                outputs[key] = ocnn.octree.merge_points(outputs[key])
            
            # 4. 特殊处理：转换label为tensor
            if 'label' == key:
                outputs['label'] = torch.tensor(outputs[key])  # [B]
            
            # 5. 特殊处理：转换labels（打分表）为tensor
            if 'labels' == key:
                arr = np.asarray(outputs[key])  # [B, 338]
                outputs['labels'] = torch.from_numpy(arr).to(torch.float32)
        
        return outputs
```

**输入示例** (batch_size=2):
```python
batch = [
    {  # sample 0
        'points': <Points对象0>,
        'octree': <Octree对象0>,
        'label': 0,
        'labels': array([338维], float32),
        'tool_params': ['-0.18', '-0.56', '-0.62', '-0.53'],
        'filename': 'models/00180129_xxx.ply',
        'inbox_mask': tensor([True, True, ...])
    },
    {  # sample 1
        'points': <Points对象1>,
        'octree': <Octree对象1>,
        'label': 0,
        'labels': array([338维], float32),
        'tool_params': ['-0.19', '-0.09', '0.62', '-0.77'],
        'filename': 'models/00182021_xxx.ply',
        'inbox_mask': tensor([True, True, ...])
    }
]
```

**输出示例** (打包后的batch):
```python
outputs = {
    'points': <合并的Points对象>,          # 所有点云合并，附带batch_id
    'octree': <合并的Octree对象>,          # 超级八叉树
    'label': tensor([0, 0]),              # [B]
    'labels': tensor([[338维], [338维]]), # [B, 338]
    'tool_params': [
        ['-0.18', '-0.56', '-0.62', '-0.53'],
        ['-0.19', '-0.09', '0.62', '-0.77']
    ],  # List[List[str]]
    'filename': [
        'models/00180129_xxx.ply',
        'models/00182021_xxx.ply'
    ],  # List[str]
    'inbox_mask': [
        tensor([True, True, ...]),
        tensor([True, True, ...])
    ]  # List[Tensor]
}
```

---

#### 关键操作详解

##### 2.1 merge_octrees: 合并八叉树

```python
octree = ocnn.octree.merge_octrees([octree_0, octree_1, ..., octree_{B-1}])
```

**原理**:
- 将多个独立的八叉树合并成一个"超级八叉树"
- 每个原始八叉树的节点会被标记上 `batch_id`
- 合并后的八叉树节点总数 = Σ(各个八叉树的节点数)

**示例**:
```
octree_0: 1000个节点 (batch_id=0)
octree_1: 1200个节点 (batch_id=1)
octree_2: 900个节点  (batch_id=2)
                ↓
merged_octree: 3100个节点
  - nodes[0:1000]    → batch_id=0
  - nodes[1000:2200] → batch_id=1
  - nodes[2200:3100] → batch_id=2
```

##### 2.2 merge_points: 合并点云

```python
merged_points = ocnn.octree.merge_points([points_0, points_1, ..., points_{B-1}])
```

**原理**:
- 将多个 `Points` 对象的坐标和法线拼接
- 附加 `batch_id` 字段标识来源

**示例**:
```
points_0: [N0, 3] 坐标 + [N0, 3] 法线
points_1: [N1, 3] 坐标 + [N1, 3] 法线
                ↓
merged_points:
  - points:   [N0+N1, 3] 坐标
  - normals:  [N0+N1, 3] 法线
  - batch_id: [N0+N1, 1]  (前N0个为0，后N1个为1)
```

##### 2.3 construct_all_neigh: 构建邻域

```python
octree.construct_all_neigh()
```

**作用**: 为八叉树卷积预先计算邻域索引
- 每个节点需要知道其26个邻居（3D卷积核）
- 预计算避免运行时查找，加速卷积操作

---

### 阶段 3: 进一步处理 (process_batch)

**位置**: `projects/segmentation.py`

**触发时机**: 训练/测试循环中，batch从DataLoader取出后

```python
def process_batch(self, batch, flags):
    """
    进一步处理batch，确保数据在GPU上，
    如果batch中没有octree，则现场构建
    """
    def points2octree(points):
        octree = ocnn.octree.Octree(flags.depth, flags.full_depth)
        octree.build_octree(points)
        return octree
    
    # 情况1: batch已经包含octree（CollateBatch已处理）
    if 'octree' in batch:
        batch['octree'] = batch['octree'].cuda(non_blocking=True)
        batch['points'] = batch['points'].cuda(non_blocking=True)
    
    # 情况2: batch没有octree，需要现场构建
    else:
        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['points'] = ocnn.octree.merge_points(points)
        batch['octree'] = octree
    
    return batch
```

**当前项目**: CollateBatch已经处理好octree，所以走情况1

**输出**: batch的所有tensor都移动到GPU上

---

### 阶段 4: 模型前向传播准备 (model_forward)

**位置**: `projects/segmentation.py`

```python
def model_forward(self, batch):
    # 1. 提取八叉树和点云
    octree, points = batch['octree'], batch['points']
    
    # 2. 提取输入特征（从octree）
    data = self.get_input_feature(octree)
    # data: [N_nodes, C_in] 八叉树节点特征
    
    # 3. 构造查询点（坐标 + batch_id）
    query_pts = torch.cat([points.points, points.batch_id], dim=1)
    # query_pts: [N_total_points, 4]
    
    # 4. 转换刀具参数为tensor
    tool_params = self._to_cuda_float_tensor(batch['tool_params'])
    # tool_params: [B, 4] float32
    
    # 5. 模型前向传播
    logit = self.model.forward(
        data,         # [N_nodes, C_in]
        octree,       # Octree对象
        octree.depth, # int
        query_pts,    # [N_pts, 4]
        tool_params   # [B, 4]
    )
    # logit: [B, 2] - [pitch, roll] in degrees
    
    # 6. 转换标签（打分表）为tensor
    labels = self._to_cuda_float_tensor(batch['labels'])
    # labels: [B, 338]
    
    return logit, labels
```

---

## 三、完整数据形状变化表

| 阶段 | 数据项 | 形状/类型 | 设备 | 说明 |
|------|--------|-----------|------|------|
| **原始文件** |
| .ply文件 | 点云坐标 | [N, 3] numpy | CPU | 原始PLY文件 |
| .ply文件 | 法线 | [N, 3] numpy | CPU | 顶点法线 |
| result/*.txt | 打分表 | [338] numpy | CPU | 归一化后的分数 |
| filelist/*.txt | 刀具参数 | [4] str | CPU | 字符串格式 |
| **Dataset.__getitem__** |
| points | Points对象 | xyz:[N,3], normal:[N,3] | CPU | 封装对象 |
| octree | Octree对象 | 稀疏结构 | CPU | 八叉树 |
| labels | numpy | [338] float32 | CPU | 打分表 |
| tool_params | list | [4] str | CPU | 字符串列表 |
| **CollateBatch** |
| points | Points对象 | xyz:[ΣN,3], batch_id:[ΣN,1] | CPU | 合并+batch_id |
| octree | Octree对象 | 超级八叉树 | CPU | 合并所有batch |
| labels | torch.Tensor | [B, 338] float32 | CPU | 打包成tensor |
| tool_params | list | [B, 4] str | CPU | 嵌套列表 |
| **process_batch** |
| points | Points对象 | xyz:[ΣN,3], batch_id:[ΣN,1] | CUDA | 移到GPU |
| octree | Octree对象 | 超级八叉树 | CUDA | 移到GPU |
| labels | torch.Tensor | [B, 338] float32 | CUDA | 移到GPU |
| tool_params | list | [B, 4] str | CPU | 暂未转换 |
| **model_forward** |
| data | torch.Tensor | [N_nodes, C_in] | CUDA | 八叉树特征 |
| query_pts | torch.Tensor | [ΣN, 4] | CUDA | 点坐标+batch_id |
| tool_params | torch.Tensor | [B, 4] float32 | CUDA | 转换为tensor |
| labels | torch.Tensor | [B, 338] float32 | CUDA | 打分表 |

---

## 四、关键类和函数速查

### 4.1 数据集相关

| 类/函数 | 位置 | 作用 |
|---------|------|------|
| `Dataset` | `thsolver/dataset.py` | 数据集基类 |
| `ReadPly` | `datasets/utils.py` | 读取PLY文件 |
| `ShapeNetTransform` | `datasets/seg_shapenet.py` | 数据预处理 |
| `get_seg_shapenet_dataset` | `datasets/seg_shapenet.py` | 创建数据集 |

### 4.2 Transform相关

| 类/函数 | 位置 | 作用 |
|---------|------|------|
| `Transform` | `ocnn/dataset.py` | 数据增强基类 |
| `preprocess` | Transform方法 | numpy→Points |
| `transform` | Transform方法 | 旋转/缩放/抖动 |
| `points2octree` | Transform方法 | 构建八叉树 |

### 4.3 批量打包相关

| 类/函数 | 位置 | 作用 |
|---------|------|------|
| `CollateBatch` | `ocnn/dataset.py` | 批量打包 |
| `merge_octrees` | ocnn.octree | 合并八叉树 |
| `merge_points` | ocnn.octree | 合并点云 |
| `construct_all_neigh` | Octree方法 | 构建邻域 |

### 4.4 训练相关

| 类/函数 | 位置 | 作用 |
|---------|------|------|
| `get_dataloader` | `thsolver/solver.py` | 创建DataLoader |
| `process_batch` | `segmentation.py` | 移到GPU |
| `model_forward` | `segmentation.py` | 准备模型输入 |

---

## 五、典型batch示例

### 5.1 batch_size=2 的完整示例

```python
# CollateBatch输出后的batch:
batch = {
    # 合并的点云（带batch_id）
    'points': <Points对象>
        points.points:   torch.Tensor([2048, 3])   # 所有点坐标
        points.normals:  torch.Tensor([2048, 3])   # 所有法线
        points.batch_id: torch.Tensor([2048, 1])   # [0,0,...0,1,1,...1]
    
    # 合并的八叉树
    'octree': <Octree对象>
        octree.depth: 5
        octree.batch_size: 2
        octree.nnum: [总节点数]  # 例如 2200
        octree.batch_nnum: [[nodes_d1_b0, nodes_d1_b1], 
                           [nodes_d2_b0, nodes_d2_b1], ...]
    
    # 打分表 [B, 338]
    'labels': torch.Tensor([
        [0.804, 0.563, 0.185, ..., 0.535],  # sample 0
        [0.978, 0.091, 0.188, ..., 0.777]   # sample 1
    ])
    
    # 刀具参数（字符串列表）
    'tool_params': [
        ['-0.185', '-0.564', '-0.629', '-0.535'],
        ['-0.188', '-0.091', '0.622', '-0.778']
    ]
    
    # 文件名
    'filename': [
        'models/00180129_xxx_collision_detection.ply',
        'models/00182021_xxx_collision_detection.ply'
    ]
    
    # 类别标签
    'label': torch.Tensor([0, 0])
    
    # inbox mask（每个样本独立）
    'inbox_mask': [
        torch.Tensor([True, True, ..., True]),  # sample 0
        torch.Tensor([True, True, ..., True])   # sample 1
    ]
}
```

### 5.2 model_forward 输入示例

```python
# 经过process_batch和model_forward处理后:

# 输入1: 八叉树节点特征
data = torch.Tensor([2200, 4])  # [N_nodes, C_in]
# 每个节点4维特征（Normal + Depth）

# 输入2: Octree对象（已在CUDA上）
octree = <Octree对象>

# 输入3: 深度
depth = 5

# 输入4: 查询点
query_pts = torch.Tensor([2048, 4])  # [N_pts, 4]
# 前3列: xyz坐标
# 第4列: batch_id (0或1)

# 输入5: 刀具参数（已转为tensor）
tool_params = torch.Tensor([
    [-0.185, -0.564, -0.629, -0.535],
    [-0.188, -0.091, 0.622, -0.778]
])  # [2, 4]

# 模型输出
angles = model(data, octree, depth, query_pts, tool_params)
# angles: [2, 2] - [[pitch_0, roll_0], [pitch_1, roll_1]]
```

---

## 六、常见问题

### Q1: 为什么需要 CollateBatch？

**A**: PyTorch的DataLoader默认只能处理tensor和简单类型。但我们的数据包含：
- `Points` 对象（自定义类）
- `Octree` 对象（自定义类）
- 可变长度的点云

CollateBatch专门处理这些特殊类型，将它们正确合并成batch。

---

### Q2: tool_params 为什么一开始是字符串？

**A**: 从文件列表读取时是字符串，转换为tensor在 `model_forward` 中进行：

```python
tool_params = self._to_cuda_float_tensor(batch['tool_params'])
```

这样设计是为了兼容不同数据类型（字符串、numpy、tensor都能处理）。

---

### Q3: merge_points 和 CollateBatch(merge_points=True) 的区别？

**A**:
- `merge_points=True`: CollateBatch会调用 `merge_points` 合并Points对象
- `merge_points=False`: Points对象保持为列表

**当前项目**: `merge_points=True`，所以batch['points']是合并的Points对象。

---

### Q4: inbox_mask 有什么用？

**A**: 标记哪些点在 `[-1, 1]³` 范围内：
- 八叉树只能表示这个范围
- 超出范围的点会被裁剪
- `inbox_mask` 记录哪些点被保留

---

### Q5: 为什么labels既是打分表又是类别ID？

**A**: 命名混淆导致：
- `label`: 类别ID（旧字段，通常为0）
- `labels`: 打分表（338维）

建议改名为 `category` 和 `score_table` 以避免混淆。

---

### Q6: 八叉树节点数 N_nodes 如何确定？

**A**: 取决于：
1. 点云密度
2. 八叉树深度（depth=5 → 最大32³分辨率）
3. 稀疏性（只有包含点的体素才创建节点）

**典型值**:
- 单个样本: 500-1500个节点
- batch_size=8: 4000-12000个节点

---

## 七、调试技巧

### 7.1 查看batch内容

```python
# 在 SegSolver.train_step 中添加
def train_step(self, batch):
    print("=" * 50)
    print("Batch keys:", batch.keys())
    print("Points shape:", batch['points'].points.shape)
    print("Octree nodes:", batch['octree'].nnum)
    print("Labels shape:", batch['labels'].shape)
    print("Tool params:", batch['tool_params'])
    print("=" * 50)
    
    # ... 继续训练 ...
```

### 7.2 可视化Points对象

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points = batch['points']
xyz = points.points.cpu().numpy()  # [N, 3]
batch_id = points.batch_id.cpu().numpy()  # [N, 1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=batch_id, s=1)
plt.savefig('batch_points.png')
```

### 7.3 检查数据增强效果

```python
# 在 Transform.transform 中添加
def transform(self, sample, idx):
    points_before = sample['points'].points.clone()
    
    # ... 数据增强 ...
    
    points_after = sample['points'].points
    print(f"Before: min={points_before.min()}, max={points_before.max()}")
    print(f"After:  min={points_after.min()}, max={points_after.max()}")
    
    return sample
```

---

## 八、性能优化建议

### 8.1 加速数据加载

```yaml
DATA:
  train:
    num_workers: 4        # 多进程加载（当前为0）
    pin_memory: True      # 启用内存锁定
    batch_size: 16        # 增大batch_size（如果显存允许）
```

### 8.2 预加载到内存

```python
# 数据集较小时
dataset = Dataset(..., in_memory=True)
```

### 8.3 减少数据增强计算

```yaml
DATA:
  test:
    distort: False  # 测试时关闭增强
```

---

## 九、总结

### Batch 加工打包的完整链路

```
1. 文件读取 (ReadPly)
   └─> 读取.ply文件 + result文件夹的打分表
   
2. 单样本加载 (Dataset.__getitem__)
   └─> 组装单个样本字典
   
3. 数据增强 (Transform)
   ├─> preprocess: numpy → Points对象
   ├─> transform: 旋转/缩放/抖动/裁剪
   └─> points2octree: 构建Octree
   
4. 批量打包 (CollateBatch)
   ├─> 收集所有字段到列表
   ├─> merge_octrees: 合并八叉树
   ├─> merge_points: 合并点云
   └─> 转换labels为tensor
   
5. GPU迁移 (process_batch)
   └─> 将octree和points移到CUDA
   
6. 模型输入准备 (model_forward)
   ├─> 提取八叉树特征
   ├─> 构造查询点
   └─> 转换刀具参数为tensor
```

### 关键要点

1. **CollateBatch 是核心**: 在DataLoader的 `collate_fn` 参数中指定
2. **merge_octrees 是关键操作**: 将多个八叉树合并成超级八叉树
3. **batch_id 贯穿始终**: 标识每个数据点属于哪个样本
4. **工具参数延迟转换**: 在 model_forward 时才转为tensor
5. **打分表来自外部**: 从 `/home/.../result` 文件夹读取并归一化

---

**文档版本**: v1.0  
**更新日期**: 2025-11-06  
**维护者**: xinguanze



