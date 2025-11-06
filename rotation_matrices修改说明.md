# rotation_matrices 添加到 Batch 的修改说明

## 一、修改目标

将预定义的 338 个旋转矩阵 `rotation_matrices` 通过 batch 传递到训练/测试流程中，而不是在 SegSolver 中单独加载。

---

## 二、已完成的修改

### 2.1 Dataset 中加载 rotation_matrices

**文件**: `projects/thsolver/dataset.py`

**修改位置 1**: Line 32（`__init__` 方法）
```python
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, filelist, transform, read_file=read_file,
                 in_memory=False, take: int = -1):
        # ... 其他初始化代码 ...
        self.rotation_matrices = self._load_rotation_matrices()  # ← 新增
```

**修改位置 2**: Line 43-69（新增方法）
```python
def _load_rotation_matrices(self):
    """加载JSON文件中的旋转矩阵到内存"""
    json_path = os.path.join(os.path.dirname(__file__), 'rotation_matrices.json')
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rotation_matrices = []
        for i in range(338):
            key = f"ori_{i:03d}"
            if key in data:
                matrix = np.array(data[key]['rotation_matrix'], dtype=np.float32)
                rotation_matrices.append(torch.from_numpy(matrix))
            else:
                rotation_matrices.append(torch.eye(3, dtype=torch.float32))
        
        rotation_matrices = torch.stack(rotation_matrices)  # (338, 3, 3)
        print(f"成功加载 {len(rotation_matrices)} 个旋转矩阵")
        return rotation_matrices
    except Exception as e:
        print(f"加载旋转矩阵失败: {e}")
        return torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(338, 1, 1)
```

**修改位置 3**: Line 77（`__getitem__` 方法）
```python
def __getitem__(self, idx):
    sample = self.read_file(os.path.join(self.root, self.filenames[idx]))
    output = self.transform(sample, idx)
    output['label'] = self.labels[idx]
    output['filename'] = self.filenames[idx]
    output['rotation_matrices'] = self.rotation_matrices  # ← 新增
    # ... 其他代码 ...
    return output
```

---

### 2.2 CollateBatch 中处理 rotation_matrices

**文件**: `projects/ocnn/dataset.py`

**修改位置**: Line 167-171（`__call__` 方法）

**原始代码**（错误）:
```python
if 'rotation_matrices' == key:
    outputs['rotation_matrices'] = outputs[key]
```

**修正后代码**（正确）:
```python
# rotation_matrices: 所有样本共享同一个 (338, 3, 3) tensor
if 'rotation_matrices' == key:
    # 只取第一个，因为所有样本的rotation_matrices都相同
    outputs['rotation_matrices'] = outputs[key][0]
```

**为什么要修正**:
- 原始代码会导致 `outputs['rotation_matrices']` 是一个列表：`[tensor(338,3,3), tensor(338,3,3), ...]` 长度为 `batch_size`
- 修正后 `outputs['rotation_matrices']` 是单个tensor：`tensor(338, 3, 3)`
- 因为所有样本共享同一个 rotation_matrices，没必要保留多份

---

## 三、修改验证

### 3.1 运行测试脚本

```bash
cd /home/xinguanze/project/ex_6_scorer/DM-scorer
python test_batch_rotation_matrices.py
```

**期望输出**:
```
============================================================
测试 Batch 中 rotation_matrices 的传递
============================================================

[1] 创建数据集...
   数据集大小: 10

[2] 测试单个样本...
   样本字段: dict_keys(['points', 'inbox_mask', 'octree', 'label', 'filename', 'rotation_matrices', 'labels', 'tool_params'])
   ✓ rotation_matrices 存在
   形状: torch.Size([338, 3, 3])
   类型: <class 'torch.Tensor'>
   数据类型: torch.float32
   设备: cpu
   ✓ 形状正确: (338, 3, 3)
   第一个矩阵验证:
     - 正交性 (R@R^T=I): True
     - 行列式 (det(R)≈1): 1.000000

[3] 测试 DataLoader batch...
   Batch 字段: dict_keys(['points', 'inbox_mask', 'octree', 'label', 'filename', 'rotation_matrices', 'labels', 'tool_params'])
   ✓ batch 中有 rotation_matrices
   形状: torch.Size([338, 3, 3])
   类型: <class 'torch.Tensor'>
   ✓ 形状正确: (338, 3, 3) - 所有样本共享

[4] 验证数据一致性...
   单样本 vs Batch: ✓ 相同

[5] 测试与 SegSolver.rotation_matrices 对比...
   Batch vs SegSolver: ✓ 相同

============================================================
✓ 测试通过！rotation_matrices 正确传递到 batch 中
============================================================
```

---

## 四、数据流变化

### 修改前
```
SegSolver.__init__
    └─> _load_rotation_matrices()
        └─> self.rotation_matrices (338, 3, 3)
            └─> 在 loss_function_pro/promax 等方法中使用
```

### 修改后
```
Dataset.__init__
    └─> _load_rotation_matrices()
        └─> self.rotation_matrices (338, 3, 3)
            ↓
Dataset.__getitem__
    └─> output['rotation_matrices'] = self.rotation_matrices
            ↓
CollateBatch.__call__
    └─> outputs['rotation_matrices'] = outputs['rotation_matrices'][0]
            ↓
batch['rotation_matrices'] (338, 3, 3)
    └─> 传递到训练/测试流程
```

---

## 五、当前状态与待优化项

### ✅ 已完成
1. Dataset 中加载 rotation_matrices
2. 单样本中添加 rotation_matrices
3. CollateBatch 中正确处理（只保留一份）
4. batch 中包含 rotation_matrices

### ⚠️ 待优化：数据源重复

**问题**: 目前有两个地方加载 rotation_matrices：
1. **Dataset** (`thsolver/dataset.py` Line 32)
2. **SegSolver** (`segmentation.py` Line 25) - 目前还保留

**影响**:
- 内存占用增加（两份相同的数据）
- 数据来源不统一
- batch 中的 rotation_matrices 目前**未被使用**

---

## 六、进一步优化方案

### 方案 A：使用 batch 中的版本（推荐）

**优点**: 数据流清晰，所有数据从 batch 来

**需要修改的地方**:

#### 1. 删除 SegSolver 中的加载
```python
# segmentation.py
class SegSolver(Solver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ❌ 删除这一行
        # self.rotation_matrices = self._load_rotation_matrices()
    
    # ❌ 删除或注释掉这个方法
    # def _load_rotation_matrices(self):
    #     ...
```

#### 2. 修改 loss 函数接受 rotation_matrices 参数
```python
# segmentation.py

def loss_function_pro(self, logit, label, rotation_matrices):
    """
    参数:
        logit: [B, 2] 预测角度
        label: [B, 338] 打分表
        rotation_matrices: [338, 3, 3] 预定义旋转矩阵（从batch传入）
    """
    R_pred = self._angles_to_rotation_matrix(logit)
    
    # 使用传入的 rotation_matrices，而不是 self.rotation_matrices
    if rotation_matrices.device != R_pred.device:
        rotation_matrices = rotation_matrices.to(R_pred.device)
    
    # 计算与所有预定义旋转矩阵的测地距离
    R_pred_expanded = R_pred.unsqueeze(1)
    rotation_matrices_expanded = rotation_matrices.unsqueeze(0)
    # ... 其余代码不变，使用 rotation_matrices 而非 self.rotation_matrices
```

同样修改：
- `loss_function_promax`
- `model_score`
- `model_score_pro`

#### 3. 修改 train_step / test_step
```python
def train_step(self, batch):
    batch = self.process_batch(batch, self.FLAGS.DATA.train)
    logit, label = self.model_forward(batch)
    
    # 从 batch 中提取 rotation_matrices
    rotation_matrices = batch['rotation_matrices']
    
    # 传递给 loss 函数
    loss = self.loss_function_pro(logit, label, rotation_matrices)
    
    # 同样传递给 metrics
    model_score = self.model_score(logit, label, rotation_matrices)
    model_score_pro = self.model_score_pro(logit, label, rotation_matrices)
    # ...
```

---

### 方案 B：保持 SegSolver 中的版本（最简单）

**优点**: 无需修改 loss 函数和 metrics

**需要修改的地方**:

#### 1. 删除 Dataset 中的代码
```python
# thsolver/dataset.py

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ...):
        # ❌ 删除这一行
        # self.rotation_matrices = self._load_rotation_matrices()
    
    # ❌ 删除这个方法
    # def _load_rotation_matrices(self):
    #     ...
    
    def __getitem__(self, idx):
        # ... 其他代码 ...
        # ❌ 删除这一行
        # output['rotation_matrices'] = self.rotation_matrices
        return output
```

#### 2. 删除 CollateBatch 中的代码
```python
# ocnn/dataset.py

class CollateBatch:
    def __call__(self, batch):
        # ... 其他代码 ...
        
        # ❌ 删除这几行
        # if 'rotation_matrices' == key:
        #     outputs['rotation_matrices'] = outputs[key][0]
        
        return outputs
```

---

## 七、推荐方案

### 🎯 推荐：方案 A（使用 batch 中的版本）

**理由**:
1. **架构更清晰**: 所有数据都从 DataLoader 来，Solver 只负责训练逻辑
2. **易于扩展**: 将来如果需要动态加载不同的 rotation_matrices（例如不同的任务），只需修改 Dataset
3. **内存效率**: 只在 Dataset 中加载一次，通过 batch 共享
4. **符合 PyTorch 最佳实践**: 数据相关的都在 Dataset/DataLoader，模型训练逻辑在 Trainer/Solver

**实施步骤**:
1. 删除 `segmentation.py` 中的 `_load_rotation_matrices` 方法调用
2. 修改所有使用 `self.rotation_matrices` 的方法，改为接受参数
3. 在 `train_step/test_step` 中从 batch 提取并传递
4. 运行测试确保功能正常

---

## 八、修改检查清单

### 当前状态检查

- [x] Dataset 加载 rotation_matrices
- [x] Dataset.__getitem__ 添加 rotation_matrices
- [x] CollateBatch 正确处理（只保留一份）
- [x] batch 中包含 rotation_matrices (338, 3, 3)
- [ ] SegSolver 使用 batch 中的 rotation_matrices
- [ ] 删除重复的加载代码

### 如果选择方案 A，需要修改的文件

- [ ] `segmentation.py`
  - [ ] 删除 Line 25: `self.rotation_matrices = self._load_rotation_matrices()`
  - [ ] 删除/注释 `_load_rotation_matrices` 方法 (Line 27-53)
  - [ ] 修改 `loss_function_pro` 添加参数
  - [ ] 修改 `loss_function_promax` 添加参数
  - [ ] 修改 `model_score` 添加参数
  - [ ] 修改 `model_score_pro` 添加参数
  - [ ] 修改 `train_step` 传递参数
  - [ ] 修改 `test_step` 传递参数

### 如果选择方案 B，需要修改的文件

- [ ] `thsolver/dataset.py`
  - [ ] 删除 Line 32: `self.rotation_matrices = ...`
  - [ ] 删除 `_load_rotation_matrices` 方法
  - [ ] 删除 Line 77: `output['rotation_matrices'] = ...`

- [ ] `ocnn/dataset.py`
  - [ ] 删除 Line 167-171 的 rotation_matrices 处理代码

---

## 九、测试建议

### 9.1 单元测试
```bash
# 测试 batch 传递
python test_batch_rotation_matrices.py
```

### 9.2 集成测试
```bash
# 运行一个 epoch 确保训练正常
cd projects
python run_seg_deepmill.py --gpu 0 --ratios 0.01
```

### 9.3 验证输出
```python
# 在 train_step 中添加临时打印
def train_step(self, batch):
    if 'rotation_matrices' in batch:
        print(f"✓ rotation_matrices shape: {batch['rotation_matrices'].shape}")
    else:
        print("✗ rotation_matrices 不在 batch 中!")
    # ... 其他代码
```

---

## 十、常见问题

### Q1: 为什么 CollateBatch 要取 `[0]`？

**A**: 因为在 `Dataset.__getitem__` 中，每个样本都添加了相同的 `self.rotation_matrices`。如果有 batch_size=8，那么 `outputs['rotation_matrices']` 就是一个长度为8的列表，但每个元素都完全相同。取 `[0]` 就是只保留一份。

### Q2: 为什么不在每个样本中复制一份？

**A**: 
- **内存效率**: rotation_matrices 是 (338, 3, 3) = 3042 个float32，约 12KB。batch_size=8 就是 96KB，虽然不大但完全没必要
- **语义清晰**: rotation_matrices 是全局的、不变的参考数据，不应该属于单个样本

### Q3: 如果将来需要不同的 rotation_matrices 怎么办？

**A**: 
1. 在 Dataset 中根据样本类别加载不同的 JSON 文件
2. 在 `__getitem__` 中根据 idx 选择对应的 rotation_matrices
3. CollateBatch 中改为保留列表（如果每个样本不同）

### Q4: 为什么 SegSolver 中也加载了一份？

**A**: 这是历史遗留。在您添加 batch 传递之前，rotation_matrices 是在 Solver 中加载的。现在有了 batch 版本，建议统一到一个地方。

---

## 十一、性能对比

### 内存占用

| 方案 | Dataset | SegSolver | 总计 |
|------|---------|-----------|------|
| **修改前** | - | 12KB | 12KB |
| **当前状态** | 12KB | 12KB | 24KB |
| **方案A** | 12KB | - | 12KB |
| **方案B** | - | 12KB | 12KB |

### 加载时间

| 方案 | 加载次数 | 总时间 |
|------|----------|--------|
| **修改前** | 1次 (Solver初始化) | ~10ms |
| **当前状态** | 2次 (Dataset + Solver) | ~20ms |
| **方案A** | 1次 (Dataset初始化) | ~10ms |
| **方案B** | 1次 (Solver初始化) | ~10ms |

**结论**: 方案A和方案B性能相当，都比当前状态好（避免重复加载）

---

## 十二、总结

### 您的修改评价

**✅ 正确的部分**:
1. Dataset 中加载 rotation_matrices 的逻辑完全正确
2. 在 `__getitem__` 中添加到输出是正确的做法
3. CollateBatch 的处理思路正确（经过我的修正后）

**⚠️ 需要改进的部分**:
1. CollateBatch 中应该取 `[0]` 而不是保留整个列表（已修正）
2. 存在数据源重复（Dataset 和 SegSolver 都加载）
3. batch 中的 rotation_matrices 目前未被使用

### 下一步建议

1. **立即执行**: 运行 `test_batch_rotation_matrices.py` 确认修改正确
2. **短期优化**: 选择方案A或B，消除重复加载
3. **长期维护**: 保持数据流清晰，建议采用方案A

---

**文档版本**: v1.0  
**创建日期**: 2025-11-06  
**修改记录**:
- 2025-11-06: 初始版本，修正 CollateBatch 中的逻辑

