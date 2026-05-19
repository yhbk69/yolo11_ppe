# 两阶段 PPE 检测方案

## 问题分析

当前数据集直接标注 `no_helmet`/`no_gloves` 等负类，导致：
- 模型召回率 = 0（全部预测为背景）
- 原因：no_xx 标注区域与 Person 重叠，模型无法区分"有头盔"和"无头盔"的同一区域

## 新方案：两阶段检测 + 逻辑推理

```
阶段1: YOLO 检测 → Person + PPE正类（helmet/gloves/vest/boots/goggles）
阶段2: 逻辑判断 → 对每个 Person，检查其框内是否有对应PPE → 缺失则生成 no_xx
```

## 数据关系（已验证）

| 关系 | 统计 |
|------|------|
| PPE 框中心在 Person 框内 | 720 次 |
| no_xx 框中心在 Person 框内 | 607 次 |
| 结论 | **PPE/no_xx 与 Person 存在明确的空间归属关系** |

## 实施步骤

### Step 1: 创建新数据集配置
- 新建 `datasets/construction-ppe-v2/data.yaml`
- 只保留 7 个类：`Person, helmet, gloves, vest, boots, goggles, none`
- 删除 `no_helmet, no_goggle, no_gloves, no_boots` 的标注
- `none` 保留（表示人身体区域，辅助 Person 检测）

### Step 2: 转换标注文件
- 新建 `scripts/tools/convert_dataset_v2.py`
- 遍历所有标注 txt，删除第 7/8/9/10 类（no_xx）的行
- 输出到新目录 `datasets/construction-ppe-v2/`

### Step 3: 训练模型
- 新建 `scripts/train/train_v2.py`
- 使用 `construction-ppe-v2/data.yaml` 训练
- 只检测 Person + PPE 正类

### Step 4: 推理 + 逻辑判断
- 新建 `scripts/inference/predict_v2.py`
- 加载 Step 3 训练的模型
- 对每张图：
  1. 检测所有 Person 和 PPE
  2. 对每个 Person 框，检查内部是否有 helmet/gloves/vest/boots/goggles
  3. 如果缺失 → 在 Person 头部/对应位置绘制 no_xx 框和标签

### Step 5: 评估
- 对比原方案（直接检测 no_xx）与新方案（逻辑推理）的效果
- 使用原始测试集评估

## 文件结构

```
scripts/
├── train/
│   ── train_v2.py              # 新训练脚本
├── inference/
│   └── predict_v2.py            # 两阶段推理脚本
└── tools/
    ├── convert_dataset_v2.py    # 数据集转换脚本
    └── analyze_person_ppe.py    # 分析脚本（已完成）

datasets/
└── construction-ppe-v2/         # 新数据集目录
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

## 优势

1. **模型简单**：只检测"存在的物体"，符合 YOLO 设计
2. **逻辑清晰**：no_xx 是推理结果，不是检测目标
3. **可解释**：可以明确知道为什么判定为 no_xx（缺少某个 PPE）
4. **可扩展**：新增 PPE 类型只需修改逻辑判断，无需重新标注

## 需要确认

1. 是否删除 `none` 类？（它是人身体区域，可能和 Person 重叠）
2. 是否保留原始 `construction-ppe/` 目录？（不删除，仅新建 v2）
3. 训练参数是否沿用 `train3.py` 的配置？
