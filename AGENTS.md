# YOLOV11 项目指南

## 项目简介

基于 Ultralytics YOLO11 的计算机视觉目标检测项目，支持训练、推理、模型导出（ONNX/TensorRT）、摄像头实时检测、热力图分析等完整流程。可用于安全帽检测、缺陷检测等自定义场景。

## 目录结构

```
YOLOV11/
── scripts/                # Python 脚本（按功能分类）
│   ├── train/             # 训练脚本（原始方案：直接检测 no_xx）
│   │   ├── train.py       # 基础训练
│   │   ├── train2.py      # 进阶训练（含完整增强参数）
│   │   └── train3.py      # 批量训练（依次训练 n/s/m/l/x）
│   ├── train_v2/          # 训练脚本（两阶段方案 ★推荐）
│   │   └── train_v2.py    # 两阶段训练：只检测 Person + PPE 正类
│   ├── inference/         # 推理检测脚本
│   │   ├── predict.py     # 单张图片推理
│   │   ├── predict2.py    # 批量推理（支持目录，输出CSV统计）
│   │   ├── predict3.py    # 推理（支持supervision高级标注）
│   │   ├── detect_cam.py  # 摄像头实时检测
│   │   ├── detect_folder.py # 文件夹批量检测 + 耗时统计
│   │   ├── 1.video.py     # 视频文件检测（含FPS统计）
│   │   └── predict_v2.py  # 两阶段推理：检测 + 逻辑判断 no_xx ★
│   ├── export/            # 模型导出脚本
│   │   ├── export_onnx.py   # 导出 ONNX 格式
│   │   └── export_engine.py # 导出 TensorRT Engine 格式
│   ├── visualization/     # 可视化脚本
│   │   ├── 1.heatmap.py     # Grad-CAM 热力图生成
│   │   ├── main.py          # 查看模型结构/参数信息
│   │   └── 查看yolo模型结构.py # 打印 checkpoint 键信息
│   └── tools/             # 工具脚本
│       ├── xml_to_txt.py  # XML标注 → YOLO TXT 格式转换
│       ├── xml2label.py   # XML → YOLO TXT 格式转换（My_Fun 版本）
│       ├── convert_dataset_v2.py # 数据集转换 v1→v2
│       ├── analyze_person_ppe.py # Person与PPE关系分析
│       └── count_classes.py # 类别统计
├── weights/               # 模型权重
│   ├── yolo11/           # YOLO11 预训练权重（n/s/m/l/x）
│   └── yolov8/           # YOLOv8 预训练权重
├── datasets/              # 数据集目录
│   ├── construction-ppe/  # 原始数据集（11类，含no_xx）
│   └── construction-ppe-v2/ # 两阶段方案数据集（6类：Person+PPE正类）
├── My_Fun/                # 自定义工具脚本
│   ├── network/           # 网络可视化和分析
│   │   ├── 1.可视化网络.py  # 网络结构可视化
│   │   ├── 2.查看网络信息.py # 打印网络结构信息
│   │   ── 0.test.py      # 测试脚本
│   ├── resize/            # 图片缩放工具
│   │   ├── 3.resize640.py # 缩放到 640 像素
│   │   ├── 4.resize2.py   # 缩放工具 2
│   │   ├── 5.resize3.py   # 缩放工具 3
│   │   └── 6.resize4.py   # 缩放工具 4
│   ├── split/             # 数据集划分工具
│   │   ├── 划分2.py       # 数据集划分（训练/验证/测试）
│   │   ├── 划分3.py       # 数据集划分工具 3
│   │   └── 划分测试验证集.py # 划分测试集和验证集
│   ├── export/            # 模型导出工具
│   │   ├── pt_to_onnx.py  # PT → ONNX 转换
│   │   ├── 转换为onnx.py   # ONNX 转换（简化版）
│   │   ├── 低精度onnx.py   # FP16/INT8 量化导出
│   │   ├── onnx_to_engine.py # ONNX → TensorRT Engine 转换
│   │   └── Conv.onnx      # 示例 ONNX 模型
│   ├── dataAug/           # 数据增强脚本
│   └── 统计数据集.py      # 数据集统计分析
├── runs/                  # 训练/推理结果输出
│   └── detect/            # 检测结果（train*/predict）
├── deploy_cpp/            # C++ 部署代码（用户自行添加）
── examples/              # 官方示例（ONNXRuntime/OpenCV/SAHI等）
├── assets/                # 静态资源（图片、视频等）
├── docs/                  # 文档
│   └── ppe_detection_plan.md # 两阶段 PPE 检测方案说明
├── ultralytics/           # Ultralytics YOLO 核心库源码
│   ├── nn/               # 网络模块（conv/block/head/transformer）
│   ├── utils/            # 工具函数（loss/metrics/ops/plots）
│   ├── models/           # 模型配置 YAML
│   ├── solutions/        # 高级解决方案（计数/测距/热力图/队列管理）
│   ├── trackers/         # 目标跟踪（ByteTrack/BOT-SORT）
│   └── cfg/              # 默认配置
├── pyproject.toml        # 项目依赖配置
├── README.md             # 项目说明（英文）
└── README.zh-CN.md       # 项目说明（中文）
```

## 环境安装

```bash
pip install ultralytics
# 或从源码安装
pip install -e .
```

## 运行方式

### 训练模型

#### 原始方案（直接检测 no_xx，效果较差）
```bash
# 基础训练
python scripts/train/train.py

# 进阶训练（含完整参数配置）
python scripts/train/train2.py

# 批量训练（依次训练 n/s/m/l/x）
python scripts/train/train3.py
```

#### 两阶段方案（★推荐，逻辑推理 no_xx）
```bash
# 两阶段训练：只检测 Person + PPE 正类
python scripts/train_v2/train_v2.py
```

### 推理检测

#### 原始方案
```bash
# 单张图片
python scripts/inference/predict.py

# 批量推理 + CSV统计
python scripts/inference/predict2.py --source <图片/视频/目录路径> --conf 0.25

# 文件夹批量检测 + 耗时统计
python scripts/inference/detect_folder.py

# 摄像头实时检测
python scripts/inference/detect_cam.py

# 视频文件检测
python scripts/inference/1.video.py
```

#### 两阶段方案（★推荐）
```bash
# 两阶段推理：检测 Person + PPE → 逻辑判断 no_xx
python scripts/inference/predict_v2.py --model runs/detect/train_v2_yolo11n/weights/best.pt --source <图片/目录>
```

### 模型导出
```bash
# 导出 ONNX
python scripts/export/export_onnx.py

# 导出 TensorRT Engine
python scripts/export/export_engine.py
```

### 数据分析
```bash
# Grad-CAM 热力图可视化
python scripts/visualization/1.heatmap.py

# 查看模型结构
python scripts/visualization/main.py
python scripts/visualization/查看yolo模型结构.py
```

### 数据预处理
```bash
# XML 转 YOLO 格式
python scripts/tools/xml_to_txt.py
python scripts/tools/xml2label.py

# 数据集转换 v1→v2
python scripts/tools/convert_dataset_v2.py

# 图片缩放
python My_Fun/resize/3.resize640.py

# 数据集划分
python My_Fun/split/划分2.py

# 统计数据集
python My_Fun/统计数据集.py
python scripts/tools/count_classes.py
```

## 两阶段检测方案说明

### 问题背景
原始方案直接标注 `no_helmet`/`no_gloves` 等负类，导致模型召回率为 0（全部预测为背景）。
原因：no_xx 标注区域与 Person 重叠，模型无法区分"有头盔"和"无头盔"的同一区域。

### 解决方案
```
阶段1: YOLO 检测 → Person + PPE正类（helmet/gloves/vest/boots/goggles）
阶段2: 逻辑判断 → 对每个 Person，检查其框内是否有对应PPE → 缺失则生成 no_xx
```

### 优势
1. **模型简单**：只检测"存在的物体"，符合 YOLO 设计
2. **逻辑清晰**：no_xx 是推理结果，不是检测目标
3. **可解释**：可以明确知道为什么判定为 no_xx
4. **效果更好**：避免了模型混淆正负类的问题

### 数据集对比
| 方案 | 类别数 | 类别 |
|------|--------|------|
| 原始（v1） | 11 | helmet, gloves, vest, boots, goggles, none, Person, no_helmet, no_goggle, no_gloves, no_boots |
| 两阶段（v2） | 6 | helmet, gloves, vest, boots, goggles, Person |

## 注意事项

- 训练前需将数据集路径配置为绝对路径（`scripts/train_v2/train_v2.py` 中的 `data` 参数）
- 推理脚本中的模型权重路径需根据实际位置修改（如 `weights/yolo11/yolo11n.pt`）
- Windows 下训练需 `freeze_support()`（已在脚本中处理）
- C++ 部署参考 `deploy_cpp/` 和 `scripts/export/export_engine.py`
- 模型权重统一存放在 `weights/` 目录下
- 原始数据集 `construction-ppe/` 已保留，新增 `construction-ppe-v2/` 用于两阶段方案
