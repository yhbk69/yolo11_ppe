import torch
from ultralytics import YOLO

print(torch.__version__)

# 1. 直接加载 pt 文件查看原始字典结构
checkpoint = torch.load('weights/yolo11/yolo11n.pt', map_location='cpu')

print("=== 文件包含的键 ===")
print(checkpoint.keys())
# 输出示例: dict_keys(['model', 'ema', 'epoch', 'train_args', 'optimizer', ...])

print("\n=== 训练参数示例 ===")
print(checkpoint['train_args'])
# 可以看到 hsv_h, degrees, translate 等具体数值

print("\n=== 模型类别信息 ===")
# 通过 Ultralytics 的 YOLO 类加载更方便查看模型属性
model = YOLO('weights/yolo11/yolo11n.pt')
print(f"类别数量: {model.model.nc}")
print(f"类别名称: {model.names}")