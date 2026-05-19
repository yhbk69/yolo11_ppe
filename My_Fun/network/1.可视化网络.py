import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 加载模型
model = YOLO(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train30\weights\best.pt")
feature_maps = []

# 钩子函数
def hook_fn(module, input, output):
    feature_maps.append(output)

# 注册钩子到卷积层
hooks = []
for layer in model.model.modules():
    if isinstance(layer, torch.nn.Conv2d):
        hooks.append(layer.register_forward_hook(hook_fn))

# 输入图像推理
results = model(r"E:\code\dataset\seat2\images_640\stain_22(admin).jpg")

# 移除钩子
for hook in hooks:
    hook.remove()

# 可视化前5层特征图
for i, fmap in enumerate(feature_maps[:5]):
    fmap = fmap[0].detach().cpu()  # 取batch中第一个样本
    channels = min(fmap.shape[0], 4)  # 最多显示4通道
    fig, axs = plt.subplots(1, channels, figsize=(15, 5))
    for ch in range(channels):
        axs[ch].imshow(fmap[ch], cmap='viridis')
        axs[ch].axis('off')
    plt.show()