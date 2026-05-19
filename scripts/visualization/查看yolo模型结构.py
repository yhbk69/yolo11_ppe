import torch
# 2. 查看YOLO模型文件的键（Keys）
yolo_model = torch.load(r'E:\code\PycharmProjects\YOLOV11\runs\detect\train47\weights\best.pt', map_location='cpu')
# YOLOv8的.pt文件通常包含模型权重、模型结构等，是一个字典
print("\nYOLO checkpoint keys:")
if isinstance(yolo_model, dict):
    for key in yolo_model.keys():
        print(f"  - {key}")