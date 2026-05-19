from ultralytics import YOLO

# 如果你想加载一个预先训练好的模型，比如 best.pt 文件
model = YOLO(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train30\weights\best.pt")
print(model.info())
print("="*30)
print(model.info(detailed=True))

