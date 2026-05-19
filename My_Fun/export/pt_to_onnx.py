from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt")

# 导出为ONNX格式
# 参数说明:
# format="onnx"     - 导出格式
# simplify=True    - 简化ONNX模型
# opset=13          - ONNX算子集版本
# imgsz=640         - 输入图像尺寸
model.export(format="onnx", simplify=True, opset=13, imgsz=640)

print("ONNX导出完成!")