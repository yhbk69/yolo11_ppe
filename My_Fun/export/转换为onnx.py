from ultralytics import YOLO
model = YOLO(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train47\weights\best.pt")
# model=YOLO(r"E:\code\PycharmProjects\DenoisingDiffusionProbabilityModel-ddpm--main\Checkpoints\ckpt_199_.pt")


model.export(format="onnx",simplify=True,opset=13)

# 导出 ONNX 时添加 dtype 强制转换
# model = YOLO("yolov11s.pt")
# model.export(format="onnx", imgsz=640, dynamic=True, int32=True)  # 关键：int32=True

# import onnx
# from onnx import version_converter
#
# model = onnx.load(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train30\weights\best.onnx")
# # 转换INT64到INT32
# converted_model = version_converter.convert_version(model, 13)
# onnx.save(converted_model, "best_fixed.onnx")