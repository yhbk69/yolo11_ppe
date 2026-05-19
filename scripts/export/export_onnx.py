"""
YOLOv11 模型导出为 ONNX 格式
用于 C++ 端部署
使用方法: python export_onnx.py
"""
from ultralytics import YOLO


def main():
    # 权重路径
    weight_path = r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt"
    # 导出路径（不写后缀，自动添加 .onnx）
    export_path = r"D:\dltt\Python\YOLOV11\yolo11_construction.onnx"

    print("=" * 60)
    print("YOLOv11 模型导出为 ONNX 格式")
    print("=" * 60)
    print(f"源权重: {weight_path}")
    print(f"导出路径: {export_path}")
    print("=" * 60)

    # 加载模型
    print("\n正在加载模型...")
    model = YOLO(weight_path)
    print("模型加载成功!")

    # 导出为 ONNX
    print("\n正在导出为 ONNX 格式...")
    # imgsz: 输入图片尺寸
    # opset: ONNX opset 版本（11/12/13/14/15/16/17）
    # simplify: 简化 ONNX 模型（需要 onnxslim）
    # dynamic: 动态输入尺寸（batch, height, width）
    # half: FP16 半精度导出
    success = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
        verbose=True
    )

    print(f"\n导出成功: {success}")
    print(f"导出的 ONNX 文件: {export_path}")

    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print("\nC++ 部署说明:")
    print("1. 使用 OpenCV DNN 模块加载 ONNX 模型")
    print("2. 或使用 ONNX Runtime C++ API")
    print("3. 参考 deploy_cpp 目录下的 C++ 代码")


if __name__ == "__main__":
    main()
