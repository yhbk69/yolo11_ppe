"""
YOLOv11 模型导出为 TensorRT Engine 格式
用于 C++ 端高性能部署
使用方法: python export_engine.py
"""
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # 路径配置
    weight_path = r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt"
    onnx_path = r"D:\dltt\Python\YOLOV11\yolo11_construction.onnx"
    engine_path = r"D:\dltt\Python\YOLOV11\yolo11_construction.engine"

    print("=" * 60)
    print("YOLOv11 模型导出为 TensorRT Engine 格式")
    print("=" * 60)
    print(f"源权重: {weight_path}")
    print(f"ONNX路径: {onnx_path}")
    print(f"Engine路径: {engine_path}")
    print("=" * 60)

    # 加载模型
    print("\n[1/3] 正在加载模型...")
    model = YOLO(weight_path)
    print("模型加载成功!")

    # 导出为 ONNX
    print("\n[2/3] 正在导出为 ONNX 格式...")
    if not Path(onnx_path).exists():
        success = model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
            verbose=True
        )
        print(f"ONNX 导出成功: {success}")
    else:
        print(f"ONNX 文件已存在: {onnx_path}")

    # 使用 TensorRT Python API 构建 Engine
    print("\n[3/3] 正在构建 TensorRT Engine...")
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("错误: 需要安装 TensorRT 和 pycuda")
        print("请运行: pip install tensorrt pycuda")
        sys.exit(1)

    # TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # 设置工作空间大小（根据显卡显存调整）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB

    # 解析 ONNX
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"ONNX Parser Error: {parser.get_error(error)}")
            sys.exit(1)

    print("ONNX 模型解析成功!")

    # 构建 Engine
    print("正在构建 Engine（这可能需要几分钟）...")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("错误: Engine 构建失败")
        sys.exit(1)

    # 保存 Engine
    with open(engine_path, 'wb') as f:
        f.write(engine)

    print(f"\nEngine 构建成功: {engine_path}")

    # 显示文件大小
    onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    engine_size = Path(engine_path).stat().st_size / (1024 * 1024)
    print(f"ONNX 大小: {onnx_size:.2f} MB")
    print(f"Engine 大小: {engine_size:.2f} MB")
    print(f"压缩比: {onnx_size/engine_size:.2f}x")

    print("\n" + "=" * 60)
    print("导出完成!")
    print("=" * 60)
    print("\n接下来进行 C++ 部署:")
    print("1. 将 .engine 文件拷贝到 C++ 项目的 models 目录")
    print("2. 参考 deploy_cpp/yolo11_engine.cpp 编写推理代码")
    print("3. 编译时链接 TensorRT 和 CUDA 库")


if __name__ == "__main__":
    main()
