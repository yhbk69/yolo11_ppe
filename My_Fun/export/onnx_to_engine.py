import tensorrt as trt
import os


def onnx_to_engine(onnx_path, engine_path,
                   max_batch_size=1,
                   max_workspace_size=4 * (1 << 30),  # 4GB显存
                   fp16_mode=True,
                   dynamic_shapes=None):
    """
    参数说明：
    - onnx_path: ONNX模型路径
    - engine_path: 输出的引擎文件路径
    - max_batch_size: 最大批处理大小
    - max_workspace_size: 显存工作空间（单位：字节）
    - fp16_mode: 启用FP16精度加速
    - dynamic_shapes: 动态输入配置，格式: {"input_name": (min_shape, opt_shape, max_shape)}
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # 设置显存和工作空间 [1,2](@ref)
    config.max_workspace_size = max_workspace_size
    if fp16_mode and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 解析ONNX模型 [2,7](@ref)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX文件 {onnx_path} 不存在")

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("ONNX解析失败")

    # 动态输入配置 [1,4](@ref)
    if dynamic_shapes:
        profile = builder.create_optimization_profile()
        for name, shapes in dynamic_shapes.items():
            min_shape, opt_shape, max_shape = shapes
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    # 构建并保存引擎 [3,6](@ref)
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"引擎已保存至 {engine_path}")
    return serialized_engine


# 示例调用
if __name__ == "__main__":
    onnx_to_engine(
        onnx_path=r"E:\code\PycharmProjects\YOLOV11\runs\detect\train30\weights\best.onnx",
        engine_path=r"E:\code\PycharmProjects\YOLOV11\runs\detect\train30\weights\best.engine",
        max_batch_size=4,
        fp16_mode=True,
        dynamic_shapes={
            "input": ((1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224))
        }
    )