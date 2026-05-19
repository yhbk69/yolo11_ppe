import onnx
from onnxsim import simplify
import argparse
import os


def simplify_onnx_model(input_path: str, output_path: str) -> None:
    """
    简化ONNX模型并保存优化结果

    参数:
        input_path: 输入ONNX模型路径
        output_path: 输出简化后模型路径
    """
    # 验证输入文件存在性
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 输入模型不存在: {input_path}")

    # 加载原始模型
    original_model = onnx.load(input_path)
    print(f"✅ 已加载原始模型: {input_path} (大小: {os.path.getsize(input_path) / 1024:.1f} KB)")

    # 执行模型简化
    simplified_model, optimization_success = simplify(
        original_model,
        perform_optimization=True,  # 启用图优化
        skip_fuse_bn=False,  # 执行BN层融合
        dynamic_input_shape=True  # 保持动态输入能力
    )

    # 验证优化结果
    if not optimization_success:
        raise RuntimeError("❌ 模型简化失败 - 可能包含不支持的算子或结构问题")

    # 保存优化后模型
    onnx.save(simplified_model, output_path)

    # 计算压缩率
    orig_size = os.path.getsize(input_path)
    simp_size = os.path.getsize(output_path)
    compression_ratio = (orig_size - simp_size) / orig_size * 100

    print(f"💾 简化模型已保存: {output_path}")
    print(f"📊 大小优化: {orig_size / 1024:.1f} KB → {simp_size / 1024:.1f} KB (-{compression_ratio:.1f}%)")


if __name__ == "__main__":
    # 命令行参数配置
    parser = argparse.ArgumentParser(description='ONNX模型简化工具')
    parser.add_argument('--input', default='best.onnx',
                        help='输入ONNX模型路径')
    parser.add_argument('--output', default='best.onnx',
                        help='输出简化模型路径')
    args = parser.parse_args()

    try:
        simplify_onnx_model(args.input, args.output)
        print("✨ 模型简化完成！")
    except Exception as e:
        print(f"🔥 错误发生: {str(e)}")
        print("💡 建议解决方案:")
        print("- 检查模型是否包含自定义算子")
        print("- 尝试固定输入尺寸: 添加参数 --dynamic-input-shape=False")
        print("- 升级onnxsim库: pip install --upgrade onnxsim")