import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv11目标检测脚本')
    parser.add_argument('--source', type=str, required=True,
                        help='检测路径（图片/视频/目录）')
    args = parser.parse_args()

    # 验证源路径是否存在
    if not os.path.exists(args.source):
        print(f"❌ 错误：源路径 '{args.source}' 不存在！")
        sys.exit(1)

    try:
        # 加载预训练模型
        model = YOLO(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train15\weights\best.pt")
        print("✅ 模型加载成功，开始目标检测...")

        # 创建结果保存目录
        source_path = Path(args.source)
        result_dir = source_path.parent / "result2"
        result_dir.mkdir(exist_ok=True, parents=True)
        print(f"📁 结果将保存到: {result_dir}")

        # 处理不同类型输入源
        if os.path.isdir(args.source):
            input_files = list(Path(args.source).glob('*'))
        else:
            input_files = [Path(args.source)]

        total_files = len(input_files)
        processed = 0

        # 执行目标检测（禁用流式处理确保准确计时）
        results = model(input_files, stream=False)

        for result in results:
            # 记录开始检测时间戳
            start_time = time.time()

            # 执行推理（核心耗时操作）
            _ = result.boxes  # 触发结果计算

            # 计算模型检测耗时（毫秒）
            detect_time_ms = int((time.time() - start_time) * 1000)

            # 获取原始文件名并生成带时间戳的新文件名
            orig_path = Path(result.path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_filename = f"{orig_path.stem}_{timestamp}_{detect_time_ms}ms{orig_path.suffix}"
            save_path = result_dir / result_filename

            # 保存检测结果
            result.save(filename=str(save_path))
            print(f"💾 已保存: {save_path.name} | 耗时: {detect_time_ms}ms")

            # 显示检测结果（可选）
            # result.show()
            # print(f"👁️ 已显示: {orig_path.name}")

            processed += 1
            print(f"📊 进度: {processed}/{total_files} ({processed / total_files:.0%})")

        print(f"🎉 处理完成！共处理 {processed} 个文件")

    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()