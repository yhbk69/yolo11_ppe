import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='YOLOv11目标检测脚本')
    parser.add_argument('--source', type=str, required=True,
                        help='检测路径（图片/视频/目录）')
    # 添加置信度参数，设置默认值为0.25[6](@ref)
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值（默认：0.25）')
    args = parser.parse_args()

    # 验证源路径是否存在
    if not os.path.exists(args.source):
        print(f"错误：源路径 '{args.source}' 不存在！")
        sys.exit(1)

    # 验证置信度参数是否在有效范围内[1](@ref)
    if not 0.0 <= args.conf <= 1.0:
        print(f"错误：置信度阈值必须在0.0和1.0之间，当前值为 {args.conf}！")
        sys.exit(1)

    try:
        # 加载预训练模型
        model = YOLO(r"E:\code\PycharmProjects\YOLOV11\runs\detect\train47\weights\best.pt")
        print("✅ 模型加载成功，开始目标检测...")
        print(f"使用置信度阈值: {args.conf}")

        # 执行目标检测，添加conf参数[1,4](@ref)
        results = model(args.source, stream=True, conf=args.conf)  # stream=True 用于大文件处理

        # 创建结果保存目录
        source_path = Path(args.source)
        result_dir = source_path.parent / "result"
        result_dir.mkdir(exist_ok=True, parents=True)
        print(f"结果将保存到: {result_dir}")

        # 准备数据收集结构
        all_defects = []
        class_names = model.names  # 获取类别名称映射
        total_files = len(list(Path(args.source).glob('*'))) if os.path.isdir(args.source) else 1
        processed = 0

        for result in results:
            # 获取原始文件名并生成新文件名
            orig_path = Path(result.path)
            result_filename = f"{orig_path.stem}_result{orig_path.suffix}"
            save_path = result_dir / result_filename

            # 保存检测结果图像
            result.save(filename=str(save_path))

            # 收集检测到的缺陷信息
            image_defects = []
            for box in result.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                image_defects.append({
                    "image_path": str(orig_path),
                    "class_id": cls_idx,
                    "class_name": class_names[cls_idx],
                    "confidence": conf,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2
                })

            # 添加当前图像的缺陷信息
            all_defects.extend(image_defects)
            print(f"已保存: {save_path.name} - 检测到 {len(image_defects)} 个缺陷")

            processed += 1
            print(f"进度: {processed}/{total_files} ({processed / total_files:.0%})")

        # 保存缺陷统计结果到CSV
        if all_defects:
            df = pd.DataFrame(all_defects)
            csv_path = result_dir / "defects_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✅ 缺陷统计已保存到: {csv_path}")
            print(f"总计检测到 {len(df)} 个缺陷，涉及 {df['class_name'].nunique()} 种缺陷类型")

            # 添加分类统计摘要
            class_summary = df.groupby('class_name').size().reset_index(name='count')
            class_summary = class_summary.sort_values('count', ascending=False)
            print("\n缺陷分类统计:")
            print(class_summary.to_string(index=False))
        else:
            print("\n⚠️ 未检测到任何缺陷")

        print(f"\n处理完成！共处理 {processed} 个文件")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()