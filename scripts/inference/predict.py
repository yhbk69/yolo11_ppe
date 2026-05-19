"""
YOLOv11 推理演示
使用方法: python predict.py [source]
"""
from datetime import datetime
import os
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # 模型路径 (使用 yolo11n.pt)
    # model_path = r"F:\ddtt\Python\YOLOV11\yolo11n.pt"
    model_path = Path(r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt")

    # 图片源 - 可以是单个图片路径或文件夹路径
    # 单个图片示例
    # source = r"D:\dltt\Python\YOLOV11\construction-ppe\all_images\image12.jpg"

    # 文件夹示例 (检测文件夹内所有图片)
    source = r"D:\dltt\Python\YOLOV11\construction-ppe\all_images"

    print("=" * 50)
    print("YOLOv11 目标检测推理演示")
    print("=" * 50)
    print(f"模型: {model_path}")
    print(f"输入源: {source}")

    # 判断输入类型
    if os.path.isfile(source):
        print(f"类型: 单个图片文件")
    elif os.path.isdir(source):
        print(f"类型: 图片文件夹")
        # 统计文件夹中的图片数量
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in os.listdir(source) if Path(f).suffix.lower() in image_extensions]
        print(f"检测到 {len(image_files)} 张图片")
    else:
        print(f"❌ 错误: 输入源不存在 - {source}")
        sys.exit(1)

    print("=" * 50)

    try:
        # 加载模型
        print("\n正在加载模型...")
        model = YOLO(model_path)
        print("✅ 模型加载成功!")

        # 执行推理
        print("\n正在进行目标检测...")
        results = model.predict(
            source=source,
            conf=0.25,  # 置信度阈值
            save=True,  # 保存结果
            show=False  # 批量处理时建议关闭显示，提高效率
        )

        # 显示检测结果
        print("\n" + "=" * 50)
        print("检测结果统计:")
        print("=" * 50)

        total_detections = 0
        processed_images = 0

        for i, result in enumerate(results):
            boxes = result.boxes
            processed_images += 1
            total_detections += len(boxes)

            # 获取图片名称
            if hasattr(result, 'path'):
                img_name = Path(result.path).name
            else:
                img_name = f"图片 {i + 1}"

            print(f"\n[{processed_images}] {img_name}")
            print(f"  检测到 {len(boxes)} 个目标:")

            for box in boxes:
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                cls_name = result.names[cls_id]
                print(f"    - {cls_name}: 置信度 {conf:.2f}")

        # 汇总统计
        print(f"\n" + "=" * 50)
        print("检测汇总:")
        print(f"  处理图片数: {processed_images}")
        print(f"  总检测目标数: {total_detections}")
        if processed_images > 0:
            print(f"  平均每张图片: {total_detections/processed_images:.1f} 个目标")

        # 获取结果保存路径
        result_dir = Path("runs/detect/predict")
        if result_dir.exists():
            print(f"\n✅ 结果已保存到: {result_dir.absolute()}")

        print("\n" + "=" * 50)
        print("推理完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"\n🚀 程序开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    main()