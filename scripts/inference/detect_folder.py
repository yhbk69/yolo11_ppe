"""
YOLOv11 文件夹批量图片检测脚本
统计每张图片的检测时间，计算总时间和平均时间
使用方法: python detect_folder.py [folder_path]
"""
import time
import sys
from pathlib import Path
from ultralytics import YOLO


def main():
    # 权重路径
    weight_path = Path(r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt")

    # 图片文件夹路径（默认为 construction-ppe 的 all_images）
    folder_path = Path(r"D:\dltt\Python\helmet_test_imgs")

    # 允许的图片格式
    img_formats = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    print("=" * 60)
    print("YOLOv11 文件夹批量图片检测")
    print("=" * 60)
    print(f"权重: {weight_path}")
    print(f"文件夹: {folder_path}")
    print("=" * 60)

    # 检查文件夹是否存在
    if not folder_path.exists():
        print(f"错误: 文件夹不存在 - {folder_path}")
        sys.exit(1)

    # 获取所有图片文件
    image_files = [f for f in folder_path.iterdir() if f.suffix.lower() in img_formats]

    if not image_files:
        print(f"错误: 文件夹中没有找到图片 - {folder_path}")
        sys.exit(1)

    print(f"\n找到 {len(image_files)} 张图片")
    print("-" * 60)

    try:
        # 加载模型
        print("正在加载模型...")
        model = YOLO(weight_path)
        print("模型加载成功!\n")

        # 存储每张图片的检测时间和目标数
        detection_times = []
        object_counts = []

        # 遍历每张图片进行检测
        for i, img_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] 正在检测: {img_path.name}", end=" ... ")

            # 记录单张图片的检测开始时间
            start_time = time.time()

            # 执行推理
            results = model.predict(
                source=str(img_path),
                conf=0.25,
                save=True,
                verbose=False
            )

            # 计算单张图片的检测时间
            detect_time = time.time() - start_time
            detection_times.append(detect_time)

            # 统计检测到的目标数量
            obj_count = len(results[0].boxes)
            object_counts.append(obj_count)

            print(f"耗时: {detect_time:.3f}s, 检测到 {obj_count} 个目标")

        # 计算统计信息
        total_time = sum(detection_times)
        avg_time = total_time / len(detection_times) if detection_times else 0
        min_time = min(detection_times) if detection_times else 0
        max_time = max(detection_times) if detection_times else 0

        # 输出统计结果
        print("\n" + "=" * 60)
        print("检测统计结果")
        print("=" * 60)
        print(f"图片总数:     {len(image_files)} 张")
        print(f"检测到目标:  {sum(object_counts)} 个")
        print("-" * 60)
        print(f"总检测时间:   {total_time:.3f} 秒")
        print(f"平均检测时间: {avg_time:.3f} 秒/张")
        print(f"最快检测时间: {min_time:.3f} 秒")
        print(f"最慢检测时间: {max_time:.3f} 秒")
        print("-" * 60)

        # 每张图片的详细时间
        print("\n每张图片检测时间详情:")
        print("-" * 60)
        print(f"{'文件名':<30} {'耗时(秒)':<10} {'目标数'}")
        print("-" * 60)
        for img_path, dt, obj_count in zip(image_files, detection_times, object_counts):
            print(f"{img_path.name:<30} {dt:<10.3f} {obj_count}")

        # 结果保存路径
        result_dir = Path("runs/detect/predict")
        if result_dir.exists():
            print(f"\n结果已保存到: {result_dir.absolute()}")

        print("\n检测完成!")

    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
