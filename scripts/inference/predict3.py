import os
import time
import cv2
from ultralytics import YOLO
from PIL import Image
import supervision as sv


def setup_yolo_model(model_type='yolo11x.pt'):
    """
    加载YOLO模型
    model_type: 模型类型，如 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt' 等
    """
    # 加载预训练的YOLO模型[3](@ref)
    model = YOLO(model_type)
    print(f"已加载YOLO模型: {model_type}")
    return model


def process_images_with_yolo(model, img_path, confidence_threshold=0.5):
    """
    使用YOLO模型处理图像
    """
    # 检查路径是文件还是目录
    if os.path.isfile(img_path):
        image_paths = [img_path]
    elif os.path.isdir(img_path):
        # 获取目录中所有图片文件
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = [os.path.join(img_path, f) for f in os.listdir(img_path)
                       if f.lower().endswith(supported_formats)]
    else:
        raise ValueError("路径既不是文件也不是目录")

    if not image_paths:
        print("未找到支持的图像文件")
        return

    # 记录总开始时间
    total_start_time = time.time()

    # 创建结果保存目录
    result_dir = os.path.join(os.path.dirname(img_path), 'yolo_results3')
    os.makedirs(result_dir, exist_ok=True)

    # 处理所有图像路径
    for i, img_path in enumerate(image_paths):
        # 记录单张图片开始时间
        image_start_time = time.time()

        print(f"\n处理第 {i + 1}/{len(image_paths)} 张图片: {img_path}")

        try:
            # 使用YOLO进行预测[3](@ref)
            detection_start = time.time()
            results = model.predict(
                source=img_path,
                conf=confidence_threshold,
                save=False,  # 我们不使用YOLO自带的保存功能，自己处理结果
                verbose=False  # 减少输出噪音
            )
            detection_time = time.time() - detection_start
            print(f"YOLO目标检测耗时: {detection_time:.5f}秒")

            # 获取第一个结果（因为一次只处理一张图片）
            result = results[0]

            # 将YOLO结果转换为supervision可用的格式[8](@ref)
            annotation_start = time.time()

            # 使用YOLO的plot方法绘制结果
            annotated_image = result.plot()  # 返回BGR格式的numpy数组

            # 转换为RGB格式用于保存
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

            annotation_time = time.time() - annotation_start
            print(f"图像标注耗时: {annotation_time:.2f}秒")

            # 保存标注后的图像
            save_start = time.time()
            output_filename = f"yolo_result_{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
            output_path = os.path.join(result_dir, output_filename)

            # 使用PIL保存图像
            Image.fromarray(annotated_image_rgb).save(output_path)
            save_time = time.time() - save_start

            # 计算单张图片总耗时
            image_total_time = time.time() - image_start_time

            # 输出检测统计信息[8](@ref)
            detections_count = len(result.boxes) if result.boxes is not None else 0
            print(f"图片 {os.path.basename(img_path)} 处理完成, 总耗时: {image_total_time:.2f}秒")
            print(f"检测到 {detections_count} 个目标")
            print(f"结果已保存到: {output_path}")

            # 显示详细的检测信息（可选）
            if detections_count > 0:
                print("检测到的目标详情:")
                for j, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].item())
                    confidence = box.conf[0].item()
                    class_name = model.names[class_id]
                    bbox = box.xyxy[0].tolist()
                    print(f"  {j + 1}. {class_name}: {confidence:.2f} (位置: {[round(x) for x in bbox]})")

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
            continue

    # 计算并输出总统计信息
    total_time = time.time() - total_start_time
    print(f"\n{'=' * 50}")
    print(f"YOLO目标检测完成!")
    print(f"总图片数量: {len(image_paths)}张")
    print(f"总处理时间: {total_time:.2f}秒")
    if image_paths:
        print(f"平均每张图片处理时间: {total_time / len(image_paths):.2f}秒")
    print(f"结果保存目录: {result_dir}")
    print(f"{'=' * 50}")


def main():
    """
    主函数：YOLO目标检测程序
    """
    # 配置参数
    img_path = r"C:\Users\fxleg\Desktop\val2017"  # 替换为您的图像路径
    model_type = 'yolo11n.pt'  # 可选的模型: 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
    confidence_threshold = 0.5  # 置信度阈值

    print("开始YOLO目标检测程序...")

    try:
        # 1. 设置YOLO模型[3](@ref)
        model = setup_yolo_model(model_type)

        # 2. 处理图像
        process_images_with_yolo(model, img_path, confidence_threshold)

    except Exception as e:
        print(f"程序执行出错: {str(e)}")

    print("程序执行完毕!")


# 高级版本：使用supervision进行更精细的标注控制
def advanced_yolo_detection(model, img_path, confidence_threshold=0.5):
    """
    使用supervision进行更高级的标注控制[8](@ref)
    """
    # 图像路径处理（同上）
    if os.path.isfile(img_path):
        image_paths = [img_path]
    elif os.path.isdir(img_path):
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_paths = [os.path.join(img_path, f) for f in os.listdir(img_path)
                       if f.lower().endswith(supported_formats)]
    else:
        raise ValueError("路径既不是文件也不是目录")

    # 创建高级结果目录
    advanced_result_dir = os.path.join(os.path.dirname(img_path), 'yolo_advanced_results')
    os.makedirs(advanced_result_dir, exist_ok=True)

    # 初始化supervision的标注器[8](@ref)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.ColorPalette.DEFAULT
    )

    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.BLACK,
        text_scale=0.5,
        text_thickness=1
    )

    for i, img_path in enumerate(image_paths):
        print(f"\n高级处理第 {i + 1}/{len(image_paths)} 张图片: {img_path}")

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue

        # YOLO预测
        results = model.predict(source=image, conf=confidence_threshold, verbose=False)
        result = results[0]

        # 转换为supervision的Detections对象[8](@ref)
        detections = sv.Detections.from_ultralytics(result)

        # 自定义标注
        labels = [
            f"{model.names[class_id]} {confidence:.2f}"
            for _, _, confidence, class_id, _ in detections
        ]

        # 绘制标注
        annotated_image = image.copy()
        annotated_image = box_annotator.annotate(annotated_image, detections)
        annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)

        # 保存结果
        output_path = os.path.join(advanced_result_dir, f"advanced_{os.path.basename(img_path)}")
        cv2.imwrite(output_path, annotated_image)
        print(f"高级标注结果已保存: {output_path}")


if __name__ == "__main__":
    # 运行基本版本
    main()

    # 如果需要运行高级版本，取消注释下面的代码
    # model = setup_yolo_model('yolo11n.pt')
    # advanced_yolo_detection(model, r"C:\Users\fxleg\Desktop\val2017")