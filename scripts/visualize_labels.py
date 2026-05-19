#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 标签可视化脚本
将 all_labels 中的标签绘制到 all_images 图片上，保存到 img_label 目录

使用方法:
    python visualize_labels.py
    python visualize_labels.py --images <图片目录> --labels <标签目录> --output <输出目录>
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np


# 类别名称和颜色（BGR格式）
CLASSES = {
    0: ("helmet", (255, 0, 0)),       # 蓝色
    1: ("gloves", (0, 255, 0),),      # 绿色
    2: ("vest", (0, 0, 255)),         # 红色
    3: ("boots", (255, 255, 0)),      # 青色
    4: ("goggles", (255, 0, 255)),    # 紫色
    5: ("none", (128, 128, 128)),     # 灰色
    6: ("Person", (0, 165, 255)),     # 橙色
    7: ("no_helmet", (0, 100, 255)),  # 浅橙
    8: ("no_goggle", (100, 0, 255)),  # 浅紫
    9: ("no_gloves", (100, 255, 100)),# 浅绿
    10: ("no_boots", (255, 100, 100)),# 浅蓝
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLO 标签可视化")
    parser.add_argument(
        "--images",
        type=str,
        default=r"D:\dltt\Python\YOLOV11\datasets\construction-ppe\all_images",
        help="图片目录路径"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=r"D:\dltt\Python\YOLOV11\datasets\construction-ppe\all_labels",
        help="标签目录路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=r"D:\dltt\Python\YOLOV11\datasets\construction-ppe\img_label",
        help="输出目录路径"
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="边界框线宽"
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.6,
        help="字体大小"
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="只画框不显示标签文字"
    )
    return parser.parse_args()


def parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """
    解析 YOLO 格式标签文件
    
    Args:
        label_path: 标签文件路径
        img_w: 图片宽度
        img_h: 图片高度
    
    Returns:
        list: [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 转换为像素坐标
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)
                
                # 边界检查
                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))
                
                boxes.append((class_id, x1, y1, x2, y2))
            except (ValueError, IndexError) as e:
                print(f"警告: 解析行失败 '{line}': {e}")
                continue
    
    return boxes


def draw_boxes(image: np.ndarray, boxes: list, thickness: int = 2, font_scale: float = 0.6, show_label: bool = True):
    """
    在图片上绘制边界框
    
    Args:
        image: 图片数组
        boxes: [(class_id, x1, y1, x2, y2), ...]
        thickness: 线宽
        font_scale: 字体大小
        show_label: 是否显示标签文字
    
    Returns:
        np.ndarray: 绘制后的图片
    """
    for class_id, x1, y1, x2, y2 in boxes:
        if class_id not in CLASSES:
            print(f"警告: 未知的类别ID {class_id}")
            color = (255, 255, 255)
            class_name = f"class_{class_id}"
        else:
            class_name, color = CLASSES[class_id]
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制标签背景和文字
        if show_label:
            label = f"{class_name}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # 文字背景
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # 文字
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1
            )
    
    return image


def visualize_labels(images_dir: str, labels_dir: str, output_dir: str, 
                     thickness: int = 2, font_scale: float = 0.6, show_label: bool = True):
    """
    批量可视化标签
    
    Args:
        images_dir: 图片目录
        labels_dir: 标签目录
        output_dir: 输出目录
        thickness: 边界框线宽
        font_scale: 字体大小
        show_label: 是否显示标签
    """
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)
    
    # 检查输入目录
    if not images_path.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_path}")
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    # 获取所有图片文件
    image_files = [f for f in images_path.iterdir() if f.suffix.lower() in img_extensions]
    
    if not image_files:
        print(f"错误: 在 {images_path} 中未找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"输出目录: {output_path}")
    print("-" * 50)
    
    # 统计信息
    total_images = 0
    total_boxes = 0
    class_counts = {}
    errors = []
    
    for img_file in image_files:
        # 对应的标签文件
        label_file = labels_path / (img_file.stem + '.txt')
        
        # 读取图片
        image = cv2.imread(str(img_file))
        if image is None:
            errors.append(f"无法读取图片: {img_file.name}")
            continue
        
        img_h, img_w = image.shape[:2]
        
        # 解析标签
        boxes = parse_yolo_label(label_file, img_w, img_h)
        
        # 绘制边界框
        if boxes:
            image = draw_boxes(image, boxes, thickness, font_scale, show_label)
            total_boxes += len(boxes)
            
            # 统计类别
            for class_id, _, _, _, _ in boxes:
                class_name = CLASSES.get(class_id, (f"class_{class_id}",))[0]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 保存结果
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), image)
        total_images += 1
        
        # 进度显示
        if total_images % 100 == 0:
            print(f"已处理: {total_images}/{len(image_files)}")
    
    print("-" * 50)
    print(f"\n[OK] 处理完成!")
    print(f"   处理图片: {total_images} 张")
    print(f"   标注框数: {total_boxes} 个")
    print(f"   平均每张: {total_boxes/total_images:.2f} 个框" if total_images > 0 else "")
    
    if class_counts:
        print(f"\n[统计] 类别统计:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {count}")
    
    if errors:
        print(f"\n[警告] 错误列表 ({len(errors)} 个):")
        for err in errors[:10]:
            print(f"   {err}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
    
    print(f"\n[输出] 输出位置: {output_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 50)
    print("YOLO 标签可视化工具")
    print("=" * 50)
    print(f"图片目录: {args.images}")
    print(f"标签目录: {args.labels}")
    print(f"输出目录: {args.output}")
    print(f"线宽: {args.thickness}")
    print(f"字体大小: {args.font_scale}")
    print(f"显示标签: {'否' if args.no_label else '是'}")
    print("=" * 50)
    
    visualize_labels(
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        thickness=args.thickness,
        font_scale=args.font_scale,
        show_label=not args.no_label
    )


if __name__ == "__main__":
    main()
