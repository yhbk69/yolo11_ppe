import os
import xml.etree.ElementTree as ET
import random
import shutil
import argparse


def convert_bbox(size, box):
    """将边界框坐标转换为YOLO格式"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return [x * dw, y * dh, w * dw, h * dh]


def convert_xml_to_yolo(xml_path, classes):
    """转换单个XML文件为YOLO格式"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        yolo_lines = []
        for obj in root.iter('object'):
            # 跳过困难样本
            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue

            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue

            cls_id = classes.index(cls_name)
            bbox = obj.find('bndbox')
            box = (
                float(bbox.find('xmin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymin').text),
                float(bbox.find('ymax').text)
            )
            yolo_bbox = convert_bbox((w, h), box)
            yolo_lines.append(f"{cls_id} {' '.join(f'{x:.6f}' for x in yolo_bbox)}")

        return yolo_lines, (w, h)
    except Exception as e:
        print(f"转换错误 {xml_path}: {str(e)}")
        return None, None


def process_dataset(xml_dir, img_dir, classes, output_dir, ratios=(0.7, 0.2, 0.1)):
    """处理整个数据集并划分训练/验证/测试集"""
    # 创建输出目录
    dirs = {
        'train': {'images': os.path.join(output_dir, 'images/train'),
                  'labels': os.path.join(output_dir, 'labels/train')},
        'val': {'images': os.path.join(output_dir, 'images/val'),
                'labels': os.path.join(output_dir, 'labels/val')},
        'test': {'images': os.path.join(output_dir, 'images/test'),
                 'labels': os.path.join(output_dir, 'labels/test')}
    }

    for split in dirs.values():
        os.makedirs(split['images'], exist_ok=True)
        os.makedirs(split['labels'], exist_ok=True)

    # 收集所有有效样本
    samples = []
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(xml_dir, xml_file)
            img_name = os.path.splitext(xml_file)[0]

            # 查找匹配的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png', '.bmp',"tiff",".bmp"]
            img_path = None
            for ext in img_extensions:
                test_path = os.path.join(img_dir, img_name + ext)
                if os.path.exists(test_path):
                    img_path = test_path
                    break

            if img_path:
                samples.append((xml_path, img_path, img_name))

    # 随机打乱并划分数据集
    random.shuffle(samples)
    total = len(samples)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])

    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }

    # 处理每个样本
    for split_name, split_samples in splits.items():
        for xml_path, img_path, base_name in split_samples:
            # 转换XML到YOLO格式
            yolo_lines, img_size = convert_xml_to_yolo(xml_path, classes)

            if yolo_lines and img_size:
                # 写入标签文件
                label_path = os.path.join(dirs[split_name]['labels'], f"{base_name}.txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                # 复制图片文件
                img_dest = os.path.join(dirs[split_name]['images'], os.path.basename(img_path))
                shutil.copy2(img_path, img_dest)

    # 生成数据集配置文件
    create_yaml_config(output_dir, classes, ratios, total, dirs)

    # 返回统计数据
    stats = {k: len(v) for k, v in splits.items()}
    stats['total'] = total
    return stats


def create_yaml_config(output_dir, classes, ratios, total_samples, dirs):
    """创建YOLO数据集配置文件"""
    config_path = os.path.join(output_dir, 'dataset33.yaml')
    rel_paths = {
        'train': os.path.relpath(dirs['train']['images'], output_dir),
        'val': os.path.relpath(dirs['val']['images'], output_dir),
        'test': os.path.relpath(dirs['test']['images'], output_dir)
    }

    with open(config_path, 'w') as f:
        f.write(f"# YOLO 数据集配置文件\n")
        f.write(f"path: {output_dir}\n")
        f.write(f"train: {rel_paths['train']}\n")
        f.write(f"val: {rel_paths['val']}\n")
        f.write(f"test: {rel_paths['test']}\n\n")
        f.write(f"# 类别信息\n")
        f.write(f"nc: {len(classes)}\n")
        f.write(f"names: {classes}\n\n")
        f.write(f"# 数据集统计\n")
        f.write(f"# 总样本数: {total_samples}\n")
        f.write(f"# 训练集比例: {ratios[0] * 100:.0f}%\n")
        f.write(f"# 验证集比例: {ratios[1] * 100:.0f}%\n")
        f.write(f"# 测试集比例: {ratios[2] * 100:.0f}%\n")

    return config_path


def main():
    parser = argparse.ArgumentParser(description='将VOC XML格式转换为YOLO TXT格式并划分数据集')
    # 设置默认路径（Windows路径需用r转义）
    parser.add_argument('--xml-dir', default=r"E:\code\dataset\seat3\xml",
                        help='XML文件目录路径')
    parser.add_argument('--img-dir', default=r"E:\code\dataset\seat3\images",
                        help='图片文件目录路径')
    parser.add_argument('--output-dir', default=r"E:\code\dataset\seat3\yolo"
                        , help='输出目录路径')
    parser.add_argument('--classes', nargs='+', help='类别名称列表')
    parser.add_argument('--ratios', nargs=3, type=float, default=[0.7, 0.2, 0.1],
                        help='训练/验证/测试集比例 (默认为 0.7 0.2 0.1)')

    args = parser.parse_args()

    # 验证比例总和为1
    if sum(args.ratios) != 1.0:
        print("错误：比例总和必须为1.0")
        return

    print("=" * 50)
    print(f"开始处理数据集:")
    print(f"  XML目录: {args.xml_dir}")
    print(f"  图片目录: {args.img_dir}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  类别列表: {args.classes}")
    print(f"  划分比例: 训练 {args.ratios[0] * 100}%, 验证 {args.ratios[1] * 100}%, 测试 {args.ratios[2] * 100}%")
    print("=" * 50)

    stats = process_dataset(
        args.xml_dir,
        args.img_dir,
        args.classes,
        args.output_dir,
        tuple(args.ratios)
    )

    print(f"XML文件数: {len(os.listdir(args.xml_dir))}")
    print(f"图片文件数: {len(os.listdir(args.img_dir))}")

    print("\n处理完成! 数据集统计:")
    print(f"  总样本数: {stats['total']}")
    print(f"  训练集: {stats['train']} 个样本")
    print(f"  验证集: {stats['val']} 个样本")
    print(f"  测试集: {stats['test']} 个样本")
    print(f"配置文件已生成: {os.path.join(args.output_dir, 'dataset.yaml')}")


if __name__ == "__main__":
    main()