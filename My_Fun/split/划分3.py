import os
import random
import shutil
import xml.etree.ElementTree as ET
import yaml
from pathlib import Path


def convert_xml_to_yolo(xml_file_path, output_label_path, class_names):
    """
    将单个XML文件转换为YOLO格式的TXT文件。

    Args:
        xml_file_path (str): XML标注文件的路径。
        output_label_path (str): 输出的YOLO格式TXT文件路径。
        class_names (list): 类别名称列表，索引即类别ID。
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        if size is None:
            # 如果XML中没有尺寸信息，尝试从图像文件获取
            img_width, img_height = get_image_size(xml_file_path)
        else:
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

        yolo_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in class_names:
                class_names.append(cls_name)  # 动态添加新类别
            class_id = class_names.index(cls_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # 计算YOLO格式的归一化中心坐标和宽高
            x_center = (xmin + xmax) / (2.0 * img_width)
            y_center = (ymin + ymax) / (2.0 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # 确保坐标在[0,1]范围内
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # 格式: class_id x_center y_center width height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 写入YOLO格式的标签文件
        with open(output_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))
        return True
    except Exception as e:
        print(f"转换XML文件 {xml_file_path} 时出错: {e}")
        return False


def get_image_size(xml_file_path):
    """
    尝试从对应的图像文件获取尺寸信息。

    Args:
        xml_file_path (str): XML文件的路径。

    Returns:
        tuple: (width, height) 图像尺寸。
    """
    # 这里需要实现从图像文件读取尺寸的逻辑
    # 由于复杂性，这里返回默认值，实际使用时应该用OpenCV等库读取图像尺寸
    return 640, 480  # 默认值，实际使用时应该替换为真实尺寸获取逻辑


def split_dataset(images_dir, labels_dir, output_base_dir, ratios=(0.7, 0.2, 0.1)):
    """
    划分数据集为训练集、验证集和测试集，并复制文件到相应目录。

    Args:
        images_dir (str): 源图像文件夹路径。
        labels_dir (str): 源标签（XML）文件夹路径。
        output_base_dir (str): 输出数据集的基础目录。
        ratios (tuple): 训练、验证、测试集的比例，默认为(0.7, 0.2, 0.1)。

    Returns:
        tuple: (splits, train_files, val_files, test_files, class_names)
    """
    # 创建输出目录结构[6,7](@ref)
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, 'labels', split), exist_ok=True)

    # 获取所有图像文件名（不带扩展名）
    image_files = []
    for f in os.listdir(images_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_files.append(os.path.splitext(f)[0])

    if not image_files:
        raise ValueError(f"在 {images_dir} 中未找到图像文件")

    random.shuffle(image_files)  # 随机打乱

    # 计算各集合数量[6,7](@ref)
    total_count = len(image_files)
    train_count = int(total_count * ratios[0])
    val_count = int(total_count * ratios[1])

    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    class_names = []  # 用于收集所有类别名称

    # 复制图像和标签文件到目标目录
    for split, files in zip(splits, [train_files, val_files, test_files]):
        for file_base in files:
            # 查找源图像文件（考虑不同扩展名）
            src_image = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                potential_path = os.path.join(images_dir, file_base + ext)
                if os.path.exists(potential_path):
                    src_image = potential_path
                    break

            if not src_image:
                print(f"警告: 未找到图像文件 {file_base}，跳过。")
                continue

            src_label_xml = os.path.join(labels_dir, file_base + '.xml')
            if not os.path.exists(src_label_xml):
                print(f"警告: 未找到XML标签文件 {src_label_xml}，跳过。")
                continue

            # 定义目标路径
            dst_image = os.path.join(output_base_dir, 'images', split, os.path.basename(src_image))
            dst_label_txt = os.path.join(output_base_dir, 'labels', split, file_base + '.txt')

            # 复制图像文件
            shutil.copy2(src_image, dst_image)
            # 转换XML标签为YOLO格式并保存
            if not convert_xml_to_yolo(src_label_xml, dst_label_txt, class_names):
                print(f"警告: 转换 {src_label_xml} 失败，但图像已复制")

    return splits, train_files, val_files, test_files, class_names


def create_yaml_file(output_base_dir, class_names, splits):
    """
    创建YOLO数据集配置文件[8](@ref)。

    Args:
        output_base_dir (str): 输出数据集的基础目录。
        class_names (list): 类别名称列表。
        splits (list): 数据集划分的列表，例如 ['train', 'val', 'test']。
    """
    # 使用相对路径，更便于移植[5](@ref)
    data = {
        'path': output_base_dir,
        'train': f"images/{splits[0]}",
        'val': f"images/{splits[1]}",
        'nc': len(class_names),
        'names': class_names
    }

    # 如果有测试集则添加
    if len(splits) > 2:
        data['test'] = f"images/{splits[2]}"

    yaml_path = os.path.join(output_base_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"YAML配置文件已生成: {yaml_path}")
    return yaml_path


def main():
    """
    主函数，执行数据集转换和划分的全部流程。
    """
    # 使用你提供的路径[1,2](@ref)
    # base_path = r"E:\code\dataset\seat4"
    base_path=r"E:\code\dataset\seat4\aug"
    # 检查基础路径是否存在[2,3](@ref)
    if not os.path.exists(base_path):
        print(f"错误: 基础路径 {base_path} 不存在")
        return

    # 配置路径
    images_dir = os.path.join(base_path, 'images')  # 你的原始图像文件夹
    xml_labels_dir = os.path.join(base_path, 'xml')  # 你的原始XML标签文件夹
    output_base_dir = os.path.join(base_path, 'yolo')  # 输出YOLO格式数据集的基础目录

    # 检查源目录是否存在[2,3](@ref)
    if not os.path.exists(images_dir):
        print(f"错误: 图像目录 {images_dir} 不存在")
        return

    if not os.path.exists(xml_labels_dir):
        print(f"错误: XML标签目录 {xml_labels_dir} 不存在")
        return

    print("开始处理数据集...")
    print(f"图像目录: {images_dir}")
    print(f"XML标签目录: {xml_labels_dir}")
    print(f"输出目录: {output_base_dir}")

    try:
        # 划分数据集并转换标签格式
        splits, train_files, val_files, test_files, class_names = split_dataset(
            images_dir, xml_labels_dir, output_base_dir
        )

        print(f"数据集划分完成：训练集 {len(train_files)} 张, 验证集 {len(val_files)} 张, 测试集 {len(test_files)} 张。")
        print(f"识别到的类别: {class_names}")

        # 生成YAML文件
        yaml_path = create_yaml_file(output_base_dir, class_names, splits)

        print("处理完成！")
        print(f"请检查生成的数据集目录: {output_base_dir}")
        print(f"请使用YAML文件进行训练: {yaml_path}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()