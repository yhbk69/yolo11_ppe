import os
import xml.etree.ElementTree as ET
from PIL import Image, UnidentifiedImageError
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np


def safe_parse_xml(xml_path):
    """安全解析XML文件，包含错误处理"""
    try:
        tree = ET.parse(xml_path)
        return tree, tree.getroot()
    except ET.ParseError:
        print(f"XML解析错误: {xml_path}")
        return None, None
    except FileNotFoundError:
        print(f"XML文件未找到: {xml_path}")
        return None, None
    except Exception:
        print(f"解析XML时发生意外错误: {xml_path}")
        return None, None


def is_valid_image(image_path):
    """检查图像是否有效且未损坏"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, UnidentifiedImageError):
        print(f"图像损坏或格式不支持: {image_path}")
        return False
    except Exception:
        print(f"检查图像时发生意外错误: {image_path}")
        return False


def resize_and_augment(image_path, xml_folder, output_folder, target_size=(640, 640)):
    """调整图像大小并增强，同时更新XML标注"""

    # 检查图像是否有效
    if not is_valid_image(image_path):
        return False

    try:
        # 读取图像
        with Image.open(image_path) as image:
            image_np = np.array(image)
    except Exception:
        print(f"读取图像失败: {image_path}")
        return False

    # 获取并解析对应的XML文件
    xml_filename = os.path.splitext(os.path.basename(image_path))[0] + '.xml'
    xml_path = os.path.join(xml_folder, xml_filename)

    tree, root = safe_parse_xml(xml_path)
    if root is None:
        return False

    # 提取XML中的原始边界框信息
    bounding_boxes = []
    object_tags = list(root.iter('object'))

    for object_tag in object_tags:
        bbox_tag = object_tag.find('bndbox')
        if bbox_tag is None:
            print(f"找不到边界框标签: {xml_path}")
            continue

        try:
            xmin = float(bbox_tag.find('xmin').text)
            ymin = float(bbox_tag.find('ymin').text)
            xmax = float(bbox_tag.find('xmax').text)
            ymax = float(bbox_tag.find('ymax').text)
            bounding_boxes.append(BoundingBox(x1=xmin, y1=ymin, x2=xmax, y2=ymax))
        except (AttributeError, ValueError, TypeError):
            print(f"边界框坐标格式错误: {xml_path}")
            continue

    # 检查是否找到有效的边界框
    if not bounding_boxes:
        print(f"未找到有效边界框: {xml_path}")
        return False

    # 创建BoundingBoxesOnImage对象
    bbs = BoundingBoxesOnImage(bounding_boxes, shape=image_np.shape)

    # 定义缩放序列
    seq = iaa.Sequential([
        iaa.Resize({"height": target_size[0], "width": target_size[1]})
    ])

    try:
        # 应用缩放到图像和边界框
        augmented_image, bbs_aug = seq(image=image_np, bounding_boxes=bbs)
    except Exception:
        print(f"图像增强失败: {image_path}")
        return False

    # 确保输出目录存在
    images_output_dir = os.path.join(output_folder, 'images')
    annotations_output_dir = os.path.join(output_folder, 'annotations')
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(annotations_output_dir, exist_ok=True)

    try:
        # 保存调整后的图像
        output_image_path = os.path.join(images_output_dir, os.path.basename(image_path))
        Image.fromarray(augmented_image).save(output_image_path)

        # 更新XML文件并保存
        success = update_xml(tree, bbs_aug, target_size, annotations_output_dir, xml_filename)
        return success
    except Exception:
        print(f"保存结果失败: {image_path}")
        return False


def update_xml(tree, bbs_aug, new_size, output_folder, xml_filename):
    """更新XML文件中的尺寸和边界框信息"""

    try:
        root = tree.getroot()

        # 更新图像尺寸
        for size_tag in root.iter('size'):
            size_tag.find('width').text = str(new_size[1])
            size_tag.find('height').text = str(new_size[0])

        # 更新对象边界框坐标
        object_tags = list(root.iter('object'))

        # 检查边界框数量是否匹配
        if len(object_tags) != len(bbs_aug.bounding_boxes):
            print("XML中的对象数量与边界框数量不匹配")
            min_length = min(len(object_tags), len(bbs_aug.bounding_boxes))
        else:
            min_length = len(object_tags)

        # 只更新匹配数量的边界框
        for i in range(min_length):
            object_tag = object_tags[i]
            bbox_aug = bbs_aug.bounding_boxes[i]

            bbox_tag = object_tag.find('bndbox')
            if bbox_tag is None:
                continue

            # 确保坐标不超出图像边界且有效
            x1 = max(0, int(bbox_aug.x1))
            y1 = max(0, int(bbox_aug.y1))
            x2 = min(new_size[1], int(bbox_aug.x2))
            y2 = min(new_size[0], int(bbox_aug.y2))

            # 只有当边界框有效时才更新
            if x1 < x2 and y1 < y2:
                bbox_tag.find('xmin').text = str(x1)
                bbox_tag.find('ymin').text = str(y1)
                bbox_tag.find('xmax').text = str(x2)
                bbox_tag.find('ymax').text = str(y2)
            else:
                print(f"无效的边界框坐标: {x1}, {y1}, {x2}, {y2}")

        # 保存更新后的XML
        output_xml_path = os.path.join(output_folder, xml_filename)
        tree.write(output_xml_path)
        return True

    except Exception:
        print(f"更新XML失败: {xml_filename}")
        return False


def process_dataset(input_folder, output_folder, target_size=(640, 640)):
    """处理整个数据集"""

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取图像目录路径
    images_dir = os.path.join(input_folder, 'images')
    xml_dir = os.path.join(input_folder, 'xml')

    # 检查输入目录是否存在
    if not os.path.exists(images_dir):
        print(f"图像目录不存在: {images_dir}")
        return False

    if not os.path.exists(xml_dir):
        print(f"XML目录不存在: {xml_dir}")
        return False

    # 支持的图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 遍历处理所有图片
    success_count = 0
    fail_count = 0
    skipped_count = 0

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(images_dir, filename)

            # 检查对应的XML文件是否存在
            xml_filename = os.path.splitext(filename)[0] + '.xml'
            xml_path = os.path.join(xml_dir, xml_filename)

            if not os.path.exists(xml_path):
                print(f"跳过 {filename}: 对应的XML文件不存在")
                skipped_count += 1
                continue

            print(f"处理: {filename}")
            success = resize_and_augment(image_path, xml_dir, output_folder, target_size)

            if success:
                success_count += 1
            else:
                fail_count += 1
        else:
            skipped_count += 1

    print(f"处理完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skipped_count}")
    return success_count > 0


# 使用示例
if __name__ == "__main__":
    input_folder = r"E:\code\dataset\seat4"
    output_folder = r"E:\code\dataset\seat4\640"
    target_size = (640, 640)

    # 处理整个数据集
    process_dataset(input_folder, output_folder, target_size)