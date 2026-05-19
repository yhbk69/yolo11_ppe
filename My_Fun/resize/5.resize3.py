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


def resize_and_augment(image_path, xml_folder, output_folder, max_size=640):
    """按原比例缩小图像并添加灰条，同时更新XML标注"""

    # 检查图像是否有效
    if not is_valid_image(image_path):
        return False

    try:
        # 使用PIL读取图像
        with Image.open(image_path) as image:
            image_np = np.array(image)
            original_height, original_width = image.size[1], image.size[0]
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

    # 计算缩放比例（保持原比例）
    if original_width > original_height:
        ratio = max_size / original_width
    else:
        ratio = max_size / original_height

    # 计算新尺寸
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # 使用PIL进行缩放
    try:
        with Image.open(image_path) as pil_image:
            # 使用高质量缩放下采样
            resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    except Exception:
        print(f"PIL图像缩放失败: {image_path}")
        return False

    # 创建640x640的灰色背景图像 (RGB模式的灰色)
    new_image = Image.new('RGB', (max_size, max_size), (128, 128, 128))

    # 计算粘贴位置（居中放置）
    paste_x = (max_size - new_width) // 2
    paste_y = (max_size - new_height) // 2

    # 将缩放后的图像粘贴到灰色背景上
    new_image.paste(resized_image, (paste_x, paste_y))

    # 转换为numpy数组
    augmented_image = np.array(new_image)

    # 调整边界框坐标（考虑灰条偏移）
    for bbox in bounding_boxes:
        bbox.x1 = bbox.x1 * ratio + paste_x
        bbox.x2 = bbox.x2 * ratio + paste_x
        bbox.y1 = bbox.y1 * ratio + paste_y
        bbox.y2 = bbox.y2 * ratio + paste_y

    # 创建BoundingBoxesOnImage对象
    bbs = BoundingBoxesOnImage(bounding_boxes, shape=augmented_image.shape)

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
        success = update_xml(tree, bbs, (max_size, max_size), annotations_output_dir, xml_filename)
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


def process_dataset(input_folder, output_folder, max_size=640):
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
            success = resize_and_augment(image_path, xml_dir, output_folder, max_size)

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
    input_folder = r"E:\code\MVSimg\IPU1(DA6999357)"
    output_folder = r"E:\code\MVSimg\2"
    max_size = 640  # 最大边长尺寸

    # 处理整个数据集
    process_dataset(input_folder, output_folder, max_size)