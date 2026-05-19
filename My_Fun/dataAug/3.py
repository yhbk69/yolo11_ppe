import os
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image, ImageEnhance
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# 配置路径
INPUT_IMAGE_FOLDER = r"E:\code\dataset\seat4\images"
INPUT_XML_FOLDER = r"E:\code\dataset\seat4\xml"
OUTPUT_IMAGE_FOLDER = r"E:\code\dataset\seat4\aug\images"
OUTPUT_XML_FOLDER = r"E:\code\dataset\seat4\aug\xml"

# 创建输出目录
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_XML_FOLDER, exist_ok=True)

# 增强方式控制标志位 (True/False 启用/禁用)
AUGMENTATIONS = {
    'flip_horizontal': True,
    'flip_vertical': False,
    'rotate': True,
    'scale': True,
    'translate': True,
    'shear': False,
    'brightness': True,
    'contrast': True,
    'blur': False,
    'noise': True
}


def parse_xml(xml_path):
    """解析XML文件，返回图像尺寸和边界框列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    boxes = []
    for obj in root.findall('object'):
        cls = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append({'class': cls, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

    return (width, height), boxes


def create_xml(xml_path, width, height, boxes):
    """创建新的XML文件"""
    root = ET.Element("annotation")

    # 添加尺寸信息
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    # 添加边界框
    for box in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = box['class']
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"

        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box['xmin']))
        ET.SubElement(bndbox, "ymin").text = str(int(box['ymin']))
        ET.SubElement(bndbox, "xmax").text = str(int(box['xmax']))
        ET.SubElement(bndbox, "ymax").text = str(int(box['ymax']))

    tree = ET.ElementTree(root)
    tree.write(xml_path)


def apply_augmentations(image, bbs):
    """应用选定的增强方式"""
    # 转换为imgaug格式
    bbs_ia = BoundingBoxesOnImage([
        BoundingBox(x1=bb['xmin'], y1=bb['ymin'], x2=bb['xmax'], y2=bb['ymax'])
        for bb in bbs
    ], shape=image.shape)

    # 1. 水平翻转
    if AUGMENTATIONS['flip_horizontal'] and random.random() > 0.5:
        aug = iaa.Fliplr(1.0)
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 2. 垂直翻转
    if AUGMENTATIONS['flip_vertical'] and random.random() > 0.5:
        aug = iaa.Flipud(1.0)
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 3. 旋转 (-30° 到 30°)
    if AUGMENTATIONS['rotate'] and random.random() > 0.5:
        angle = random.uniform(-30, 30)
        aug = iaa.Affine(rotate=angle)
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 4. 缩放 (0.8 到 1.2倍)
    if AUGMENTATIONS['scale'] and random.random() > 0.5:
        scale = random.uniform(0.8, 1.2)
        aug = iaa.Affine(scale=scale)
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 5. 平移 (-20% 到 20%)
    if AUGMENTATIONS['translate'] and random.random() > 0.5:
        tx = random.uniform(-0.2, 0.2)
        ty = random.uniform(-0.2, 0.2)
        aug = iaa.Affine(translate_percent={"x": tx, "y": ty})
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 6. 错切 (-20° 到 20°)
    if AUGMENTATIONS['shear'] and random.random() > 0.5:
        shear = random.uniform(-20, 20)
        aug = iaa.Affine(shear=shear)
        image, bbs_ia = aug(image=image, bounding_boxes=bbs_ia)

    # 7. 亮度调整 (0.5 到 1.5倍)
    if AUGMENTATIONS['brightness'] and random.random() > 0.5:
        factor = random.uniform(0.5, 1.5)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(factor)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 8. 对比度调整 (0.5 到 1.5倍)
    if AUGMENTATIONS['contrast'] and random.random() > 0.5:
        factor = random.uniform(0.5, 1.5)
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(factor)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 9. 高斯模糊
    if AUGMENTATIONS['blur'] and random.random() > 0.5:
        sigma = random.uniform(0.1, 3.0)
        aug = iaa.GaussianBlur(sigma=sigma)
        image = aug(image=image)

    # 10. 添加高斯噪声
    if AUGMENTATIONS['noise'] and random.random() > 0.5:
        aug = iaa.AdditiveGaussianNoise(scale=0.05 * 255)
        image = aug(image=image)

    # 转换回边界框格式
    new_bbs = []
    for bb in bbs_ia:
        new_bbs.append({
            'class': bbs[bbs_ia.bounding_boxes.index(bb)]['class'],
            'xmin': max(0, bb.x1),
            'ymin': max(0, bb.y1),
            'xmax': min(image.shape[1], bb.x2),
            'ymax': min(image.shape[0], bb.y2)
        })

    return image, new_bbs


# 在配置部分添加每张图片的增强次数
NUM_AUGMENTATIONS_PER_IMAGE = 5  # 每张原始图片生成5个增强版本


def main():
    """主函数：处理文件夹中的所有图像"""
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for img_file in image_files:
        image_path = os.path.join(INPUT_IMAGE_FOLDER, img_file)
        xml_file = os.path.splitext(img_file)[0] + '.xml'
        xml_path = os.path.join(INPUT_XML_FOLDER, xml_file)

        if not os.path.exists(xml_path):
            print(f"跳过 {img_file}，未找到对应的XML文件")
            continue

        # 每张图片生成多个增强版本
        for aug_idx in range(NUM_AUGMENTATIONS_PER_IMAGE):
            # 生成新文件名（包含增强序号）
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            new_image_name = f"{base_name}_aug{aug_idx + 1}.jpg"
            new_xml_name = f"{base_name}_aug{aug_idx + 1}.xml"

            # 处理并保存增强版本
            process_single_augmentation(
                image_path,
                xml_path,
                os.path.join(OUTPUT_IMAGE_FOLDER, new_image_name),
                os.path.join(OUTPUT_XML_FOLDER, new_xml_name)
            )
            print(f"已生成: {img_file} 的增强版本 {aug_idx + 1}/{NUM_AUGMENTATIONS_PER_IMAGE}")


def process_single_augmentation(image_path, xml_path, output_image_path, output_xml_path):
    """处理单个增强版本"""
    # 读取图像和XML
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    (width, height), bbs = parse_xml(xml_path)

    # 应用增强
    augmented_image, augmented_bbs = apply_augmentations(image.copy(), bbs)

    # 保存增强后的图像和XML
    cv2.imwrite(output_image_path, augmented_image)
    create_xml(output_xml_path, augmented_image.shape[1], augmented_image.shape[0], augmented_bbs)




if __name__ == "__main__":
    main()