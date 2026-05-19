import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import random
from xml.dom import minidom

# 路径设置
INPUT_IMAGE_FOLDER = r"E:\code\dataset\seat4\images"  # 输入图像文件夹
INPUT_XML_FOLDER = r"E:\code\dataset\seat4\xml"  # 输入XML文件夹
OUTPUT_IMAGE_FOLDER = r"E:\code\dataset\seat4\aug\images"  # 输出图像文件夹
OUTPUT_XML_FOLDER = r"E:\code\dataset\seat4\aug\xml"  # 输出XML文件夹

# 创建输出文件夹（如果不存在）
os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_XML_FOLDER, exist_ok=True)

# 增强方式标志位（True表示开启，False表示关闭）
AUGMENTATION_FLAGS = {
    "brightness": True,  # 亮度调整
    "contrast": True,  # 对比度调整
    "saturation": True,  # 饱和度调整
    "hue": True,  # 色调调整
    "horizontal_flip": True,  # 水平翻转
    "vertical_flip": True,  # 垂直翻转
    "rotation": True,  # 随机旋转
    "gaussian_blur": True,  # 高斯模糊
    "noise": True,  # 噪声添加
    "random_crop": True  # 随机裁剪
}


def adjust_brightness(image, xml_tree, factor_range=(0.5, 1.5)):
    """调整图像亮度"""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted, xml_tree  # 亮度调整不影响坐标


def adjust_contrast(image, xml_tree, factor_range=(0.5, 1.5)):
    """调整图像对比度"""
    factor = random.uniform(*factor_range)
    mean = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=mean * (1 - factor))
    return adjusted, xml_tree  # 对比度调整不影响坐标


def adjust_saturation(image, xml_tree, factor_range=(0.5, 1.5)):
    """调整图像饱和度"""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * factor
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted, xml_tree  # 饱和度调整不影响坐标


def adjust_hue(image, xml_tree, factor_range=(-10, 10)):
    """调整图像色调"""
    factor = random.uniform(*factor_range)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 0] = (hsv[:, :, 0] + factor) % 180
    hsv = np.array(hsv, dtype=np.uint8)
    adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return adjusted, xml_tree  # 色调调整不影响坐标


def horizontal_flip(image, xml_tree):
    """水平翻转图像"""
    flipped = cv2.flip(image, 1)
    height, width = image.shape[:2]

    # 调整XML中的坐标
    root = xml_tree.getroot()
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        xmax = float(bbox.find('xmax').text)

        # 水平翻转后x坐标变换
        new_xmin = width - xmax
        new_xmax = width - xmin

        bbox.find('xmin').text = str(new_xmin)
        bbox.find('xmax').text = str(new_xmax)

    return flipped, xml_tree


def vertical_flip(image, xml_tree):
    """垂直翻转图像"""
    flipped = cv2.flip(image, 0)
    height, width = image.shape[:2]

    # 调整XML中的坐标
    root = xml_tree.getroot()
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        ymin = float(bbox.find('ymin').text)
        ymax = float(bbox.find('ymax').text)

        # 垂直翻转后y坐标变换
        new_ymin = height - ymax
        new_ymax = height - ymin

        bbox.find('ymin').text = str(new_ymin)
        bbox.find('ymax').text = str(new_ymax)

    return flipped, xml_tree


def rotate_image(image, xml_tree, angle_range=(-30, 30)):
    """随机旋转图像"""
    angle = random.uniform(*angle_range)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像尺寸
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵以防止裁剪
    M[0, 2] += (new_width / 2) - center[0]
    M[1, 2] += (new_height / 2) - center[1]

    # 执行旋转
    rotated = cv2.warpAffine(image, M, (new_width, new_height))

    # 调整XML中的坐标
    root = xml_tree.getroot()

    # 更新图像尺寸
    root.find('size').find('width').text = str(new_width)
    root.find('size').find('height').text = str(new_height)

    # 旋转边界框
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 定义边界框的四个点
        points = np.array([
            [xmin, ymin], [xmax, ymin],
            [xmax, ymax], [xmin, ymax]
        ], dtype=np.float32)

        # 应用旋转
        transformed_points = cv2.transform(np.array([points]), M)[0]

        # 计算新的边界框
        new_xmin = np.min(transformed_points[:, 0])
        new_ymin = np.min(transformed_points[:, 1])
        new_xmax = np.max(transformed_points[:, 0])
        new_ymax = np.max(transformed_points[:, 1])

        # 更新边界框
        bbox.find('xmin').text = str(new_xmin)
        bbox.find('ymin').text = str(new_ymin)
        bbox.find('xmax').text = str(new_xmax)
        bbox.find('ymax').text = str(new_ymax)

    return rotated, xml_tree


def gaussian_blur(image, xml_tree, ksize_range=(3, 7)):
    """高斯模糊"""
    ksize = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))  # 确保是奇数
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred, xml_tree  # 高斯模糊不影响坐标


def add_noise(image, xml_tree, noise_level_range=(10, 50)):
    """添加高斯噪声"""
    noise_level = random.randint(*noise_level_range)
    row, col, ch = image.shape
    mean = 0
    var = noise_level
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, xml_tree  # 噪声添加不影响坐标


def random_crop(image, xml_tree, crop_ratio_range=(0.7, 0.9)):
    """随机裁剪"""
    height, width = image.shape[:2]
    crop_ratio = random.uniform(*crop_ratio_range)
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)

    # 随机选择裁剪区域
    x_start = random.randint(0, width - crop_width)
    y_start = random.randint(0, height - crop_height)

    # 执行裁剪
    cropped = image[y_start:y_start + crop_height, x_start:x_start + crop_width]

    # 调整XML中的坐标
    root = xml_tree.getroot()

    # 更新图像尺寸
    root.find('size').find('width').text = str(crop_width)
    root.find('size').find('height').text = str(crop_height)

    # 调整边界框坐标
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # 计算裁剪后的坐标
        new_xmin = max(0, xmin - x_start)
        new_ymin = max(0, ymin - y_start)
        new_xmax = min(crop_width, xmax - x_start)
        new_ymax = min(crop_height, ymax - y_start)

        # 如果边界框完全在裁剪区域外，则移除该对象
        if new_xmin >= new_xmax or new_ymin >= new_ymax:
            root.remove(obj)
            continue

        # 更新边界框
        bbox.find('xmin').text = str(new_xmin)
        bbox.find('ymin').text = str(new_ymin)
        bbox.find('xmax').text = str(new_xmax)
        bbox.find('ymax').text = str(new_ymax)

    return cropped, xml_tree


def prettify_xml(elem):
    """格式化XML输出"""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def process_image(image_path, xml_path):
    """处理单张图像和对应的XML文件"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 读取XML文件
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"处理XML文件出错 {xml_path}: {e}")
        return

    # 获取文件名（不含扩展名）
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # 原始图像和XML也保存一份
    cv2.imwrite(os.path.join(OUTPUT_IMAGE_FOLDER, f"{filename}_original.jpg"), image)
    with open(os.path.join(OUTPUT_XML_FOLDER, f"{filename}_original.xml"), 'w', encoding='utf-8') as f:
        f.write(prettify_xml(tree.getroot()))

    # 应用各种增强方式
    augmentations = [
        ("brightness", adjust_brightness),
        ("contrast", adjust_contrast),
        ("saturation", adjust_saturation),
        ("hue", adjust_hue),
        ("horizontal_flip", horizontal_flip),
        ("vertical_flip", vertical_flip),
        ("rotation", rotate_image),
        ("gaussian_blur", gaussian_blur),
        ("noise", add_noise),
        ("random_crop", random_crop)
    ]

    for aug_name, aug_func in augmentations:
        if AUGMENTATION_FLAGS.get(aug_name, False):
            # 对原始图像进行增强（每次增强都基于原始图像，而不是链式增强）
            augmented_img, augmented_tree = aug_func(image.copy(), ET.parse(xml_path))

            # 保存增强后的图像
            output_img_path = os.path.join(OUTPUT_IMAGE_FOLDER, f"{filename}_{aug_name}.jpg")
            cv2.imwrite(output_img_path, augmented_img)

            # 保存增强后的XML
            output_xml_path = os.path.join(OUTPUT_XML_FOLDER, f"{filename}_{aug_name}.xml")
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(prettify_xml(augmented_tree.getroot()))

            print(f"已生成增强图像: {output_img_path} 和对应的XML: {output_xml_path}")


def main():
    """主函数"""
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp','tiff']
    image_files = [f for f in os.listdir(INPUT_IMAGE_FOLDER)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    print(f"找到 {len(image_files)} 个图像文件，开始处理...")

    # 处理每个图像文件
    for image_file in image_files:
        image_path = os.path.join(INPUT_IMAGE_FOLDER, image_file)
        xml_filename = os.path.splitext(image_file)[0] + '.xml'
        xml_path = os.path.join(INPUT_XML_FOLDER, xml_filename)

        if os.path.exists(xml_path):
            process_image(image_path, xml_path)
        else:
            print(f"警告: 未找到 {image_file} 对应的XML文件 {xml_path}，跳过此图像")

    print("所有图像处理完成！")


if __name__ == "__main__":
    main()
