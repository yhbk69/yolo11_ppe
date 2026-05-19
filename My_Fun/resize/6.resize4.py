import os
from PIL import Image, UnidentifiedImageError
import numpy as np


def process2(image_path, output_folder, target_size=(640, 640)):
    """
    拉伸图像到指定尺寸（不保持原比例，不添加灰条）

    参数:
        image_path: 输入图像路径
        output_folder: 输出文件夹
        target_size: 目标尺寸 (宽, 高)

    返回:
        bool: 处理成功返回True，否则返回False
    """
    try:
        # 使用PIL读取图像
        with Image.open(image_path) as img:
            # 直接拉伸到目标尺寸（不保持原比例）
            stretched_image = img.resize(target_size, Image.Resampling.LANCZOS)

            # 确保输出目录存在
            os.makedirs(output_folder, exist_ok=True)

            # 保存处理后的图像
            output_path = os.path.join(output_folder, os.path.basename(image_path))
            stretched_image.save(output_path)

            print(f"成功拉伸: {os.path.basename(image_path)}, 目标尺寸: {target_size[0]}x{target_size[1]}")
            return True

    except Exception as e:
        print(f"处理图像失败: {image_path}, 错误: {str(e)}")
        return False


def is_valid_image(image_path):
    """检查图像是否有效且未损坏"""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError, UnidentifiedImageError):
        print(f"图像损坏或格式不支持: {image_path}")
        return False
    except Exception as e:
        print(f"检查图像时发生意外错误: {image_path}, 错误: {str(e)}")
        return False


def resize_and_pad(image_path, output_folder, max_size=640, pad_color=(128, 128, 128)):
    """
    按原比例缩小图像并添加灰条（只处理图片）

    参数:
        image_path: 输入图像路径
        output_folder: 输出文件夹
        max_size: 目标尺寸（正方形边长）
        pad_color: 填充颜色 (R, G, B)

    返回:
        bool: 处理成功返回True，否则返回False
    """
    # 检查图像是否有效
    if not is_valid_image(image_path):
        return False

    try:
        # 使用PIL读取图像
        with Image.open(image_path) as image:
            # 获取原始尺寸 (PIL的size返回的是(width, height))
            original_width, original_height = image.size

            # 计算缩放比例（保持原比例）
            if original_width > original_height:
                ratio = max_size / original_width
            else:
                ratio = max_size / original_height

            # 计算新尺寸
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            # 使用高质量缩放下采样
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 创建max_size×max_size的灰色背景图像
            new_image = Image.new('RGB', (max_size, max_size), pad_color)

            # 计算粘贴位置（居中放置）
            paste_x = (max_size - new_width) // 2
            paste_y = (max_size - new_height) // 2

            # 将缩放后的图像粘贴到灰色背景上
            new_image.paste(resized_image, (paste_x, paste_y))

            # 确保输出目录存在
            images_output_dir = output_folder
            os.makedirs(images_output_dir, exist_ok=True)

            # 保存处理后的图像
            output_image_path = os.path.join(images_output_dir, os.path.basename(image_path))
            new_image.save(output_image_path)

            print(
                f"成功处理: {os.path.basename(image_path)}, 原始尺寸: {original_width}x{original_height}, 新尺寸: {max_size}x{max_size}")
            return True

    except Exception as e:
        print(f"处理图像失败: {image_path}, 错误: {str(e)}")
        return False


def process_images(input_folder, output_folder, max_size=640):
    """
    处理整个目录中的图片

    参数:
        input_folder: 输入文件夹路径（包含images子目录）
        output_folder: 输出文件夹路径
        max_size: 目标尺寸（正方形边长）
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 获取图像目录路径
    images_dir = input_folder

    # 检查输入目录是否存在
    if not os.path.exists(images_dir):
        print(f"图像目录不存在: {images_dir}")
        return False

    # 支持的图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')

    # 遍历处理所有图片
    success_count = 0
    fail_count = 0
    skipped_count = 0

    print(f"开始处理目录: {images_dir}")
    print(f"输出目录: {output_folder}")
    print(f"目标尺寸: {max_size}x{max_size}")
    print("-" * 50)

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(images_dir, filename)

            success = resize_and_pad(image_path, output_folder, max_size)

            if success:
                success_count += 1
            else:
                fail_count += 1
        else:
            skipped_count += 1

    print("-" * 50)
    print(f"处理完成: 成功 {success_count}, 失败 {fail_count}, 跳过 {skipped_count}")
    return success_count > 0


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出路径
    input_folder = r"E:\code\MVSimg\IPU1(DA6999357)"  # 输入目录（应包含images子文件夹）
    output_folder = r"E:\code\MVSimg\3"  # 输出目录
    max_size = 640  # 最大边长尺寸

    # 处理整个目录中的图片
    process2(input_folder, output_folder, max_size)