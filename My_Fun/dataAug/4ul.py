import os
import cv2
import albumentations as A
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from glob import glob


def augment_folder(input_img_dir, input_xml_dir, output_dir, num_augmentations=5):
    """
    批量增强文件夹中的图像并同步处理对应的XML标注

    参数:
        input_img_dir: 输入图像文件夹路径
        input_xml_dir: 输入XML标注文件夹路径
        output_dir: 输出文件夹路径
        num_augmentations: 每张图像生成的增强版本数量
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)

    # 定义增强管道
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.CLAHE(p=0.3),
        A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # 获取所有图像文件
    img_files = glob(os.path.join(input_img_dir, "*.jpg")) + \
                glob(os.path.join(input_img_dir, "*.png")) + \
                glob(os.path.join(input_img_dir, "*.jpeg")) + \
                glob(os.path.join(input_img_dir, "*.bmp")) + \
                glob(os.path.join(input_img_dir, "*.tiff"))

    print(f"找到 {len(img_files)} 张图像，每张将生成 {num_augmentations} 个增强版本")

    # 处理每张图像
    for img_path in tqdm(img_files, desc="处理图像"):
        # 获取对应的XML文件路径
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        xml_path = os.path.join(input_xml_dir, f"{base_name}.xml")

        if not os.path.exists(xml_path):
            print(f"警告: 未找到 {img_name} 对应的XML文件，跳过")
            continue

        # 读取图像
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 解析XML文件
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        orig_width = int(size.find('width').text)
        orig_height = int(size.find('height').text)

        # 提取边界框和类别
        bboxes = []
        class_labels = []
        for obj in root.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            bboxes.append([xmin, ymin, xmax, ymax])
            class_labels.append(cls)

        # 为每张图像生成多个增强版本
        for aug_idx in range(num_augmentations):
            # 应用增强
            transformed = transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            # 过滤掉无效的边界框
            valid_bboxes = []
            valid_labels = []
            for bbox, label in zip(transformed_bboxes, transformed_class_labels):
                xmin, ymin, xmax, ymax = bbox
                # 检查边界框是否有效（面积大于0且在图像内）
                if xmax > xmin and ymax > ymin and \
                        xmin >= 0 and ymin >= 0 and \
                        xmax <= transformed_image.shape[1] and ymax <= transformed_image.shape[0]:
                    valid_bboxes.append(bbox)
                    valid_labels.append(label)

            # 如果增强后没有有效边界框，跳过保存
            if not valid_bboxes:
                print(f"警告: {img_name} 的增强版本 {aug_idx} 没有有效边界框，跳过保存")
                continue

            # 保存增强后的图像
            output_img_name = f"{base_name}_aug{aug_idx}.jpg"
            output_img_path = os.path.join(output_dir, "images", output_img_name)
            cv2.imwrite(output_img_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

            # 创建新的XML树
            new_root = ET.Element("annotation")

            # 添加文件夹信息
            folder = ET.SubElement(new_root, "folder")
            folder.text = "augmented"

            # 添加文件名
            filename = ET.SubElement(new_root, "filename")
            filename.text = output_img_name

            # 添加路径
            path = ET.SubElement(new_root, "path")
            path.text = output_img_path

            # 添加图像尺寸
            new_size = ET.SubElement(new_root, "size")
            width = ET.SubElement(new_size, "width")
            width.text = str(transformed_image.shape[1])
            height = ET.SubElement(new_size, "height")
            height.text = str(transformed_image.shape[0])
            depth = ET.SubElement(new_size, "depth")
            depth.text = "3"

            # 添加分割信息
            segmented = ET.SubElement(new_root, "segmented")
            segmented.text = "0"

            # 添加增强后的边界框
            for bbox, label in zip(valid_bboxes, valid_labels):
                obj = ET.SubElement(new_root, "object")

                name = ET.SubElement(obj, "name")
                name.text = label

                pose = ET.SubElement(obj, "pose")
                pose.text = "Unspecified"

                truncated = ET.SubElement(obj, "truncated")
                truncated.text = "0"

                difficult = ET.SubElement(obj, "difficult")
                difficult.text = "0"

                bndbox = ET.SubElement(obj, "bndbox")
                xmin_elem = ET.SubElement(bndbox, "xmin")
                xmin_elem.text = str(int(round(bbox[0])))
                ymin_elem = ET.SubElement(bndbox, "ymin")
                ymin_elem.text = str(int(round(bbox[1])))
                xmax_elem = ET.SubElement(bndbox, "xmax")
                xmax_elem.text = str(int(round(bbox[2])))
                ymax_elem = ET.SubElement(bndbox, "ymax")
                ymax_elem.text = str(int(round(bbox[3])))

            # 保存新的XML文件
            new_tree = ET.ElementTree(new_root)
            output_xml_name = f"{base_name}_aug{aug_idx}.xml"
            output_xml_path = os.path.join(output_dir, "xml", output_xml_name)
            new_tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    # 配置路径
    input_image_dir = r"E:\code\dataset\seat4\images"
    input_xml_dir = r"E:\code\dataset\seat4\xml"
    output_dir = r"E:\code\dataset\seat4\aug"
    num_augmentations = 5  # 每张图像生成的增强版本数量

    # 执行批量增强
    augment_folder(input_image_dir, input_xml_dir, output_dir, num_augmentations)
    print("批量增强完成！")