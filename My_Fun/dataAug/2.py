import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import shutil
import matplotlib.pyplot as plt

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


def get_image_format(img_path):
    """自动检测图片格式，如果无法识别则默认保存为 JPEG"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']
    ext = os.path.splitext(img_path)[1].lower()
    return ext if ext in valid_extensions else '.jpg'


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id),encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):
        bndbox = object.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        bndboxlist.append([xmin, ymin, xmax, ymax])
    return bndboxlist


def change_xml_list_annotation(root, image_id, new_target, saveroot, id, img_name, original_ext, save_format='keep'):
    """
    root: XML文件目录
    image_id: 文件名（无扩展名）
    new_target: 新的边界框列表
    saveroot: XML保存目录
    id: 增强ID
    img_name: 图像基础名
    original_ext: 原始图像扩展名 (e.g., '.jpg')
    save_format: 'keep' 保留原格式，或指定如 '.jpg'
    """
    in_file = open(os.path.join(root, str(image_id) + '.xml'))
    tree = ET.parse(in_file)
    elem = tree.find('filename')

    # 确定保存的图像扩展名
    if save_format == 'keep':
        save_ext = original_ext
    else:
        save_ext = save_format

    elem.text = (img_name + str("_%06d" % int(id)) + save_ext)
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):
        bndbox = object.find('bndbox')
        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)
        index = index + 1

    tree.write(os.path.join(saveroot, img_name + str("_%06d" % int(id)) + '.xml'))


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":

    IMG_DIR = r"E:\code\dataset\seat4\images"  ### 原始数据集图像的路径
    XML_DIR = r"E:\code\dataset\seat4\xml"  ### 原始xml文件的路径

    AUG_XML_DIR = r"E:\code\dataset\seat4\aug\xml"  ### 数据增强后的xml文件的保存路径
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)

    AUG_IMG_DIR = r"E:\code\dataset\seat4\images\aug\images"  ### 数据增强后图片的保存路径
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)

    # 增强配置
    AUGLOOP = 5  # 每张影像增强的数量
    SAVE_FORMAT = 'keep'  # 'keep' 保留原格式，或指定如 '.jpg'、'.png'

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强序列
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        iaa.Fliplr(0.5),  # 镜像
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.GaussianBlur(sigma=(0, 3.0)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_px={"x": 15, "y": 15},
            scale=(0.8, 0.95),
            rotate=(-30, 30)
        )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
    ])

    for root, sub_folders, files in os.walk(XML_DIR):
        for name in files:
            if not name.endswith('.xml'):
                continue
            print("Processing:", name)
            base_name = name[:-4]  # 移除 '.xml'

            # 查找原始图像路径（支持多种格式）
            original_img_path = None
            original_ext = None
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif']:
                test_path = os.path.join(IMG_DIR, base_name + ext)
                if os.path.exists(test_path):
                    original_img_path = test_path
                    original_ext = ext
                    break

            if original_img_path is None:
                print(f"Warning: No image found for {base_name}. Skipping.")
                continue

            bndbox = read_xml_annotation(XML_DIR, name)
            # 复制原XML和原图到增强目录
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(original_img_path, AUG_IMG_DIR)

            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
                # 读取图片
                img = Image.open(original_img_path)
                # 如果图像有透明度通道（RGBA），转换为RGB（JPEG不支持透明度）
                if img.mode == 'RGBA' and (
                        SAVE_FORMAT == '.jpg' or SAVE_FORMAT == 'keep' and original_ext in ['.jpg', '.jpeg']):
                    img = img.convert('RGB')
                img_np = np.asarray(img)

                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img_np.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    n_x1 = int(max(1, min(img_np.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img_np.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img_np.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img_np.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])

                # 存储变化后的图片
                image_aug = seq_det.augment_images([img_np])[0]

                # 确定保存的扩展名
                if SAVE_FORMAT == 'keep':
                    save_ext = original_ext
                else:
                    save_ext = SAVE_FORMAT

                aug_img_name = base_name + str("_%06d" % (epoch + 1)) + save_ext
                path = os.path.join(AUG_IMG_DIR, aug_img_name)

                # 使用Pillow保存，确保格式正确[1,6](@ref)
                Image.fromarray(image_aug).save(path)
                # 如果需要绘制边界框，可以使用以下代码（但保存的图片会包含框）
                # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                # Image.fromarray(image_auged).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, base_name, new_bndbox_list, AUG_XML_DIR, epoch + 1, base_name,
                                           original_ext, SAVE_FORMAT)
                print("Augmented:", aug_img_name)
                new_bndbox_list = []