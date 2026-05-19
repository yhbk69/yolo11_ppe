import os
import xml.etree.ElementTree as ET

# 1. 定义类别列表（需根据实际修改）
classes = ["gap", "scratch", "stain","thread_defect","wrinkle"]  # 替换为你的类别名称


# 2. 转换函数（处理坐标归一化）
def convert(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


# 3. 核心转换逻辑
def convert_annotation(xml_path, output_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 获取图像尺寸
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 创建输出路径
        txt_name = os.path.basename(xml_path).replace('.xml', '.txt')
        txt_path = os.path.join(output_dir, txt_name)

        with open(txt_path, 'w') as f:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                # 跳过未定义类别
                if cls not in classes:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (
                    float(xmlbox.find('xmin').text),
                    float(xmlbox.find('xmax').text),
                    float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text)
                )
                bb = convert((w, h), b)  # 坐标归一化
                f.write(f"{cls_id} {' '.join(str(x) for x in bb)}\n")
        return True
    except Exception as e:
        print(f"转换失败: {xml_path}, 错误: {str(e)}")
        return False


# 4. 遍历所有子文件夹
def process_all_xmls(root_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                convert_annotation(xml_path, output_dir)
                print(f"已转换: {xml_path}")


# 5. 执行转换
if __name__ == "__main__":
    xml_root = r"E:\code\dataset\seat2\xml"  # XML根目录
    output_dir = r"E:\code\dataset\seat2\labels"  # 输出目录（建议新建）
    process_all_xmls(xml_root, output_dir)
    print("全部转换完成！")