import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def count_object_names(folder_path):
    """
    统计XML文件中<object>标签内<name>的类别及数量
    :param folder_path: 根文件夹路径
    :return: 类别计数字典
    """
    label_count = defaultdict(int)
    xml_count = 0
    error_count = 0

    # 递归遍历所有子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.xml'):
                xml_path = os.path.join(root, file)
                try:
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()

                    # 查找所有<object>标签
                    for obj in xml_root.findall('.//object'):
                        name_tag = obj.find('name')
                        if name_tag is not None and name_tag.text:
                            label = name_tag.text.strip()
                            label_count[label] += 1

                    xml_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"解析失败: {xml_path} - {str(e)}")

    # 输出统计结果
    print(f"已扫描 {xml_count} 个XML文件 | 错误文件: {error_count}")
    print("类别分布统计:")
    for label, count in sorted(label_count.items()):
        print(f"  {label}: {count}")
    print(f"总实例数: {sum(label_count.values())}")
    return label_count


if __name__ == "__main__":
    target_dir = r"E:\code\dataset\seat2\xml"
    count_object_names(target_dir)