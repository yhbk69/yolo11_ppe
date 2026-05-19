# import os
#
#
# def generate_directory_structure(root_dir, output_file="directory_structure.txt"):
#     """
#     生成目录结构树，忽略图片文件，并保存到txt文件
#
#     Parameters:
#     root_dir (str): 要遍历的根目录路径
#     output_file (str): 输出的文本文件名
#     """
#     # 图片文件扩展名列表
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp',
#                         '.tiff', '.webp', '.ico', '.svg', '.heic']
#
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"目录结构: {root_dir}\n")
#             f.write("=" * 50 + "\n\n")
#
#             for root, dirs, files in os.walk(root_dir):
#                 # 计算当前目录相对于根目录的层级
#                 level = root.replace(root_dir, '').count(os.sep)
#                 indent = '    ' * level
#
#                 # 写入当前目录
#                 dir_name = os.path.basename(root) if root != root_dir else os.path.basename(root_dir)
#                 f.write(f"{indent}{dir_name}/\n")
#
#                 # 写入文件，过滤掉图片文件
#                 sub_indent = '    ' * (level + 1)
#                 for file in files:
#                     # 获取文件扩展名并检查是否为图片
#                     file_extension = os.path.splitext(file)[1].lower()
#                     if file_extension not in image_extensions:
#                         f.write(f"{sub_indent}{file}\n")
#
#             f.write(
#                 f"\n生成时间: {os.path.basename(root_dir)} - {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
#
#         print(f"目录结构已成功生成并保存到: {output_file}")
#
#     except Exception as e:
#         print(f"生成目录结构时出错: {e}")
#
#
# def generate_directory_tree(root_dir, output_file="directory_tree.txt"):
#     """
#     使用树状结构生成目录，更直观的显示方式
#     """
#     image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp',
#                         '.tiff', '.webp', '.ico', '.svg', '.heic']
#
#     try:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(f"{os.path.basename(root_dir)}/\n")
#
#             for root, dirs, files in os.walk(root_dir):
#                 # 计算缩进
#                 level = root.replace(root_dir, '').count(os.sep)
#                 indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
#
#                 # 如果是子目录，写入目录名
#                 if level > 0:
#                     dir_name = os.path.basename(root)
#                     f.write(f"{indent}{dir_name}/\n")
#
#                 # 写入文件
#                 file_indent = '│   ' * level + '├── '
#                 for i, file in enumerate(sorted(files)):
#                     file_extension = os.path.splitext(file)[1].lower()
#                     if file_extension not in image_extensions:
#                         # 如果是最后一个文件，使用不同的前缀
#                         if i == len([f for f in files if os.path.splitext(f)[1].lower() not in image_extensions]) - 1:
#                             file_indent = '│   ' * level + '└── '
#                         f.write(f"{file_indent}{file}\n")
#
#         print(f"树状目录结构已生成: {output_file}")
#
#     except Exception as e:
#         print(f"生成树状结构时出错: {e}")
#
#
# if __name__ == "__main__":
#     # 指定要遍历的目录
#     target_directory = r"E:\code\PycharmProjects\YOLOV11"
#     output_filename = "YOLOV11_directory_structure.txt"
#     tree_output_filename = "YOLOV11_directory_tree.txt"
#
#     # 检查目录是否存在
#     if not os.path.exists(target_directory):
#         print(f"错误: 目录 {target_directory} 不存在!")
#     else:
#         # 生成普通目录结构
#         generate_directory_structure(target_directory, output_filename)
#
#         # 生成树状目录结构（更直观）
#         generate_directory_tree(target_directory, tree_output_filename)
#
#         print("\n两种格式的目录结构已生成:")
#         print(f"1. 简单结构: {output_filename}")
#         print(f"2. 树状结构: {tree_output_filename}")


from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f"{m._get_name()}.onnx"
torch.onnx.export(m, x, f)
os.system(f"onnxslim {f} {f} && open {f}")  # pip install onnxslim



