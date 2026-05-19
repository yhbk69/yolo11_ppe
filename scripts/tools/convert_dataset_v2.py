"""
数据集转换脚本 v1 → v2
删除 none(5) 和 no_xx(7,8,9,10) 类，保留 Person + PPE 正类
类别重新映射：
  原 0(helmet) → 新 0(helmet)
  原 1(gloves) → 新 1(gloves)
  原 2(vest)   → 新 2(vest)
  原 3(boots)  → 新 3(boots)
  原 4(goggles)→ 新 4(goggles)
  原 6(Person) → 新 5(Person)
"""
import os
import shutil

# 源目录
src_base = r'D:\dltt\Python\YOLOV11\datasets\construction-ppe'
# 目标目录
dst_base = r'D:\dltt\Python\YOLOV11\datasets\construction-ppe-v2'

# 要删除的类别
remove_classes = {5, 7, 8, 9, 10}  # none, no_helmet, no_goggle, no_gloves, no_boots

# 类别重新映射
cls_map = {
    0: 0,   # helmet
    1: 1,   # gloves
    2: 2,   # vest
    3: 3,   # boots
    4: 4,   # goggles
    6: 5,   # Person
}

def convert_label_file(src_path, dst_path):
    """转换单个标注文件"""
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        old_cls = int(parts[0])
        
        # 跳过要删除的类别
        if old_cls in remove_classes:
            continue
        
        # 重新映射类别 ID
        if old_cls in cls_map:
            new_cls = cls_map[old_cls]
            parts[0] = str(new_cls)
            new_lines.append(' '.join(parts))
    
    # 写入新文件（即使为空也创建，保持文件对应关系）
    with open(dst_path, 'w') as f:
        f.write('\n'.join(new_lines))
        if new_lines:
            f.write('\n')
    
    return len(new_lines)

def convert_split(split_name):
    """转换一个 split（train/val/test）"""
    src_img_dir = os.path.join(src_base, f'images/{split_name}')
    src_lbl_dir = os.path.join(src_base, f'labels/{split_name}')
    dst_img_dir = os.path.join(dst_base, f'images/{split_name}')
    dst_lbl_dir = os.path.join(dst_base, f'labels/{split_name}')
    
    total_files = 0
    total_boxes = 0
    skipped = 0
    
    for fn in os.listdir(src_lbl_dir):
        if not fn.endswith('.txt'):
            continue
        
        src_lbl = os.path.join(src_lbl_dir, fn)
        dst_lbl = os.path.join(dst_lbl_dir, fn)
        
        # 复制图片
        img_fn = fn.replace('.txt', '.jpg')
        src_img = os.path.join(src_img_dir, img_fn)
        dst_img = os.path.join(dst_img_dir, img_fn)
        if not os.path.exists(src_img):
            img_fn = fn.replace('.txt', '.png')
            src_img = os.path.join(src_img_dir, img_fn)
            dst_img = os.path.join(dst_img_dir, img_fn)
        
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # 转换标注
        box_count = convert_label_file(src_lbl, dst_lbl)
        total_boxes += box_count
        total_files += 1
        
        if box_count == 0:
            skipped += 1
    
    print(f'  {split_name}: {total_files} 文件, {total_boxes} 个框, {skipped} 个空文件')
    return total_files, total_boxes

print("=" * 50)
print("数据集转换 v1 → v2")
print("=" * 50)
print(f'源目录: {src_base}')
print(f'目标目录: {dst_base}')
print(f'删除类别: none(5), no_helmet(7), no_goggle(8), no_gloves(9), no_boots(10)')
print(f'保留类别: helmet(0), gloves(1), vest(2), boots(3), goggles(4), Person(5)')
print()

for split in ['train', 'val', 'test']:
    print(f'转换 {split} ...')
    convert_split(split)

print('\n转换完成！')
