"""
分析 no_xx 类标注问题
检查标注框大小、与正类的关系等
"""
import os
import yaml
import cv2
import numpy as np

data_path = r'D:\dltt\Python\YOLOV11\datasets\construction-ppe\data.yaml'
with open(data_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

base = cfg['path']
names = cfg['names']

# 类别映射
no_to_pos = {
    7: 0,   # no_helmet -> helmet
    8: 4,   # no_goggle -> goggles
    9: 1,   # no_gloves -> gloves
    10: 3,  # no_boots -> boots
}

def analyze_split(split='val', num_samples=10):
    """分析一个 split 的标注"""
    label_dir = os.path.join(base, f'labels/{split}')
    img_dir = os.path.join(base, f'images/{split}')
    
    # 找到包含 no_xx 的图片
    files_with_no = []
    for fn in sorted(os.listdir(label_dir)):
        if not fn.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fn), 'r') as f:
            classes = [int(line.split()[0]) for line in f if line.strip()]
        if any(c in [7,8,9,10] for c in classes):
            files_with_no.append(fn)
    
    print(f'\n=== {split} 集：共 {len(files_with_no)} 张图片包含 no_xx 类 ===')
    
    for i, fn in enumerate(files_with_no[:num_samples]):
        print(f'\n--- {fn} ---')
        label_path = os.path.join(label_dir, fn)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        img_path = os.path.join(img_dir, fn.replace('.txt', '.jpg'))
        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')
        
        h, w = 640, 640  # 默认
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:5]]
            
            # 转换为像素坐标
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            area = (x2-x1) * (y2-y1)
            
            cls_name = names.get(cls_id, f'class_{cls_id}')
            print(f'  {cls_name:15s}: box=({x1},{y1},{x2},{y2}) w={x2-x1:4d} h={y2-y1:4d} area={area:6d}')

# 分析 val 集
analyze_split('val', num_samples=20)
