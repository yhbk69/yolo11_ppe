"""
分析 Person 与 PPE 装备的空间关系
判断是否适合做逻辑推理判断
"""
import os
import yaml
import json

data_path = r'D:\dltt\Python\YOLOV11\datasets\construction-ppe\data.yaml'
with open(data_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

base = cfg['path']
names = cfg['names']

# 类别定义
person_cls = 6
ppe_cls = {
    0: 'helmet',
    1: 'gloves',
    2: 'vest',
    3: 'boots',
    4: 'goggles',
}
no_xx_cls = {
    7: 'no_helmet',
    8: 'no_goggle',
    9: 'no_gloves',
    10: 'no_boots',
}

def box_center(b):
    """计算框的中心点 (cx, cy)"""
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

def box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])

def iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0

def contains(outer, inner):
    """outer 是否包含 inner 的中心点"""
    cx, cy = box_center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]

def analyze_file(label_path):
    """分析单个标注文件"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    boxes = {}  # cls_id -> [box, ...]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
        x1, y1 = cx - bw/2, cy - bh/2
        x2, y2 = cx + bw/2, cy + bh/2
        boxes.setdefault(cls_id, []).append([x1, y1, x2, y2])
    
    return boxes

# 分析 val 集
label_dir = os.path.join(base, 'labels/val')
stats = {
    'total_files': 0,
    'person_only': 0,
    'person_with_ppe': 0,
    'person_with_no_xx': 0,
    'no_xx_overlap_person': 0,
    'ppe_inside_person': 0,
}

# 抽样分析
samples = []
for fn in sorted(os.listdir(label_dir))[:200]:
    if not fn.endswith('.txt'):
        continue
    
    boxes = analyze_file(os.path.join(label_dir, fn))
    stats['total_files'] += 1
    
    person_boxes = boxes.get(person_cls, [])
    ppe_boxes = {k: v for k, v in boxes.items() if k in ppe_cls}
    no_xx_boxes = {k: v for k, v in boxes.items() if k in no_xx_cls}
    
    # 统计
    if person_boxes:
        if ppe_boxes:
            stats['person_with_ppe'] += 1
        if no_xx_boxes:
            stats['person_with_no_xx'] += 1
    
    # 检查 PPE 是否在 Person 框内
    for p_box in person_boxes:
        for ppe_id, ppe_list in ppe_boxes.items():
            for pb in ppe_list:
                if contains(p_box, pb):
                    stats['ppe_inside_person'] += 1
    
    # 检查 no_xx 与 Person 的关系
    for p_box in person_boxes:
        for no_id, no_list in no_xx_boxes.items():
            for nb in no_list:
                if contains(p_box, nb):
                    stats['no_xx_overlap_person'] += 1
    
    # 保存样本
    if person_boxes and (ppe_boxes or no_xx_boxes):
        samples.append({
            'file': fn,
            'person_count': len(person_boxes),
            'ppe': {ppe_cls[k]: len(v) for k, v in ppe_boxes.items()},
            'no_xx': {no_xx_cls[k]: len(v) for k, v in no_xx_boxes.items()},
        })

print("=" * 60)
print("Person 与 PPE 关系分析")
print("=" * 60)
print(f"总文件数: {stats['total_files']}")
print(f"包含 Person: {stats['person_with_ppe'] + stats['person_with_no_xx']}")
print(f"Person + 有 PPE: {stats['person_with_ppe']}")
print(f"Person + 有 no_xx: {stats['person_with_no_xx']}")
print(f"PPE 框在 Person 内: {stats['ppe_inside_person']}")
print(f"no_xx 框在 Person 内: {stats['no_xx_overlap_person']}")

print("\n" + "=" * 60)
print("抽样文件分析 (前 20 个)")
print("=" * 60)
for s in samples[:20]:
    ppe_str = ', '.join([f"{k}:{v}" for k, v in s['ppe'].items()])
    no_str = ', '.join([f"{k}:{v}" for k, v in s['no_xx'].items()])
    print(f"{s['file']}: Person={s['person_count']}, PPE=[{ppe_str}], no_xx=[{no_str}]")
