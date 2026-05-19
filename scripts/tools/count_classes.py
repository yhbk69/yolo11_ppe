import os
import yaml

data_path = r'D:\dltt\Python\YOLOV11\datasets\construction-ppe\data.yaml'
with open(data_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

base = cfg['path']
for split in ['train', 'val', 'test']:
    label_dir = os.path.join(base, f'labels/{split}')
    counts = {}
    for fn in os.listdir(label_dir):
        if fn.endswith('.txt'):
            with open(os.path.join(label_dir, fn), 'r') as lf:
                for line in lf:
                    cls = int(line.split()[0])
                    counts[cls] = counts.get(cls, 0) + 1
    
    print(f'=== {split} 集类别分布 ===')
    for i in range(11):
        name = cfg['names'][i]
        cnt = counts.get(i, 0)
        print(f'  {i:2d}: {name:15s} -> {cnt:5d} 个实例')
    print()
