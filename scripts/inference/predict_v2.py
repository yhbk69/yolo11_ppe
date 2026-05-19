"""
两阶段 PPE 检测 - 推理脚本 v2
阶段1: YOLO 检测 Person + PPE 正类
阶段2: 逻辑判断未佩戴装备的人 → 生成 no_xx 警告
"""
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# 类别定义（v2 模型输出的类别）
CLASS_NAMES = {
    0: 'helmet',
    1: 'gloves',
    2: 'vest',
    3: 'boots',
    4: 'goggles',
    5: 'Person',
}

# PPE 类别到 no_xx 的映射
PPE_TO_NO_XX = {
    'helmet': 'no_helmet',
    'goggles': 'no_goggle',
    'gloves': 'no_gloves',
    'boots': 'no_boots',
    'vest': 'no_vest',  # 可选
}

# 颜色配置
COLORS = {
    'Person': (255, 255, 255),      # 白色
    'helmet': (0, 255, 0),          # 绿色
    'gloves': (0, 255, 255),        # 黄色
    'vest': (255, 0, 255),          # 紫色
    'boots': (0, 165, 255),         # 橙色
    'goggles': (255, 255, 0),       # 青色
    'no_helmet': (0, 0, 255),       # 红色
    'no_goggle': (0, 0, 255),
    'no_gloves': (0, 0, 255),
    'no_boots': (0, 0, 255),
    'no_vest': (0, 0, 255),
}


def is_inside(outer_box, inner_box, iou_thresh=0.3):
    """
    判断 inner_box 的中心点是否在 outer_box 内，且有一定 IoU
    box 格式: [x1, y1, x2, y2]
    """
    # 计算中心点
    cx = (inner_box[0] + inner_box[2]) / 2
    cy = (inner_box[1] + inner_box[3]) / 2
    
    # 中心点在 outer 内
    if not (outer_box[0] <= cx <= outer_box[2] and outer_box[1] <= cy <= outer_box[3]):
        return False
    
    # 计算 IoU
    x1 = max(outer_box[0], inner_box[0])
    y1 = max(outer_box[1], inner_box[1])
    x2 = min(outer_box[2], inner_box[2])
    y2 = min(outer_box[3], inner_box[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    
    if inner_area == 0:
        return False
    
    return inter / inner_area >= iou_thresh


def check_ppe_for_person(person_box, ppe_detections, ppe_class_names):
    """
    检查一个 Person 框内缺失哪些 PPE
    返回缺失的 PPE 列表
    """
    missing_ppe = []
    
    for ppe_name in ['helmet', 'gloves', 'vest', 'boots', 'goggles']:
        # 检查是否有对应 PPE 在这个 Person 框内
        has_ppe = False
        for det in ppe_detections:
            cls_name = ppe_class_names.get(det['cls'], '')
            if cls_name == ppe_name:
                if is_inside(person_box, det['box'], iou_thresh=0.2):
                    has_ppe = True
                    break
        
        if not has_ppe:
            missing_ppe.append(ppe_name)
    
    return missing_ppe


def estimate_no_xx_box(person_box, ppe_name):
    """
    根据缺失的 PPE 类型，估算 no_xx 框的位置
    """
    x1, y1, x2, y2 = person_box
    
    if ppe_name == 'helmet':
        # 头盔在头部，取 Person 框上部 30%
        h = y2 - y1
        return [x1 + (x2-x1)*0.2, y1, x2 - (x2-x1)*0.2, y1 + h*0.35]
    elif ppe_name == 'goggles':
        # 护目镜在眼睛位置，取 Person 框上部 20-35%
        h = y2 - y1
        return [x1 + (x2-x1)*0.25, y1 + h*0.15, x2 - (x2-x1)*0.25, y1 + h*0.35]
    elif ppe_name == 'gloves':
        # 手套在手部，取 Person 框中部两侧
        w = x2 - x1
        h = y2 - y1
        return [x1, y1 + h*0.3, x1 + w*0.3, y1 + h*0.6]
    elif ppe_name == 'boots':
        # 靴子在脚部，取 Person 框底部
        h = y2 - y1
        return [x1 + (x2-x1)*0.2, y1 + h*0.8, x2 - (x2-x1)*0.2, y2]
    elif ppe_name == 'vest':
        # 背心在躯干，取 Person 框中部
        h = y2 - y1
        return [x1 + (x2-x1)*0.15, y1 + h*0.25, x2 - (x2-x1)*0.15, y1 + h*0.65]
    
    return person_box


def draw_detections(img, detections, class_names, show_conf=True):
    """绘制检测结果"""
    for det in detections:
        box = det['box']
        cls = det['cls']
        conf = det['conf']
        cls_name = class_names.get(cls, f'class_{cls}')
        
        color = COLORS.get(cls_name, (128, 128, 128))
        
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = cls_name
        if show_conf:
            label += f' {conf:.2f}'
        
        # 绘制标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_image(model, img_path, conf_thresh=0.25, iou_thresh=0.45):
    """
    处理单张图片：两阶段检测
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    # 阶段1: YOLO 检测
    results = model(str(img_path), conf=conf_thresh, iou=iou_thresh, verbose=False)
    result = results[0]
    
    # 解析检测结果
    ppe_detections = []  # PPE 检测
    person_detections = []  # Person 检测
    
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = CLASS_NAMES.get(cls, '')
            
            det = {
                'box': [x1, y1, x2, y2],
                'cls': cls,
                'conf': conf,
                'cls_name': cls_name,
            }
            
            if cls_name == 'Person':
                person_detections.append(det)
            else:
                ppe_detections.append(det)
    
    # 阶段2: 逻辑判断 no_xx
    no_xx_detections = []
    
    for person_det in person_detections:
        person_box = person_det['box']
        missing_ppe = check_ppe_for_person(person_box, ppe_detections, CLASS_NAMES)
        
        for ppe_name in missing_ppe:
            no_xx_name = PPE_TO_NO_XX.get(ppe_name)
            if no_xx_name:
                no_xx_box = estimate_no_xx_box(person_box, ppe_name)
                no_xx_detections.append({
                    'box': no_xx_box,
                    'cls_name': no_xx_name,
                    'conf': 1.0,  # 逻辑推理结果，置信度设为 1.0
                })
    
    # 绘制结果
    # 1. 绘制 PPE 正类（半透明）
    for det in ppe_detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(det['cls_name'], (128, 128, 128))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # 2. 绘制 Person
    for det in person_detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        color = COLORS['Person']
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = 'Person'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 3. 绘制 no_xx 警告（红色粗框）
    for det in no_xx_detections:
        box = det['box']
        x1, y1, x2, y2 = map(int, box)
        color = COLORS.get(det['cls_name'], (0, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        
        label = det['cls_name']
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
        cv2.putText(img, label, (x1+3, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img, {
        'persons': len(person_detections),
        'ppe': len(ppe_detections),
        'no_xx': len(no_xx_detections),
        'details': {
            'ppe_detections': ppe_detections,
            'person_detections': person_detections,
            'no_xx_detections': no_xx_detections,
        }
    }


def main():
    parser = argparse.ArgumentParser(description='两阶段 PPE 检测推理')
    parser.add_argument('--model', type=str, 
                        default='runs/detect/train_v2_yolo11n/weights/best.pt',
                        help='模型权重路径')
    parser.add_argument('--source', type=str,
                        default='datasets/construction-ppe/images/val',
                        help='输入图片/目录')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU 阈值')
    parser.add_argument('--save-dir', type=str, default='runs/detect/predict_v2',
                        help='结果保存目录')
    
    args = parser.parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model}")
    model = YOLO(args.model)
    
    # 准备输出目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取输入文件列表
    source = Path(args.source)
    if source.is_dir():
        files = sorted(source.glob('*.jpg')) + sorted(source.glob('*.png'))
    else:
        files = [source]
    
    print(f"处理 {len(files)} 个文件...")
    
    # 统计
    total_persons = 0
    total_ppe = 0
    total_no_xx = 0
    
    for i, img_path in enumerate(files):
        result = process_image(model, img_path, conf_thresh=args.conf, iou_thresh=args.iou)
        if result is None:
            continue
        
        img, stats = result
        total_persons += stats['persons']
        total_ppe += stats['ppe']
        total_no_xx += stats['no_xx']
        
        # 保存结果
        out_path = save_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        
        # 打印统计
        print(f"[{i+1}/{len(files)}] {img_path.name}: "
              f"Person={stats['persons']}, PPE={stats['ppe']}, no_xx={stats['no_xx']}")
        
        # 保存 JSON 详情（可选）
        import json
        with open(save_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n" + "=" * 50)
    print("处理完成！")
    print(f"总 Person: {total_persons}")
    print(f"总 PPE 检测: {total_ppe}")
    print(f"总 no_xx 警告: {total_no_xx}")
    print(f"结果保存在: {save_dir}")
    print("=" * 50)


if __name__ == '__main__':
    main()
