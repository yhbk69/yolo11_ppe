"""
两阶段 PPE 检测 - 训练脚本 v2
阶段1: 训练 YOLO 检测 Person + PPE 正类（helmet/gloves/vest/boots/goggles）
阶段2: 推理时逻辑判断未佩戴装备的人 → 生成 no_xx 警告
"""
from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path
import time


def main():
    # 数据集配置
    data_path = r"D:\dltt\Python\YOLOV11\datasets\construction-ppe-v2\data.yaml"
    
    # 训练参数（沿用 train3.py 配置）
    train_config = {
        # ====== 基础配置 ======
        'data': data_path,        # 数据集 YAML 配置文件路径
        
        'epochs': 1000,           # 训练总轮次
        'imgsz': 640,             # 训练图像尺寸（像素）
        'batch': 4,               # 每批处理的图像数量
        'device': '0',            # 计算设备
        
        # ====== 优化器与学习率 ======
        'optimizer': 'AdamW',     # 优化器类型
        'lr0': 0.01,              # 初始学习率
        'lrf': 0.01,              # 最终学习率比例
        'weight_decay': 0.0005,   # 权重衰减系数
        'cos_lr': True,           # 余弦退火学习率调度
        
        # ====== 数据增强 ======
        'mosaic': 1.0,            # 马赛克增强概率
        'mixup': 0.2,             # MixUp 增强概率
        'hsv_h': 0.015,           # 色调增强幅度
        'hsv_s': 0.7,             # 饱和度增强幅度
        'hsv_v': 0.4,             # 亮度增强幅度
        'fliplr': 0.5,            # 水平翻转概率
        'close_mosaic': 10,       # 最后 N 轮禁用马赛克
        
        # ====== 训练控制 ======
        'patience': 20,           # 早停等待轮次
        'workers': 8,             # 数据加载线程数
        'rect': True,             # 矩形训练
        'cache': 'ram',           # 数据缓存策略
        'amp': True,              # 自动混合精度训练
        
        # ====== 模型保存与日志 ======
        'project': 'runs/detect',
        'name': 'exp',            # 占位符，循环中更新
        'exist_ok': False,
        'plots': True,
    }
    
    # 要训练的模型列表（模型权重路径 -> 实验名称）
    models_to_train = [
        ('weights/yolo11/yolo11n.pt', 'train_v2_yolo11n'),
        ('weights/yolo11/yolo11s.pt', 'train_v2_yolo11s'),
        ('weights/yolo11/yolo11m.pt', 'train_v2_yolo11m'),
        ('weights/yolo11/yolo11l.pt', 'train_v2_yolo11l'),
        ('weights/yolo11/yolo11x.pt', 'train_v2_yolo11x'),
    ]
    
    total_models = len(models_to_train)
    
    print("=" * 60)
    print(f"两阶段 PPE 检测 - 开始训练 {total_models} 个模型")
    print(f"数据集: {data_path}")
    print(f"类别: helmet, gloves, vest, boots, goggles, Person")
    print(f"Epochs: {train_config['epochs']}, Batch: {train_config['batch']}")
    print("=" * 60)
    
    for idx, (weight_path, exp_name) in enumerate(models_to_train, 1):
        model_name = Path(weight_path).stem
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_models}] 开始训练: {model_name}")
        print(f"实验名称: {exp_name}")
        print(f"{'='*60}")
        
        if not Path(weight_path).exists():
            print(f"️  跳过：权重文件不存在 - {weight_path}")
            continue
        
        try:
            model = YOLO(weight_path)
            
            train_config['name'] = exp_name
            
            start_time = time.time()
            results = model.train(**train_config)
            elapsed = time.time() - start_time
            
            print(f"\n✅ {model_name} 训练完成！耗时: {elapsed/60:.1f} 分钟")
            print(f"结果保存在: runs/detect/{exp_name}/")
            
            if idx < total_models:
                print("\n⏳ 等待 10 秒后开始下一个训练...")
                time.sleep(10)
                
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
            print("继续下一个模型...")
            continue
    
    print("\n" + "=" * 60)
    print(" 所有模型训练完成！")
    print("=" * 60)
    
    print("\n训练结果汇总:")
    for _, exp_name in models_to_train:
        best_weights = Path(f"runs/detect/{exp_name}/weights/best.pt")
        status = "✅" if best_weights.exists() else "❌"
        print(f"  {status} {exp_name}: best.pt {'存在' if best_weights.exists() else '未找到'}")
    
    print("\n💡 下一步：使用 scripts/inference/predict_v2.py 进行两阶段推理")


if __name__ == '__main__':
    freeze_support()
    main()
