from ultralytics import YOLO
from multiprocessing import freeze_support
from pathlib import Path
import time


def main():
    # 数据集配置
    data_path = r"D:\dltt\Python\YOLOV11\datasets\construction-ppe\data.yaml"
    
    # 训练参数（每个参数的作用和可选值见注释）
    train_config = {
        # ====== 基础配置 ======
        'data': data_path,        # 数据集 YAML 配置文件路径，定义训练/验证/测试集和类别信息
        
        'epochs': 1000,           # 训练总轮次
                                  # 小数据集: 100-300 | 大数据集: 300-1000+
                                  # 配合 patience 可实现早停（提前终止）
        
        'imgsz': 640,             # 训练图像尺寸（像素）
                                  # 常用值: 320(快)/640(标准)/1280(高精度)
                                  # 必须是 32 的倍数
        
        'batch': 4,               # 每批处理的图像数量
                                  # 根据 GPU 显存调整: 显存大→16/32/64，显存小→2/4/8
                                  # 也可设为 -1 或 'auto' 自动检测
        
        'device': '0',            # 计算设备
                                  # '0' 使用第一个GPU | '0,1,2' 多卡训练 | 'cpu' 仅CPU
        
        # ====== 优化器与学习率 ======
        'optimizer': 'AdamW',     # 优化器类型
                                  # 'SGD'(经典稳定) | 'Adam'(收敛快) | 'AdamW'(带权重衰减，推荐)
                                  # 'auto' 自动选择
        
        'lr0': 0.01,              # 初始学习率
                                  # SGD 推荐: 0.01 | Adam/AdamW 推荐: 0.001-0.01
                                  # 过大会导致震荡，过小收敛慢
        
        'lrf': 0.01,              # 最终学习率比例（最终学习率 = lr0 × lrf）
                                  # 范围: 0.01-0.1，越小末期学习率越低
        
        'weight_decay': 0.0005,   # 权重衰减系数（L2 正则化，防止过拟合）
                                  # 常用范围: 0.0001-0.001
                                  # AdamW 中此参数直接控制衰减，SGD 中作为额外正则
        
        'cos_lr': True,           # 是否使用余弦退火学习率调度
                                  # True: 学习率按余弦曲线逐渐降低 | False: 线性降低
                                  # 通常 True 收敛更好
        
        # ====== 数据增强 ======
        'mosaic': 1.0,            # 马赛克增强概率（4张图拼接成1张）
                                  # 范围: 0.0-1.0 | 1.0 表示始终启用 | 0.0 禁用
                                  # 对检测任务非常有效，提升小目标检测
        
        'mixup': 0.2,             # MixUp 增强概率（两张图线性叠加）
                                  # 范围: 0.0-1.0 | 推荐: 0.1-0.3
                                  # 提升泛化能力，防止过拟合
        
        'hsv_h': 0.015,           # 色调（Hue）增强幅度
                                  # 范围: 0.0-0.5 | 值越大颜色变化越明显
                                  # 对光照变化敏感的场景有用
        
        'hsv_s': 0.7,             # 饱和度（Saturation）增强幅度
                                  # 范围: 0.0-1.0 | 推荐: 0.5-0.8
                                  # 提升对不同颜色饱和度的鲁棒性
        
        'hsv_v': 0.4,             # 亮度（Value）增强幅度
                                  # 范围: 0.0-1.0 | 推荐: 0.3-0.5
                                  # 模拟不同光照条件
        
        'fliplr': 0.5,            # 水平翻转概率
                                  # 范围: 0.0-1.0 | 0.5 表示 50% 概率翻转
                                  # 对左右对称目标有效，目标有方向性时应降低或禁用
        
        'close_mosaic': 10,       # 最后 N 轮禁用马赛克增强
                                  # 防止末期训练不稳定 | 0 表示不关闭
                                  # 通常 5-15 轮
        
        # ====== 训练控制 ======
        'patience': 20,           # 早停机制等待轮次（验证指标不改善时提前终止）
                                  # 设为 0 禁用早停 | 50+ 更保守 | 10-30 较激进
                                  # 配合 resume=True 可自动恢复
        
        'workers': 8,             # 数据加载线程数（并行读取和预处理图像）
                                  # 根据 CPU 核心数调整 | Windows 推荐 2-8 | Linux 可更高
                                  # 过大可能内存溢出
        
        'rect': True,             # 矩形训练（减少填充，保持原始宽高比）
                                  # True: 减少计算量，加速训练 | False: 强制正方形
                                  # 对长宽比变化大的数据集效果好
        
        'cache': 'ram',           # 数据缓存策略（加速数据加载）
                                  # 'ram': 缓存到内存（最快，需要足够内存）
                                  # 'disk': 缓存到磁盘（较慢，省内存）
                                  # False 或 None: 不缓存（最慢，最省资源）
        
        'amp': True,              # 自动混合精度训练（FP16 + FP32 混合）
                                  # True: 显存减半，速度提升 20-50% | False: 全精度
                                  # 现代 GPU 强烈推荐开启
        
        # ====== 模型保存与日志 ======
        'project': '',        # 项目保存目录（所有实验的父目录）
                                  # 可设为绝对路径如 r'D:\my_experiments'
        
        'name': 'exp',            # 实验名称（在 project 下创建子目录）
                                  # 如 'train_yolo11n' → runs/detect/train_yolo11n/
                                  # 设 None 自动生成 exp/ exp2/ ...
        
        'exist_ok': False,        # 是否覆盖同名实验目录
                                  # True: 覆盖旧结果 | False: 报错或自动创建新编号
        
        'plots': True,            # 是否保存训练曲线图和可视化结果
                                  # True: 生成 PR曲线/混淆矩阵等 | False: 不生成
    }
    
    # 要训练的模型列表（模型权重路径 -> 实验名称）
    models_to_train = [
        ('weights/yolo11/yolo11n.pt', 'train_yolo11n'),
        ('weights/yolo11/yolo11s.pt', 'train_yolo11s'),
        ('weights/yolo11/yolo11m.pt', 'train_yolo11m'),
        ('weights/yolo11/yolo11l.pt', 'train_yolo11l'),
       # ('weights/yolo11/yolo11x.pt', 'train_yolo11x'),
    ]
    
    total_models = len(models_to_train)
    
    print("=" * 60)
    print(f"开始批量训练 {total_models} 个模型")
    print(f"数据集: {data_path}")
    print(f"Epochs: {train_config['epochs']}, Batch: {train_config['batch']}")
    print("=" * 60)
    
    for idx, (weight_path, exp_name) in enumerate(models_to_train, 1):
        model_name = Path(weight_path).stem
        print(f"\n{'='*60}")
        print(f"[{idx}/{total_models}] 开始训练: {model_name}")
        print(f"实验名称: {exp_name}")
        print(f"{'='*60}")
        
        # 检查权重文件是否存在
        if not Path(weight_path).exists():
            print(f"⚠️  跳过：权重文件不存在 - {weight_path}")
            continue
        
        try:
            # 加载模型
            model = YOLO(weight_path)
            
            # 更新实验名称
            train_config['project'] = ''
            train_config['name'] = exp_name
            train_config['exist_ok'] = False
            
            # 开始训练
            start_time = time.time()
            results = model.train(**train_config)
            elapsed = time.time() - start_time
            
            print(f"\n✅ {model_name} 训练完成！耗时: {elapsed/60:.1f} 分钟")
            print(f"结果保存在: runs/detect/{exp_name}/")
            
            # 每个模型之间等待一下（可选，避免 GPU 过热）
            if idx < total_models:
                print("\n⏳ 等待 10 秒后开始下一个训练...")
                time.sleep(10)
                
        except Exception as e:
            print(f"\n❌ {model_name} 训练失败: {str(e)}")
            print("继续下一个模型...")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 所有模型训练完成！")
    print("=" * 60)
    
    # 打印训练结果汇总
    print("\n训练结果汇总:")
    for _, exp_name in models_to_train:
        best_weights = Path(f"runs/detect/{exp_name}/weights/best.pt")
        last_weights = Path(f"runs/detect/{exp_name}/weights/last.pt")
        
        status = "✅" if best_weights.exists() else "❌"
        print(f"  {status} {exp_name}: best.pt {'存在' if best_weights.exists() else '未找到'}")


if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()
