from ultralytics import YOLO
from multiprocessing import freeze_support


def main():
    # 加载预训练模型 (支持YOLOv11多种架构)
    model = YOLO("yolo11x.pt")  # 可替换为yolov11s/m/l/x等不同尺寸模型

    # 训练模型（包含增强参数）
    train_results = model.train(
        data=r"E:\code\dataset\seat3\yolo\data.yaml",  # 数据集配置文件路径
        epochs=2000,  # 训练轮次（大数据集建议300-500轮）
        imgsz=640,  # 输入图像尺寸（高精度检测推荐640-1280）
        batch=4,  # 批次大小（根据GPU显存调整）
        device="0",  # 使用GPU设备（多卡训练用逗号分隔，如"0,1"）

        # ====== 优化器与学习率 ======
        optimizer="AdamW",  # 优化器选择（SGD/Adam/AdamW）
        lr0=0.01,  # 初始学习率（SGD=0.01, Adam=0.001）
        lrf=0.01,  # 最终学习率比例（lr0*lrf）
        weight_decay=0.0005,  # 权重衰减系数
        cos_lr=True,  # 启用余弦退火学习率调度

        # ====== 数据增强 ======
        augment=False,  # 启用自动数据增强
        mosaic=1.0,  # 马赛克增强概率
        mixup=0.2,  # MixUp增强系数
        hsv_h=0.015,  # 色调增强幅度
        hsv_s=0.7,  # 饱和度增强幅度
        hsv_v=0.4,  # 亮度增强幅度
        fliplr=0.5,  # 水平翻转概率
        close_mosaic=10,  # 最后10轮禁用马赛克增强

        # ====== 训练控制 ======
        patience=20,  # 早停机制等待轮次
        resume=False,  # 是否恢复训练
        workers=8,  # 数据加载线程数
        rect=True,  # 矩形训练（减少填充
        cache="ram",  # 数据缓存策略（ram/disk）
        amp=True,  # 自动混合精度训练[4]

        # ====== 模型保存与日志 ======
        project="runs/detect",  # 项目保存目录
        name="train41",  # 实验名称
        exist_ok=False,  # 覆盖同名实验
        save_period=10,  # 每10轮保存一次模型
        plots=True,  # 保存训练曲线图
    )


if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()