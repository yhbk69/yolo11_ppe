from ultralytics import YOLO

from multiprocessing import freeze_support

def main():
    # Load a model
    # model = YOLO("yolo11x.pt")
    model = YOLO("runs/detect/train10/weights/last.pt")
    train_results = model.train(
        resume=True,  # 关键参数：设置为 True 以恢复训练
        # 其他参数（如 epochs, imgsz, batch 等）无需再次指定，
        # 它们会从 last.pt 文件中自动恢复。
        # 如果你需要修改，可以在这里重新指定。
    )

    # Train the model
    # train_results = model.train(
    #     data=r"D:\dltt\Python\YOLOV11\datasets\construction-ppe\data.yaml",  # path to dataset YAML
    #     epochs=1000,  # number of training epochs
    #     imgsz=640,  # training image size
    #     batch=4,
    #     # accumulation=4,
    #     device="0"  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    # )


if __name__ == '__main__':
    freeze_support()  # Windows多进程必需
    main()
