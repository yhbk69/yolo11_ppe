"""
YOLOv11 摄像头实时检测脚本
使用方法: python detect_cam.py
"""
import sys
import cv2
from pathlib import Path
from ultralytics import YOLO


def main():
    # 权重路径
    weight_path = Path(r"D:\dltt\Python\YOLOV11\runs\detect\train8\weights\best.pt")

    print("=" * 50)
    print("YOLOv11 摄像头实时检测")
    print("=" * 50)
    print(f"权重: {weight_path}")
    print("=" * 50)

    try:
        # 加载模型
        print("\n正在加载模型...")
        model = YOLO(weight_path)
        print("模型加载成功!")

        # 打开摄像头
        print("\n正在打开摄像头...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("错误: 无法打开摄像头")
            sys.exit(1)

        print("摄像头已打开，关闭窗口即可停止")
        print("=" * 50)

        # 创建显示窗口
        window_name = "YOLOv11 Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 手动循环读取帧并预测
        while True:
            # 检查窗口是否关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("\n窗口已关闭，停止检测!")
                break

            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("\n无法读取摄像头帧")
                break

            # 进行预测
            results = model.predict(
                source=frame,
                conf=0.5,
                show=False,
                verbose=False
            )

            # 获取带标注的帧并显示
            annotated_frame = results[0].plot()
            cv2.imshow(window_name, annotated_frame)

            # 等待1ms，处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # 清理资源
        cap.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出!")
        if 'cap' in dir():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        if 'cap' in dir():
            cap.release()
        cv2.destroyAllWindows()
        sys.exit(1)

    print("\n检测结束!")


if __name__ == "__main__":
    main()
