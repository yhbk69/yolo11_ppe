from ultralytics import YOLO
import cv2
import time
from datetime import datetime


# 时间统计装饰器
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{func.__name__} 执行时间: {execution_time:.4f} 秒")
        return result

    return wrapper


class YOLODetector:
    def __init__(self, model_path='yolov11x.pt'):
        """初始化YOLO检测器"""
        self.model_load_time = time.perf_counter()
        self.model = YOLO(model_path)
        self.model_load_time = time.perf_counter() - self.model_load_time
        print(f"loading model time: {self.model_load_time:.4f} 秒")

        # 时间统计变量
        self.frame_count = 0
        self.total_preprocess_time = 0
        self.total_detection_time = 0
        self.total_postprocess_time = 0
        self.total_fps_time = 0

    @timeit
    def preprocess_frame(self, frame):
        """帧预处理（如果需要特殊处理）"""
        # 可以在这里添加自定义预处理逻辑
        return frame

    @timeit
    def detect_objects(self, frame):
        """执行目标检测"""
        results = self.model.predict(source=frame, verbose=False)
        return results

    @timeit
    def postprocess_results(self, results, frame):
        """后处理：绘制检测框"""
        annotated_frame = results[0].plot()
        return annotated_frame

    def calculate_fps(self, start_time, end_time):
        """计算帧率"""
        processing_time = end_time - start_time
        fps = 1 / processing_time if processing_time > 0 else 0
        return fps, processing_time

    def run_detection(self, source=0, output_file=None):
        """
        运行目标检测流程

        参数:
            source: 视频源（0=默认摄像头，文件路径=视频文件）
            output_file: 输出文件路径（可选）
        """
        # 打开视频源
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"无法打开视频源: {source}")
            return

        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"video size: {width}x{height}, FPS: {fps:.2f}")

        # 初始化视频写入器（如果指定了输出文件）
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        print("开始目标检测，按 'q' 键退出，按 'p' 键暂停...")

        paused = False
        start_time_total = time.perf_counter()

        while True:
            if not paused:
                # 记录帧开始时间
                frame_start_time = time.perf_counter()

                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    print("无法读取帧，退出...")
                    break

                self.frame_count += 1

                # 预处理
                preprocess_start = time.perf_counter()
                processed_frame = self.preprocess_frame(frame)
                preprocess_time = time.perf_counter() - preprocess_start
                self.total_preprocess_time += preprocess_time

                # 目标检测
                detection_start = time.perf_counter()
                results = self.detect_objects(processed_frame)
                detection_time = time.perf_counter() - detection_start
                self.total_detection_time += detection_time

                # 后处理
                postprocess_start = time.perf_counter()
                annotated_frame = self.postprocess_results(results, processed_frame)
                postprocess_time = time.perf_counter() - postprocess_start
                self.total_postprocess_time += postprocess_time

                # 计算帧率
                fps_calc_start = time.perf_counter()
                current_fps, processing_time = self.calculate_fps(frame_start_time, time.perf_counter())
                fps_time = time.perf_counter() - fps_calc_start
                self.total_fps_time += fps_time

                # 在帧上显示统计信息
                self.display_stats(annotated_frame, current_fps, processing_time,
                                   preprocess_time, detection_time, postprocess_time)

                # 写入输出文件
                if writer is not None:
                    writer.write(annotated_frame)

                # 显示结果
                cv2.imshow('YOLO实时目标检测', annotated_frame)

            # 键盘输入处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 退出
                break
            elif key == ord('p'):  # 暂停/继续
                paused = not paused
                print("检测已暂停" if paused else "检测继续")
            elif key == ord('s'):  # 保存当前帧
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"帧已保存为: {filename}")

        # 计算总时间统计
        total_time = time.perf_counter() - start_time_total

        # 释放资源
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # 打印详细的时间分析报告
        self.print_detailed_report(total_time)

    def display_stats(self, frame, fps, processing_time, preprocess_time, detection_time, postprocess_time):
        """在帧上显示统计信息"""
        stats = [
            f"FPS: {fps:.2f}",
            f"frame time: {processing_time * 1000:.1f}ms",
            f"preprocess: {preprocess_time * 1000:.1f}ms",
            f"detection: {detection_time * 1000:.1f}ms",
            f"postprocess: {postprocess_time * 1000:.1f}ms",
            f"total  frames: {self.frame_count}"
        ]

        # 绘制统计信息背景
        for i, text in enumerate(stats):
            y_position = 30 + i * 25
            cv2.rectangle(frame, (10, y_position - 20), (300, y_position + 5), (0, 0, 0), -1)

        # 绘制文本
        for i, text in enumerate(stats):
            y_position = 30 + i * 25
            color = (0, 255, 0) if i == 0 else (255, 255, 255)  # FPS用绿色突出显示
            cv2.putText(frame, text, (10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def print_detailed_report(self, total_time):
        """打印详细的时间分析报告"""
        print("\n" + "=" * 50)
        print("YOLO目标检测时间分析报告")
        print("=" * 50)

        if self.frame_count > 0:
            avg_preprocess = self.total_preprocess_time / self.frame_count
            avg_detection = self.total_detection_time / self.frame_count
            avg_postprocess = self.total_postprocess_time / self.frame_count
            avg_fps_calc = self.total_fps_time / self.frame_count
            avg_total_per_frame = total_time / self.frame_count
            actual_fps = self.frame_count / total_time

            print(f"总运行时间: {total_time:.2f} 秒")
            print(f"处理总帧数: {self.frame_count}")
            print(f"实际平均FPS: {actual_fps:.2f}")
            print(f"平均每帧总时间: {avg_total_per_frame * 1000:.1f} ms")
            print(f"平均预处理时间: {avg_preprocess * 1000:.1f} ms ({avg_preprocess / avg_total_per_frame * 100:.1f}%)")
            print(f"平均检测时间: {avg_detection * 1000:.1f} ms ({avg_detection / avg_total_per_frame * 100:.1f}%)")
            print(
                f"平均后处理时间: {avg_postprocess * 1000:.1f} ms ({avg_postprocess / avg_total_per_frame * 100:.1f}%)")
            print(f"平均FPS计算时间: {avg_fps_calc * 1000:.1f} ms ({avg_fps_calc / avg_total_per_frame * 100:.1f}%)")

            # 时间分布饼图数据
            times = [avg_preprocess, avg_detection, avg_postprocess, avg_fps_calc]
            labels = ['预处理', '目标检测', '后处理', 'FPS计算']

            print("\n时间分布:")
            for label, time_val in zip(labels, times):
                percentage = time_val / avg_total_per_frame * 100
                print(f"  {label}: {percentage:.1f}%")
        else:
            print("未处理任何帧")


def main():
    """主函数"""
    print("YOLO目标检测器启动中...")

    # 创建检测器实例
    detector = YOLODetector('yolo11x.pt')  # 可以使用 yolov8s.pt, yolov8m.pt 等更大模型

    try:
        # 运行检测（0=默认摄像头，也可以指定视频文件路径）
        detector.run_detection(source=r"E:\code\PycharmProjects\rf-detr-develop\rf_video.mp4", output_file='rf_video_result_yolo.mp4')  # 可选保存输出视频

    except KeyboardInterrupt:
        print("\n检测被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        print("检测结束")


if __name__ == "__main__":
    main()