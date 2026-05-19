import warnings

warnings.filterwarnings('ignore')  # 忽略所有警告
warnings.simplefilter('ignore')  # 设置简单过滤器忽略警告
import torch, yaml, cv2, os, shutil, sys
import numpy as np

np.random.seed(0)  # 设置NumPy随机种子保证可重复性
import matplotlib.pyplot as plt
from tqdm import trange  # 进度条工具
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights  # YOLO模型加载工具
from ultralytics.utils.torch_utils import intersect_dicts  # 模型字典处理工具
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression  # 坐标转换和NMS操作
# 导入多种Grad-CAM变体
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image  # CAM可视化工具
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients  # 激活和梯度处理


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """图像缩放和填充函数，保持宽高比并添加边缘填充"""
    shape = im.shape[:2]  # 获取原始图像尺寸(height, width)
    # 处理新尺寸为整数的情形
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例（取宽高比例中较小的）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只缩小不放大（优化验证mAP）
        r = min(r, 1.0)

    # 计算填充尺寸
    ratio = r, r  # 宽高缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 缩放后尺寸
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽高方向需要填充的像素
    # 自动计算最小矩形填充
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 确保填充后尺寸是stride的倍数
    elif scaleFill:  # 直接拉伸不保持比例
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 独立的宽高比例

    # 将填充分摊到两边
    dw /= 2
    dh /= 2

    # 缩放图像
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    # 计算上下左右填充值
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 添加边框填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


class ActivationsAndGradients:
    """用于提取激活值和目标中间层的梯度"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []  # 存储梯度
        self.activations = []  # 存储激活值
        self.reshape_transform = reshape_transform  # 形状转换函数
        self.handles = []  # 钩子句柄
        # 为目标层注册前向钩子
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """保存激活值"""
        activation = output
        # 应用形状变换（如果需要）
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        """保存梯度"""
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return  # 只处理需要梯度的张量

        # 定义梯度存储函数
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            # 梯度按反向顺序存储
            self.gradients = [grad.cpu().detach()] + self.gradients

        # 注册梯度钩子
        output.register_hook(_store_grad)

    def post_process(self, result):
        """后处理模型输出"""
        logits_ = result[:, 4:]  # 类别置信度
        boxes_ = result[:, :4]  # 边界框坐标
        # 按置信度降序排序
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        # 返回排序后的结果
        return (
            torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]],
            torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]],
            xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
        )

    def __call__(self, x):
        """前向传播调用"""
        self.gradients = []  # 清空梯度
        self.activations = []  # 清空激活值
        model_output = self.model(x)  # 模型前向传播
        # 获取后处理结果
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        """释放所有钩子"""
        for handle in self.handles:
            handle.remove()


class yolov8_target(torch.nn.Module):
    """目标类用于计算Grad-CAM的梯度"""

    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type  # 目标类型：class/box/all
        self.conf = conf  # 置信度阈值
        self.ratio = ratio  # 参与计算的目标比例

    def forward(self, data):
        """前向传播计算目标值"""
        post_result, pre_post_boxes = data
        result = []  # 存储结果
        # 只处理前ratio比例的高置信度目标
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:  # 低于置信度阈值则跳过
                break
            # 根据目标类型累加值
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())  # 类别分数
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])  # 边界框坐标
        return sum(result)  # 返回所有值的和


class yolov11_heatmap:
    """YOLO热力图生成主类"""

    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)  # 设备设置
        ckpt = torch.load(weight)  # 加载模型权重
        model_names = ckpt['model'].names  # 获取类别名称
        model = attempt_load_weights(weight, device)  # 加载模型
        model.info()  # 打印模型信息
        # 启用梯度计算
        for p in model.parameters():
            p.requires_grad_(True)
            model.eval()  # 设置为评估模式

        # 初始化目标函数
        target = yolov8_target(backward_type, conf_threshold, ratio)
        # 获取目标层
        target_layers = [model.model[l] for l in layer]
        # 初始化Grad-CAM方法
        method = eval(method)(model, target_layers)
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)

        # 为每个类别生成随机颜色
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.uint8)
        # 将局部变量存入类的__dict__
        self.__dict__.update(locals())

    def post_process(self, result):
        """后处理：非极大值抑制(NMS)"""
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        """在图像上绘制检测框和标签"""
        xmin, ymin, xmax, ymax = list(map(int, list(box)))  # 转换坐标为整数
        # 绘制边界框
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        # 添加标签文本
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
                    lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """在边界框内归一化CAM，边界框外归零"""
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:  # 遍历所有边界框
            # 确保坐标不越界
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            # 在边界框内缩放CAM值
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        # 整体缩放CAM
        renormalized_cam = scale_cam_image(renormalized_cam)
        # 生成可视化图像
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def process(self, img_path, save_path):
        """处理单张图像生成热力图"""
        # 读取和预处理图像
        img = cv2.imread(img_path)
        img = letterbox(img)[0]  # 缩放和填充
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB
        img = np.float32(img) / 255.0  # 归一化
        # 转换为张量并添加批次维度
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        try:
            # 计算Grad-CAM
            grayscale_cam = self.method(tensor, [self.target])
        except AttributeError as e:
            return  # 跳过异常

        grayscale_cam = grayscale_cam[0, :]  # 获取单通道热力图
        # 在原始图像上叠加热力图
        cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        # 模型推理
        pred = self.model(tensor)[0]
        pred = self.post_process(pred)  # NMS后处理
        # 可选：在检测框内重新归一化热力图
        if self.renormalize:
            cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img,
                                                               grayscale_cam)
        # 可选：绘制检测框
        if self.show_box:
            for data in pred:
                data = data.cpu().detach().numpy()
                # 获取类别索引和置信度
                class_id = int(data[4:].argmax())
                confidence = float(data[4:].max())
                # 绘制检测结果
                cam_image = self.draw_detections(
                    data[:4],
                    self.colors[class_id],
                    f'{self.model_names[class_id]} {confidence:.2f}',
                    cam_image
                )

        # 保存结果图像
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)

    def __call__(self, img_path, save_path, grad_name):
        """处理入口：支持单张图像或整个文件夹"""
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # 判断输入是目录还是文件
        if os.path.isdir(img_path):
            # 处理目录中所有图像
            for img_path_ in os.listdir(img_path):
                # 解析文件名
                name, extension = os.path.splitext(img_path_)
                # 处理单张图像
                self.process(
                    os.path.join(img_path, img_path_),
                    os.path.join(save_path, f'{name}_{grad_name}{extension}')
                )
        else:
            # 处理单张图像
            self.process(img_path, os.path.join(save_path, f'result_{grad_name}.png'))


def get_params():
    """生成不同Grad-CAM变体的参数配置"""
    # 支持的Grad-CAM方法列表
    grad_list = [
        'GradCAM',
        'GradCAMPlusPlus',
        'XGradCAM',
        'EigenCAM',
        'HiResCAM',
        'LayerCAM',
        'RandomCAM',
        'EigenGradCAM'
    ]
    # 目标层配置（可单层或多层组合）
    layers = [8]  # 使用第8层

    # 为每种方法生成参数配置
    for grad_name in grad_list:
        params = {
            'weight': r'E:\code\PycharmProjects\YOLOV11\runs\detect\train38\weights\best.pt',  # 模型权重路径
            'device': 'cpu',  # 运行设备
            'method': grad_name,  # Grad-CAM方法
            'layer': layers,  # 目标层索引
            'backward_type': 'class',  # 反向传播类型：class/box/all
            'conf_threshold': 0.2,  # 检测置信度阈值
            'ratio': 0.02,  # 参与计算的目标比例(2%)
            'show_box': True,  # 是否显示检测框
            'renormalize': False  # 是否在边界框内归一化热力图
        }
        yield params  # 生成参数配置


if __name__ == '__main__':
    # 主执行入口
    for each in get_params():  # 遍历所有参数配置
        # 初始化热力图生成器
        model = yolov11_heatmap(**each)
        # 处理图像并保存结果
        # 单张图像：model('images/00052.jpg', 'result', each['method'])
        #model(r'images', 'result', each['method'])  # 处理整个目录
        model(r'E:\code\MVSimg\2025827', r'E:\code\MVSimg\2025827\result', each['method'])
