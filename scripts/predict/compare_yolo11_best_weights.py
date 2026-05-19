#!/usr/bin/env python
"""批量对比 YOLOv11 多个 best 权重在 all_images 上的预测效果。

脚本会自动完成下面几件事：
1. 扫描 runs/detect/train_yolo11*/weights 下的 best*.pt。
2. 对 datasets/construction-ppe/all_images 进行批量预测。
3. 为每个模型分别保存预测结果、CSV/JSON 明细和 Markdown 详细报告。
4. 在 runs/predict_compare 下生成一个总汇总 Markdown，方便横向比较。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yaml
from ultralytics import YOLO

try:
    import torch
except ImportError:  # pragma: no cover - 仅作为兼容兜底
    torch = None


IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class WeightInfo:
    """描述单个待测试权重的基础信息。"""

    model_name: str
    train_dir: Path
    weight_path: Path
    weight_size_mb: float
    args: dict[str, Any]
    best_metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    # 默认认为脚本放在 scripts/predict 目录下，因此向上三级就是项目根目录。
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent

    parser = argparse.ArgumentParser(
        description="批量比较 YOLOv11 多个 best 权重的预测结果。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="YOLOV11 项目根目录，默认按当前脚本位置自动推断。",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=project_root / "datasets" / "construction-ppe" / "all_images",
        help="用于预测的图片目录。",
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=project_root / "runs" / "detect",
        help="包含 train_yolo11* 训练实验目录的根路径。",
    )
    parser.add_argument(
        "--pattern",
        default="train_yolo11*",
        help="weights-root 下实验目录的匹配模式。",
    )
    parser.add_argument(
        "--weight-pattern",
        default="best*.pt",
        help="每个实验 weights 目录内权重文件的匹配模式。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "runs" / "predict_compare",
        help="用于保存预测结果和报告的输出目录。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="预测时使用的置信度阈值。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理图片尺寸。",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="预测批次大小。显存不足时建议保持为 1。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="可选设备参数，例如 0 或 cpu。",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="调试时可限制仅预测前 N 张图。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """安全读取 YAML，缺失或格式异常时返回空字典。"""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def format_float(value: Any, digits: int = 4) -> str:
    """统一格式化浮点数，方便写入 Markdown 表格。"""

    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def format_milliseconds(value: float) -> str:
    return f"{value:.2f}ms"


def clear_torch_memory() -> None:
    """在模型切换后主动释放 Python 和 PyTorch 占用的缓存。"""

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_rel(path: Path, base: Path) -> str:
    """优先输出相对路径，失败时退回绝对路径。"""

    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def collect_images(source_dir: Path, max_images: int | None = None) -> list[Path]:
    # 递归收集 all_images 下所有常见图片格式，保证子目录也能被处理。
    images = sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ]
    )
    if max_images is not None:
        return images[:max_images]
    return images


def read_training_best_metrics(results_csv: Path) -> dict[str, Any]:
    """从训练阶段的 results.csv 中取出 mAP50-95 最优那一行指标。"""

    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return {}

    def metric_value(row: dict[str, str], key: str) -> float:
        # 某些字段可能为空，这里统一兜底，便于后续排序和展示。
        raw = row.get(key, "")
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return float("-inf")

    best_row = max(rows, key=lambda row: metric_value(row, "metrics/mAP50-95(B)"))
    return {
        "epoch": int(float(best_row["epoch"])) if best_row.get("epoch") else None,
        "time": metric_value(best_row, "time"),
        "precision": metric_value(best_row, "metrics/precision(B)"),
        "recall": metric_value(best_row, "metrics/recall(B)"),
        "mAP50": metric_value(best_row, "metrics/mAP50(B)"),
        "mAP50_95": metric_value(best_row, "metrics/mAP50-95(B)"),
        "val_box_loss": metric_value(best_row, "val/box_loss"),
        "val_cls_loss": metric_value(best_row, "val/cls_loss"),
        "val_dfl_loss": metric_value(best_row, "val/dfl_loss"),
    }


def collect_weights(weights_root: Path, pattern: str, weight_pattern: str) -> list[WeightInfo]:
    """扫描 train_yolo11* 目录，收集待比较的 best 权重。"""

    weights: list[WeightInfo] = []

    for train_dir in sorted(weights_root.glob(pattern)):
        if not train_dir.is_dir():
            continue

        weight_dir = train_dir / "weights"
        for weight_path in sorted(weight_dir.glob(weight_pattern)):
            # 同步读取训练配置和训练结果，便于最终报告把“训练指标”和“预测表现”放一起比较。
            args = load_yaml(train_dir / "args.yaml")
            best_metrics = read_training_best_metrics(train_dir / "results.csv")
            model_name = train_dir.name
            if weight_path.name != "best.pt":
                model_name = f"{train_dir.name}_{weight_path.stem}"

            weights.append(
                WeightInfo(
                    model_name=model_name,
                    train_dir=train_dir,
                    weight_path=weight_path,
                    weight_size_mb=weight_path.stat().st_size / (1024 * 1024),
                    args=args,
                    best_metrics=best_metrics,
                )
            )

    return weights


def count_label_distribution(label_dir: Path, names: dict[int, str]) -> dict[str, Any]:
    """统计 all_labels 中每个类别的标注框数量，用于预测分布对照。"""

    if not label_dir.exists():
        return {}

    class_counts: Counter[str] = Counter()
    total_boxes = 0
    labeled_images = 0

    for label_file in sorted(label_dir.glob("*.txt")):
        lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue

        labeled_images += 1
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                continue
            class_name = names.get(class_id, f"class_{class_id}")
            class_counts[class_name] += 1
            total_boxes += 1

    return {
        "labeled_images": labeled_images,
        "total_boxes": total_boxes,
        "class_counts": dict(sorted(class_counts.items())),
    }


def load_class_name_map(data_yaml_path: Path) -> dict[int, str]:
    """从 data.yaml 中读取类别 id 到类别名的映射。"""

    data_yaml = load_yaml(data_yaml_path)
    names_config = data_yaml.get("names")
    if isinstance(names_config, list):
        return {idx: name for idx, name in enumerate(names_config)}
    if isinstance(names_config, dict):
        class_map: dict[int, str] = {}
        for idx, name in names_config.items():
            try:
                class_map[int(idx)] = str(name)
            except (TypeError, ValueError):
                continue
        return class_map
    return {}


def serialise_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """把 Path 等对象转换成可写入 JSON/CSV 的基础类型。"""

    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Path):
            output[key] = str(value)
        elif isinstance(value, dict):
            output[key] = dict(value)
        else:
            output[key] = value
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV 明细，使用 utf-8-sig 方便 Windows Excel 直接打开。"""

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 报告。"""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_class_count_text(class_counts: dict[str, int]) -> str:
    """把类别计数字典压成单行文本，便于放进表格。"""

    if not class_counts:
        return "-"
    return ", ".join(f"{name}:{count}" for name, count in sorted(class_counts.items()))


def ensure_hardlink_or_copy(src: Path, dst: Path) -> None:
    """优先创建硬链接，失败时退回复制，避免重复占满磁盘空间。"""

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        # 某些 Python 版本的 pathlib 没有 Path.hardlink_to()，这里直接用 os.link 提高兼容性。
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prepare_eval_dataset_view(
    run_dir: Path,
    source_dir: Path,
    image_paths: list[Path],
    label_dir: Path,
    class_name_map: dict[int, str],
) -> Path:
    """构造一个临时评估数据集视图，供 Ultralytics 标准验证器计算 P/R/mAP。

    这样做的原因是：
    - 你当前真实目录是 `all_images` / `all_labels`
    - Ultralytics 默认验证逻辑要求路径形如 `images/...` 和 `labels/...`
    - 所以这里在本次输出目录下临时搭一个标准结构，再直接调用 `model.val(...)`
    """

    eval_root = run_dir / "_eval_dataset"
    eval_images_dir = eval_root / "images" / "val"
    eval_labels_dir = eval_root / "labels" / "val"
    eval_images_dir.mkdir(parents=True, exist_ok=True)
    eval_labels_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        relative_image = image_path.relative_to(source_dir)
        eval_image_path = eval_images_dir / relative_image
        ensure_hardlink_or_copy(image_path, eval_image_path)

        # all_labels 当前是平铺目录，因此这里按图片 stem 去找对应标签。
        label_src = label_dir / f"{image_path.stem}.txt"
        if label_src.exists():
            eval_label_path = eval_labels_dir / relative_image.with_suffix(".txt")
            ensure_hardlink_or_copy(label_src, eval_label_path)

    eval_yaml_path = eval_root / "data_eval.yaml"
    eval_yaml = {
        "path": str(eval_root),
        "train": "images/val",
        "val": "images/val",
        "names": class_name_map,
    }
    with eval_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(eval_yaml, handle, allow_unicode=True, sort_keys=False)
    return eval_yaml_path


def evaluate_one_model(
    model: YOLO,
    model_info: WeightInfo,
    eval_data_yaml: Path,
    run_dir: Path,
    imgsz: int,
    batch: int,
    device: str | None,
    class_name_map: dict[int, str],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """在 all_images/all_labels 上做一次标准验证，拿到真正的 P/R/mAP。"""

    validation_project = run_dir / "validation"
    validation_kwargs = {
        "data": str(eval_data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "conf": 0.001,  # mAP 评估通常使用较低置信度阈值，避免提前截断候选框
        "iou": 0.7,
        "max_det": 300,
        "workers": 0,  # Windows 下更稳妥，避免多进程带来的额外问题
        "plots": False,
        "save_json": False,
        "verbose": False,
        "project": str(validation_project),
        "name": model_info.model_name,
        "exist_ok": True,
    }
    if device:
        validation_kwargs["device"] = device

    metrics = model.val(**validation_kwargs)
    results_dict = metrics.results_dict

    overall_metrics = {
        "precision": float(results_dict.get("metrics/precision(B)", 0.0)),
        "recall": float(results_dict.get("metrics/recall(B)", 0.0)),
        "mAP50": float(results_dict.get("metrics/mAP50(B)", 0.0)),
        "mAP50_95": float(results_dict.get("metrics/mAP50-95(B)", 0.0)),
        "fitness": float(results_dict.get("fitness", 0.0)),
    }

    # 先把验证器真正算出来的分类别指标取出来。
    per_class_found: dict[int, dict[str, Any]] = {}
    for metric_index, class_id in enumerate(getattr(metrics, "ap_class_index", [])):
        precision, recall, ap50, ap = metrics.class_result(metric_index)
        per_class_found[int(class_id)] = {
            "class_id": int(class_id),
            "class_name": class_name_map.get(int(class_id), f"class_{class_id}"),
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        }

    # 最终输出时按 data.yaml 的类别顺序补全，没有参与 AP 统计的类别填 0。
    per_class_metrics: list[dict[str, Any]] = []
    for class_id, class_name in sorted(class_name_map.items()):
        metric_row = per_class_found.get(
            int(class_id),
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50_95": 0.0,
            },
        )
        metric_row["gt_count"] = int(ground_truth.get("class_counts", {}).get(class_name, 0))
        per_class_metrics.append(metric_row)

    return {
        "overall": overall_metrics,
        "per_class": per_class_metrics,
        "speed": {key: float(value) for key, value in metrics.speed.items()},
        "validation_dir": str(validation_project / model_info.model_name),
    }


def build_model_detail_markdown(
    run_dir: Path,
    model_summary: dict[str, Any],
    detail_rows: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> str:
    """构建单个模型的详细 Markdown 报告。"""

    train_metrics = model_summary["train_best_metrics"]
    eval_metrics = model_summary["evaluation_metrics"]["overall"]
    eval_speed = model_summary["evaluation_metrics"]["speed"]
    per_class_metrics = model_summary["evaluation_metrics"]["per_class"]

    lines: list[str] = []
    lines.append(f"# {model_summary['model_name']} Prediction Detail Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Generated at: {model_summary['generated_at']}")
    lines.append(f"- Weight: `{model_summary['weight_path']}`")
    lines.append(f"- Training run: `{model_summary['train_dir']}`")
    lines.append(f"- Weight size: {model_summary['weight_size_mb']:.2f} MB")
    lines.append(f"- Source: `{model_summary['source_dir']}`")
    lines.append(f"- Batch size: {model_summary['batch_size']}")
    lines.append(f"- Validation output: `{model_summary['evaluation_metrics']['validation_dir']}`")
    lines.append(f"- Prediction output: `{model_summary['prediction_dir']}`")
    lines.append(f"- Detail CSV: `{model_summary['detail_csv']}`")
    lines.append(f"- Detail JSON: `{model_summary['detail_json']}`")
    lines.append("")

    lines.append("## Evaluation Metrics On all_images/all_labels")
    lines.append("")
    lines.append("| precision | recall | mAP50 | mAP50-95 | val_preprocess_ms | val_inference_ms | val_postprocess_ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| "
        f"{format_float(eval_metrics.get('precision'))} | "
        f"{format_float(eval_metrics.get('recall'))} | "
        f"{format_float(eval_metrics.get('mAP50'))} | "
        f"{format_float(eval_metrics.get('mAP50_95'))} | "
        f"{format_milliseconds(eval_speed.get('preprocess', 0.0))} | "
        f"{format_milliseconds(eval_speed.get('inference', 0.0))} | "
        f"{format_milliseconds(eval_speed.get('postprocess', 0.0))} |"
    )
    lines.append("")

    lines.append("## Per-Class Evaluation Metrics")
    lines.append("")
    lines.append("| class_id | class | gt_count | precision | recall | mAP50 | mAP50-95 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in per_class_metrics:
        lines.append(
            "| "
            f"{row['class_id']} | "
            f"{row['class_name']} | "
            f"{row['gt_count']} | "
            f"{format_float(row['precision'])} | "
            f"{format_float(row['recall'])} | "
            f"{format_float(row['mAP50'])} | "
            f"{format_float(row['mAP50_95'])} |"
        )
    lines.append("")

    lines.append("## Training Best Metrics")
    lines.append("")
    lines.append("| epoch | precision | recall | mAP50 | mAP50-95 | val_box_loss | val_cls_loss | val_dfl_loss |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| "
        f"{train_metrics.get('epoch', '-')} | "
        f"{format_float(train_metrics.get('precision'))} | "
        f"{format_float(train_metrics.get('recall'))} | "
        f"{format_float(train_metrics.get('mAP50'))} | "
        f"{format_float(train_metrics.get('mAP50_95'))} | "
        f"{format_float(train_metrics.get('val_box_loss'))} | "
        f"{format_float(train_metrics.get('val_cls_loss'))} | "
        f"{format_float(train_metrics.get('val_dfl_loss'))} |"
    )
    lines.append("")

    lines.append("## Prediction Summary")
    lines.append("")
    lines.append("| images | detected_images | total_detections | avg_det_per_image | avg_conf | wall_time | avg_wall_ms | avg_inference_ms |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        "| "
        f"{model_summary['image_count']} | "
        f"{model_summary['detected_images']} | "
        f"{model_summary['total_detections']} | "
        f"{format_float(model_summary['avg_detections_per_image'], 2)} | "
        f"{format_float(model_summary['avg_confidence'], 4)} | "
        f"{format_seconds(model_summary['wall_time_seconds'])} | "
        f"{format_milliseconds(model_summary['avg_wall_time_ms'])} | "
        f"{format_milliseconds(model_summary['avg_inference_ms'])} |"
    )
    lines.append("")

    lines.append("## Prediction Class Distribution")
    lines.append("")
    lines.append("| class | predicted_count |")
    lines.append("| --- | --- |")
    for class_name, count in sorted(model_summary["predicted_class_counts"].items()):
        lines.append(f"| {class_name} | {count} |")
    if not model_summary["predicted_class_counts"]:
        lines.append("| - | 0 |")
    lines.append("")

    if ground_truth:
        lines.append("## Ground Truth Distribution")
        lines.append("")
        lines.append(f"- Label directory: `{ground_truth['label_dir']}`")
        lines.append(f"- Labeled images: {ground_truth['labeled_images']}, total boxes: {ground_truth['total_boxes']}")
        lines.append("")
        lines.append("| class | gt_count | predicted_count | diff(pred-gt) |")
        lines.append("| --- | --- | --- | --- |")
        all_classes = sorted(
            set(ground_truth["class_counts"].keys())
            | set(model_summary["predicted_class_counts"].keys())
        )
        for class_name in all_classes:
            gt_count = ground_truth["class_counts"].get(class_name, 0)
            pred_count = model_summary["predicted_class_counts"].get(class_name, 0)
            lines.append(f"| {class_name} | {gt_count} | {pred_count} | {pred_count - gt_count} |")
        lines.append("")

    lines.append("## Per-Image Detail")
    lines.append("")
    lines.append(
        "| image | detections | avg_conf | preprocess_ms | inference_ms | postprocess_ms | speed_total_ms | class_counts | saved_image | saved_label |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in detail_rows:
        lines.append(
            "| "
            f"{row['image_name']} | "
            f"{row['detections']} | "
            f"{format_float(row['avg_conf'], 4)} | "
            f"{format_milliseconds(row['preprocess_ms'])} | "
            f"{format_milliseconds(row['inference_ms'])} | "
            f"{format_milliseconds(row['postprocess_ms'])} | "
            f"{format_milliseconds(row['speed_total_ms'])} | "
            f"{build_class_count_text(row['class_counts'])} | "
            f"`{row['saved_image_rel']}` | "
            f"`{row['saved_label_rel']}` |"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `Evaluation Metrics` 是在 `all_images/all_labels` 上重新跑标准验证得到的真实测试指标。")
    lines.append("- `Prediction Summary` 统计的是逐张推理时保存图片、标签和耗时的情况。")
    lines.append("- `wall_time` 是整套逐图预测的总耗时，`avg_inference_ms` 来自 Ultralytics 的单图速度统计。")
    lines.append("")

    return "\n".join(lines)


def build_summary_markdown(
    run_dir: Path,
    summaries: list[dict[str, Any]],
    source_dir: Path,
    ground_truth: dict[str, Any],
) -> str:
    """构建所有模型的总汇总 Markdown 报告。"""

    sorted_by_map = sorted(
        summaries,
        key=lambda item: item["evaluation_metrics"]["overall"].get("mAP50_95", float("-inf")),
        reverse=True,
    )
    sorted_by_speed = sorted(summaries, key=lambda item: item["avg_inference_ms"])

    lines: list[str] = []
    lines.append("# YOLOv11 Best Weights Prediction Comparison")
    lines.append("")
    lines.append("## Run Info")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Report directory: `{run_dir}`")
    lines.append(f"- Source directory: `{source_dir}`")
    lines.append(f"- Image count: {summaries[0]['image_count'] if summaries else 0}")
    lines.append(f"- Model count: {len(summaries)}")
    if ground_truth:
        lines.append(
            f"- Ground-truth labels: `{ground_truth['label_dir']}` ({ground_truth['labeled_images']} images, {ground_truth['total_boxes']} boxes)"
        )
    lines.append("")

    lines.append("## Model Comparison")
    lines.append("")
    lines.append(
        "| model | best_epoch | eval_precision | eval_recall | eval_mAP50 | eval_mAP50-95 | total_det | detected_images | avg_conf | avg_infer_ms | detail_report |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in summaries:
        train_metrics = item["train_best_metrics"]
        eval_metrics = item["evaluation_metrics"]["overall"]
        lines.append(
            "| "
            f"{item['model_name']} | "
            f"{train_metrics.get('epoch', '-')} | "
            f"{format_float(eval_metrics.get('precision'))} | "
            f"{format_float(eval_metrics.get('recall'))} | "
            f"{format_float(eval_metrics.get('mAP50'))} | "
            f"{format_float(eval_metrics.get('mAP50_95'))} | "
            f"{item['total_detections']} | "
            f"{item['detected_images']} | "
            f"{format_float(item['avg_confidence'], 4)} | "
            f"{format_milliseconds(item['avg_inference_ms'])} | "
            f"`{item['detail_md_rel']}` |"
        )
    lines.append("")

    lines.append("## Ranking")
    lines.append("")
    if sorted_by_map:
        best_map = sorted_by_map[0]
        lines.append(
            f"- Highest eval mAP50-95: `{best_map['model_name']}` ({format_float(best_map['evaluation_metrics']['overall'].get('mAP50_95'))})"
        )
    if sorted_by_speed:
        fastest = sorted_by_speed[0]
        lines.append(
            f"- Fastest avg inference: `{fastest['model_name']}` ({format_milliseconds(fastest['avg_inference_ms'])})"
        )
    most_detections = max(summaries, key=lambda item: item["total_detections"], default=None)
    if most_detections is not None:
        lines.append(
            f"- Most predicted boxes: `{most_detections['model_name']}` ({most_detections['total_detections']})"
        )
    lines.append("")

    # 分类别对比优先展示真正的评估指标，而不是只看预测框数量。
    if summaries:
        reference_classes = summaries[0]["evaluation_metrics"]["per_class"]
        if reference_classes:
            lines.append("## Per-Class Eval mAP50-95")
            lines.append("")
            class_names = [row["class_name"] for row in reference_classes]
            lines.append("| model | " + " | ".join(class_names) + " |")
            lines.append("| --- | " + " | ".join(["---"] * len(class_names)) + " |")
            for item in summaries:
                per_class_map = {
                    row["class_name"]: row["mAP50_95"] for row in item["evaluation_metrics"]["per_class"]
                }
                values = [format_float(per_class_map.get(class_name, 0.0)) for class_name in class_names]
                lines.append(f"| {item['model_name']} | " + " | ".join(values) + " |")
            lines.append("")

    if ground_truth:
        lines.append("## Ground Truth Class Counts")
        lines.append("")
        lines.append("| class | gt_count |")
        lines.append("| --- | --- |")
        for class_name, count in sorted(ground_truth["class_counts"].items()):
            lines.append(f"| {class_name} | {count} |")
        lines.append("")

    return "\n".join(lines)


def infer_saved_paths(
    prediction_dir: Path,
    image_path: Path,
    source_dir: Path,
) -> tuple[Path, Path]:
    """根据原图相对路径推断可视化图片和标签文本的保存位置。"""

    relative_image = image_path.relative_to(source_dir)
    saved_image = prediction_dir / relative_image
    saved_label = prediction_dir / "labels" / relative_image.with_suffix(".txt")
    return saved_image, saved_label


def build_predict_kwargs(
    source: str,
    prediction_project: Path,
    model_name: str,
    conf: float,
    imgsz: int,
    batch: int,
    device: str | None,
) -> dict[str, Any]:
    """构建单次预测调用参数。"""

    predict_kwargs = {
        "source": source,
        "conf": conf,
        "imgsz": imgsz,
        "batch": batch,
        "save": True,
        "save_txt": True,
        "save_conf": True,
        "project": str(prediction_project),
        "name": model_name,
        "exist_ok": True,
        "verbose": False,
    }
    if device:
        predict_kwargs["device"] = device
    return predict_kwargs


def predict_one_model(
    model_info: WeightInfo,
    image_paths: list[Path],
    source_dir: Path,
    run_dir: Path,
    eval_data_yaml: Path,
    class_name_map: dict[int, str],
    ground_truth: dict[str, Any],
    conf: float,
    imgsz: int,
    batch: int,
    device: str | None,
) -> dict[str, Any]:
    """执行单个模型的批量预测，并汇总该模型的全部统计信息。"""

    # predictions 保存可视化预测结果，reports 保存结构化明细和 Markdown 文档。
    prediction_project = run_dir / "predictions"
    prediction_dir = prediction_project / model_info.model_name
    detail_dir = run_dir / "reports" / model_info.model_name
    detail_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"[{model_info.model_name}] loading weight: {model_info.weight_path}")
    model = YOLO(model_info.weight_path)
    model_load_finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 先在 all_images/all_labels 上跑一次标准验证，得到真正的 P/R/mAP 指标。
    print(f"[{model_info.model_name}] running validation on all_images/all_labels ...")
    evaluation_metrics = evaluate_one_model(
        model=model,
        model_info=model_info,
        eval_data_yaml=eval_data_yaml,
        run_dir=run_dir,
        imgsz=imgsz,
        batch=batch,
        device=device,
        class_name_map=class_name_map,
        ground_truth=ground_truth,
    )
    print(
        f"[{model_info.model_name}] eval done: "
        f"P={evaluation_metrics['overall']['precision']:.4f}, "
        f"R={evaluation_metrics['overall']['recall']:.4f}, "
        f"mAP50={evaluation_metrics['overall']['mAP50']:.4f}, "
        f"mAP50-95={evaluation_metrics['overall']['mAP50_95']:.4f}"
    )

    wall_start = time.perf_counter()
    # 下面这些列表用于后续统计平均耗时、平均置信度和类别分布。
    predicted_class_counts: Counter[str] = Counter()
    per_image_rows: list[dict[str, Any]] = []
    image_level_detections: list[int] = []
    image_level_confidences: list[float] = []
    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    total_speed_times: list[float] = []
    detected_images = 0

    # 这里改成“逐张按文件路径预测”。
    # 原因是 source 传 Python 列表时，Ultralytics 会先把图片转成内存对象列表，
    # 反而绕开了真正的文件流式加载，导致 batch=1 也可能触发异常的大显存申请。
    for index, image_path in enumerate(image_paths, start=1):
        if index == 1 or index % 100 == 0 or index == len(image_paths):
            print(f"[{model_info.model_name}] {index}/{len(image_paths)} -> {image_path.name}")

        predict_kwargs = build_predict_kwargs(
            source=str(image_path),
            prediction_project=prediction_project,
            model_name=model_info.model_name,
            conf=conf,
            imgsz=imgsz,
            batch=batch,
            device=device,
        )

        results = model.predict(**predict_kwargs)
        if not results:
            continue

        result = results[0]
        boxes = result.boxes
        detections = len(boxes)
        image_level_detections.append(detections)
        if detections > 0:
            detected_images += 1

        confs: list[float] = []
        class_counts: Counter[str] = Counter()
        for box in boxes:
            # 每个框记录类别和置信度，后面会同时写入单图明细和模型总统计。
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            conf_value = float(box.conf.item())
            confs.append(conf_value)
            class_counts[class_name] += 1
            predicted_class_counts[class_name] += 1
            image_level_confidences.append(conf_value)

        preprocess_ms = float(result.speed.get("preprocess", 0.0))
        inference_ms = float(result.speed.get("inference", 0.0))
        postprocess_ms = float(result.speed.get("postprocess", 0.0))
        speed_total_ms = preprocess_ms + inference_ms + postprocess_ms

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        total_speed_times.append(speed_total_ms)

        saved_image, saved_label = infer_saved_paths(prediction_dir, image_path, source_dir)

        # 每张图都保留一条明细记录，方便之后排查具体哪张图差异最大。
        per_image_rows.append(
            {
                "image_name": image_path.name,
                "image_rel": str(image_path.relative_to(source_dir)),
                "image_path": image_path,
                "detections": detections,
                "avg_conf": mean(confs) if confs else 0.0,
                "max_conf": max(confs) if confs else 0.0,
                "min_conf": min(confs) if confs else 0.0,
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "speed_total_ms": speed_total_ms,
                "class_counts": dict(sorted(class_counts.items())),
                "saved_image": saved_image,
                "saved_label": saved_label,
                "saved_image_rel": safe_rel(saved_image, run_dir),
                "saved_label_rel": safe_rel(saved_label, run_dir) if saved_label.exists() else "-",
            }
        )

        # 单张预测完成后释放一次缓存，降低长时间连续推理的显存波动。
        clear_torch_memory()
    wall_time = time.perf_counter() - wall_start

    detail_csv = detail_dir / f"{model_info.model_name}_prediction_details.csv"
    detail_json = detail_dir / f"{model_info.model_name}_prediction_details.json"
    detail_md = detail_dir / f"{model_info.model_name}_prediction_detail.md"

    # CSV 里把 class_counts 展开成一行文本，避免 Excel 中出现复杂对象。
    csv_rows = [
        {
            **serialise_result_row(row),
            "class_counts": build_class_count_text(row["class_counts"]),
        }
        for row in per_image_rows
    ]
    write_csv(detail_csv, csv_rows)
    write_json(
        detail_json,
        {
            "model_name": model_info.model_name,
            "weight_path": str(model_info.weight_path),
            "source_dir": str(source_dir),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": [serialise_result_row(row) for row in per_image_rows],
        },
    )

    avg_wall_time_ms = (wall_time / len(image_paths)) * 1000 if image_paths else 0.0

    # model_summary 是后续单模型 Markdown 和总汇总 Markdown 的共同数据源。
    model_summary = {
        "model_name": model_info.model_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_load_finished": model_load_finished,
        "train_dir": str(model_info.train_dir),
        "weight_path": str(model_info.weight_path),
        "weight_size_mb": model_info.weight_size_mb,
        "source_dir": str(source_dir),
        "batch_size": batch,
        "prediction_dir": str(prediction_dir),
        "image_count": len(image_paths),
        "detected_images": detected_images,
        "total_detections": sum(image_level_detections),
        "avg_detections_per_image": mean(image_level_detections) if image_level_detections else 0.0,
        "avg_confidence": mean(image_level_confidences) if image_level_confidences else 0.0,
        "wall_time_seconds": wall_time,
        "avg_wall_time_ms": avg_wall_time_ms,
        "avg_preprocess_ms": mean(preprocess_times) if preprocess_times else 0.0,
        "avg_inference_ms": mean(inference_times) if inference_times else 0.0,
        "avg_postprocess_ms": mean(postprocess_times) if postprocess_times else 0.0,
        "avg_speed_total_ms": mean(total_speed_times) if total_speed_times else 0.0,
        "predicted_class_counts": dict(sorted(predicted_class_counts.items())),
        "detail_csv": str(detail_csv),
        "detail_json": str(detail_json),
        "detail_md": str(detail_md),
        "detail_md_rel": safe_rel(detail_md, run_dir),
        "evaluation_metrics": evaluation_metrics,
        "train_best_metrics": model_info.best_metrics,
        "args": model_info.args,
    }

    return {
        "summary": model_summary,
        "detail_rows": per_image_rows,
        "detail_md": detail_md,
    }


def main() -> None:
    args = parse_args()

    # 把所有输入路径都先 resolve，后面生成日志和报告时更直观，也能减少相对路径歧义。
    project_root = args.project_root.resolve()
    source_dir = args.source.resolve()
    weights_root = args.weights_root.resolve()
    output_root = args.output_root.resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not weights_root.exists():
        raise FileNotFoundError(f"Weights root does not exist: {weights_root}")

    image_paths = collect_images(source_dir, args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {source_dir}")

    weights = collect_weights(weights_root, args.pattern, args.weight_pattern)
    if not weights:
        raise FileNotFoundError(
            f"No weights found in {weights_root} matching {args.pattern}/weights/{args.weight_pattern}"
        )

    # 每次运行生成独立时间戳目录，避免多次测试互相覆盖。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"compare_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 100)
    print("YOLOv11 batch prediction comparison")
    print(f"project_root : {project_root}")
    print(f"source_dir   : {source_dir}")
    print(f"weights_root : {weights_root}")
    print(f"output_root  : {run_dir}")
    print(f"images       : {len(image_paths)}")
    print(f"weights      : {len(weights)}")
    print(f"batch        : {args.batch}")
    print("=" * 100)

    ground_truth: dict[str, Any] = {}
    label_dir = source_dir.parent / "all_labels"
    class_name_map = load_class_name_map(project_root / "datasets" / "construction-ppe" / "data.yaml")
    if label_dir.exists():
        # 标注统计只需要做一次，后面所有模型共用这份对照数据。
        ground_truth = count_label_distribution(label_dir, class_name_map)
        if ground_truth:
            ground_truth["label_dir"] = str(label_dir)

    # 这里构造一份符合 YOLO 标准目录结构的临时评估数据视图，供 model.val() 直接使用。
    eval_data_yaml = prepare_eval_dataset_view(
        run_dir=run_dir,
        source_dir=source_dir,
        image_paths=image_paths,
        label_dir=label_dir,
        class_name_map=class_name_map,
    )

    summaries: list[dict[str, Any]] = []

    for index, weight_info in enumerate(weights, start=1):
        print(f"\n[{index}/{len(weights)}] processing {weight_info.model_name}")
        result_bundle = predict_one_model(
            model_info=weight_info,
            image_paths=image_paths,
            source_dir=source_dir,
            run_dir=run_dir,
            eval_data_yaml=eval_data_yaml,
            class_name_map=class_name_map,
            ground_truth=ground_truth,
            conf=args.conf,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

        model_summary = result_bundle["summary"]
        detail_rows = result_bundle["detail_rows"]

        detail_md_path = result_bundle["detail_md"]
        # 先生成每个模型自己的详细 Markdown，最后再生成总汇总对比报告。
        detail_md_text = build_model_detail_markdown(
            run_dir=run_dir,
            model_summary=model_summary,
            detail_rows=detail_rows,
            ground_truth=ground_truth,
        )
        detail_md_path.write_text(detail_md_text, encoding="utf-8")

        summaries.append(model_summary)
        print(
            f"[{weight_info.model_name}] done: "
            f"P={model_summary['evaluation_metrics']['overall']['precision']:.4f}, "
            f"R={model_summary['evaluation_metrics']['overall']['recall']:.4f}, "
            f"mAP50={model_summary['evaluation_metrics']['overall']['mAP50']:.4f}, "
            f"mAP50-95={model_summary['evaluation_metrics']['overall']['mAP50_95']:.4f}, "
            f"total_det={model_summary['total_detections']}, "
            f"avg_infer={model_summary['avg_inference_ms']:.2f}ms, "
            f"wall_time={model_summary['wall_time_seconds']:.2f}s"
        )

        # 一个模型处理结束后主动释放资源，避免后续大模型叠加占用显存。
        clear_torch_memory()

    summary_md = run_dir / "summary_comparison.md"
    summary_json = run_dir / "summary_comparison.json"

    # 总报告面向横向比较，JSON 则方便后续你再做二次分析。
    summary_md.write_text(
        build_summary_markdown(
            run_dir=run_dir,
            summaries=summaries,
            source_dir=source_dir,
            ground_truth=ground_truth,
        ),
        encoding="utf-8",
    )
    write_json(summary_json, summaries)

    print("\n" + "=" * 100)
    print("Comparison finished")
    print(f"Summary report : {summary_md}")
    print(f"Summary json   : {summary_json}")
    print("=" * 100)


if __name__ == "__main__":
    main()
