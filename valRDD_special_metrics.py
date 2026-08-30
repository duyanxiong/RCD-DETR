import warnings
warnings.filterwarnings('ignore')

import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.data.utils import check_det_dataset, img2label_paths
from ultralytics.utils.torch_utils import model_info

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =========================
# 基础验证配置
# =========================
MODEL_PATH = 'runs/train/exp12/weights/best.pt'
DATA_YAML = 'dataset/RDD2022dataSplit.yaml'
SPLIT = 'test'
IMGSZ = 640
BATCH = 8
DEVICE = '6'
PROJECT = 'runs/val'
NAME = 'exp'

# =========================
# 裂缝专项评估配置
# =========================
ENABLE_SPECIAL_EVAL = True

# 高长宽比：AR = max(w / h, h / w)
# 论文中可根据数据分布改成 3、5、8 等；建议先统计 GT 分布后固定阈值。
HIGH_AR_THRESHOLD = 5.0

# 小尺度目标：默认采用 COCO small 的面积界限 area < 32^2 px^2（在原图坐标中计算）。
# 如果你的 RDD 图像分辨率明显高于 COCO，建议同时报告阈值依据，或改为面积占比方案。
SMALL_AREA_MAX = 32.0 ** 2

# 专项 AP 必须保留低置信度预测，不能直接用 0.25，否则会截断 PR 曲线。
SPECIAL_PRED_CONF = 0.001
SPECIAL_MAX_DET = 300
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)

IMAGE_EXTS = {
    '.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp', '.pfm', '.heic'
}
EPS = 1e-16


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


def box_iou_np(box, boxes):
    """计算单个 xyxy box 与 N 个 xyxy boxes 的 IoU。"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area1 = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area2 = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
    return inter / (area1 + area2 - inter + EPS)


def box_geometry_masks(boxes_xyxy):
    """返回每个 box 的 high-AR / small 掩码。boxes 使用原图像素坐标。"""
    if boxes_xyxy.size == 0:
        empty = np.zeros((0,), dtype=bool)
        return {'high_ar': empty, 'small': empty}

    w = np.clip(boxes_xyxy[:, 2] - boxes_xyxy[:, 0], EPS, None)
    h = np.clip(boxes_xyxy[:, 3] - boxes_xyxy[:, 1], EPS, None)
    ar = np.maximum(w / h, h / w)
    area = w * h

    return {
        'high_ar': ar >= HIGH_AR_THRESHOLD,
        'small': area < SMALL_AREA_MAX,
    }


def load_yolo_gt(label_path, image_shape):
    """
    读取 YOLO detect 标签：class x_center y_center width height（归一化），
    并转换成原图 xyxy 像素坐标。
    """
    h, w = image_shape
    label_path = Path(label_path)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 4), dtype=np.float32)

    classes, boxes = [], []
    with label_path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(
                    f'{label_path} 第 {line_no} 行不是 YOLO detection 的 5 列格式：{line.strip()}'
                )

            cls_id, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2.0) * w
            y1 = (yc - bh / 2.0) * h
            x2 = (xc + bw / 2.0) * w
            y2 = (yc + bh / 2.0) * h
            boxes.append([
                np.clip(x1, 0, w), np.clip(y1, 0, h),
                np.clip(x2, 0, w), np.clip(y2, 0, h)
            ])
            classes.append(int(cls_id))

    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _resolve_candidate(path_text, base_dir=None, data_root=None):
    p = Path(path_text).expanduser()
    if p.exists():
        return p.resolve()
    if base_dir is not None:
        p2 = (Path(base_dir) / p).resolve()
        if p2.exists():
            return p2
    if data_root is not None:
        p3 = (Path(data_root) / p).resolve()
        if p3.exists():
            return p3
    return p


def collect_image_files(source, data_root=None):
    """把 Ultralytics data yaml 中的 split 路径展开成图像文件列表。"""
    output = []

    def add_source(src, parent=None):
        if src is None:
            return
        if isinstance(src, (list, tuple)):
            for item in src:
                add_source(item, parent)
            return

        src = str(src)
        if any(ch in src for ch in '*?[]'):
            for match in sorted(glob.glob(src, recursive=True)):
                add_source(match, parent)
            return

        p = _resolve_candidate(src, base_dir=parent, data_root=data_root)
        if p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    output.append(str(f.resolve()))
        elif p.is_file() and p.suffix.lower() == '.txt':
            with p.open('r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        add_source(line, p.parent)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            output.append(str(p.resolve()))

    add_source(source)

    # 去重但保持顺序
    seen = set()
    unique = []
    for p in output:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def compute_ap_101(recall, precision):
    """COCO 风格 101 个 recall 阈值上的插值 AP。"""
    if recall.size == 0:
        return 0.0

    # 对每个 recall 阈值 r，取所有 recall >= r 的最大 precision，
    # 再对 101 个阈值求均值。完美检测应得到 AP=1.0。
    ap_points = []
    for r in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= r]
        ap_points.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(ap_points))


def evaluate_one_class(records, gt_target_by_image, gt_ignore_by_image, num_gt):
    """
    对一个类别、一个专项 subset 计算 P/R/F1/AP。

    records: [(conf, image_id, box_xyxy, pred_is_in_subset), ...]
    gt_target_by_image: 属于该 subset 的 GT
    gt_ignore_by_image: 不属于该 subset 的同类 GT（用于忽略正常检测，避免误计 FP）
    """
    if num_gt == 0:
        return None

    records = sorted(records, key=lambda x: x[0], reverse=True)
    aps = []
    p_best = r_best = f1_best = 0.0

    for iou_index, iou_thr in enumerate(IOU_THRESHOLDS):
        matched = {
            img_id: np.zeros(len(boxes), dtype=bool)
            for img_id, boxes in gt_target_by_image.items()
        }

        tp_flags, fp_flags, confs = [], [], []

        for conf, img_id, pred_box, pred_in_subset in records:
            target_boxes = gt_target_by_image.get(img_id, np.empty((0, 4), dtype=np.float32))
            target_used = matched.get(img_id, np.empty((0,), dtype=bool))

            # 1) 优先与该 subset 的未匹配 GT 做一对一匹配
            is_tp = False
            if len(target_boxes):
                ious = box_iou_np(pred_box, target_boxes)
                valid = np.where(~target_used)[0]
                if valid.size:
                    best_local = valid[np.argmax(ious[valid])]
                    if ious[best_local] >= iou_thr:
                        matched[img_id][best_local] = True
                        is_tp = True

            if is_tp:
                tp_flags.append(1.0)
                fp_flags.append(0.0)
                confs.append(conf)
                continue

            # 2) 若预测实际上正确检测到了“非本 subset”的同类 GT，则在专项评估中忽略
            ignore_boxes = gt_ignore_by_image.get(img_id, np.empty((0, 4), dtype=np.float32))
            if len(ignore_boxes) and np.max(box_iou_np(pred_box, ignore_boxes), initial=0.0) >= iou_thr:
                continue

            # 3) COCO area-range 类似规则：预测框自身不属于该 subset，则忽略
            if not pred_in_subset:
                continue

            # 4) 剩余预测作为该 subset 的 FP
            tp_flags.append(0.0)
            fp_flags.append(1.0)
            confs.append(conf)

        if len(tp_flags) == 0:
            aps.append(0.0)
            continue

        tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
        fp = np.cumsum(np.asarray(fp_flags, dtype=np.float64))
        recall = tp / (num_gt + EPS)
        precision = tp / (tp + fp + EPS)
        aps.append(compute_ap_101(recall, precision))

        # P/R/F1 采用 IoU=0.50 下 PR 曲线上最佳 F1 点，接近 Ultralytics 的展示方式
        if iou_index == 0:
            f1_curve = 2 * precision * recall / (precision + recall + EPS)
            best = int(np.argmax(f1_curve))
            p_best = float(precision[best])
            r_best = float(recall[best])
            f1_best = float(f1_curve[best])

    aps = np.asarray(aps, dtype=np.float64)
    ap75_idx = int(np.where(np.isclose(IOU_THRESHOLDS, 0.75))[0][0])
    return {
        'precision': p_best,
        'recall': r_best,
        'f1': f1_best,
        'ap50': float(aps[0]),
        'ap75': float(aps[ap75_idx]),
        'map50_95': float(aps.mean()),
    }


def evaluate_special_targets(model, data_yaml, split, imgsz, batch, device, class_names):
    """二次低置信度预测，计算 high-AR / small 两类专项指标。"""
    data_cfg = check_det_dataset(data_yaml)
    split_source = data_cfg.get(split)
    if split_source is None:
        raise KeyError(f'data yaml 中没有 split={split!r}。可用键：{list(data_cfg.keys())}')

    data_root = data_cfg.get('path', None)
    image_files = collect_image_files(split_source, data_root=data_root)
    if not image_files:
        raise FileNotFoundError(f'无法从 {split_source} 解析到 {split} 图像。')

    label_files = img2label_paths(image_files)
    num_classes = len(class_names)

    # subset_data[subset][class_id] = {...}
    subset_data = {}
    for subset in ('high_ar', 'small'):
        subset_data[subset] = []
        for _ in range(num_classes):
            subset_data[subset].append({
                'records': [],
                'gt_target_by_image': defaultdict(lambda: np.empty((0, 4), dtype=np.float32)),
                'gt_ignore_by_image': defaultdict(lambda: np.empty((0, 4), dtype=np.float32)),
                'num_gt': 0,
                'num_pred_geom': 0,
            })

    # 重要：不要把 image_files 整个 list 直接传给 model.predict。
    # 某些 Ultralytics/RT-DETR 版本会把 Python list 当成一个内存 source，
    # 从而一次性把大量图像堆成一个超大 batch；即使传入 batch=1/8 也可能不会
    # 按预期切分，最终在第一层卷积处 OOM。这里固定逐图推理，专项指标仍然在
    # 全测试集上统一累计，因此不会改变 P/R/AP 的统计定义。
    total_images = len(image_files)

    for image_id, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        pred_results = model.predict(
            source=image_path,
            imgsz=imgsz,
            batch=1,
            device=device,
            conf=SPECIAL_PRED_CONF,
            max_det=SPECIAL_MAX_DET,
            stream=False,
            verbose=False,
            save=False,
        )

        if not pred_results:
            # 理论上单张图片也会返回一个 Results；这里做防御性处理。
            continue

        pred_result = pred_results[0]

        if (image_id + 1) % 100 == 0 or (image_id + 1) == total_images:
            print(f'  专项评估进度: {image_id + 1}/{total_images}')

        h, w = pred_result.orig_shape
        gt_cls, gt_boxes = load_yolo_gt(label_path, (h, w))
        gt_masks = box_geometry_masks(gt_boxes)

        if pred_result.boxes is None or len(pred_result.boxes) == 0:
            pred_cls = np.empty((0,), dtype=np.int64)
            pred_conf = np.empty((0,), dtype=np.float32)
            pred_boxes = np.empty((0, 4), dtype=np.float32)
        else:
            pred_cls = pred_result.boxes.cls.detach().cpu().numpy().astype(np.int64)
            pred_conf = pred_result.boxes.conf.detach().cpu().numpy().astype(np.float32)
            pred_boxes = pred_result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        pred_masks = box_geometry_masks(pred_boxes)

        for subset in ('high_ar', 'small'):
            for cls_id in range(num_classes):
                d = subset_data[subset][cls_id]

                gt_same_class = gt_cls == cls_id
                target_mask = gt_same_class & gt_masks[subset]
                ignore_mask = gt_same_class & (~gt_masks[subset])
                target_boxes = gt_boxes[target_mask]
                ignore_boxes = gt_boxes[ignore_mask]

                d['gt_target_by_image'][image_id] = target_boxes
                d['gt_ignore_by_image'][image_id] = ignore_boxes
                d['num_gt'] += len(target_boxes)

                pred_same_class_idx = np.where(pred_cls == cls_id)[0]
                for pi in pred_same_class_idx:
                    pred_in_subset = bool(pred_masks[subset][pi])
                    d['records'].append((
                        float(pred_conf[pi]), image_id, pred_boxes[pi].copy(), pred_in_subset
                    ))
                    if pred_in_subset:
                        d['num_pred_geom'] += 1

    table = PrettyTable()
    table.title = (
        f"Special Crack Metrics | High-AR >= {HIGH_AR_THRESHOLD:g} | "
        f"Small area < {SMALL_AREA_MAX:.0f}px^2"
    )
    table.field_names = [
        'Subset', 'Class Name', 'GT Count', 'Pred Count',
        'Precision', 'Recall', 'F1-Score', 'AP50', 'AP75', 'mAP50-95'
    ]

    summary = {}
    subset_display = {
        'high_ar': f'High-AR(>={HIGH_AR_THRESHOLD:g})',
        'small': f'Small(<{SMALL_AREA_MAX:.0f}px²)',
    }

    for subset in ('high_ar', 'small'):
        class_metrics = []
        total_gt = 0
        total_pred_geom = 0

        for cls_id, cls_name in enumerate(class_names):
            d = subset_data[subset][cls_id]
            metrics = evaluate_one_class(
                d['records'], d['gt_target_by_image'], d['gt_ignore_by_image'], d['num_gt']
            )
            total_gt += d['num_gt']
            total_pred_geom += d['num_pred_geom']

            if metrics is None:
                table.add_row([
                    subset_display[subset], cls_name, d['num_gt'], d['num_pred_geom'],
                    '-', '-', '-', '-', '-', '-'
                ])
                continue

            class_metrics.append(metrics)
            table.add_row([
                subset_display[subset], cls_name, d['num_gt'], d['num_pred_geom'],
                f"{metrics['precision']:.4f}", f"{metrics['recall']:.4f}", f"{metrics['f1']:.4f}",
                f"{metrics['ap50']:.4f}", f"{metrics['ap75']:.4f}", f"{metrics['map50_95']:.4f}"
            ])

        if class_metrics:
            mean_metrics = {
                key: float(np.mean([m[key] for m in class_metrics]))
                for key in class_metrics[0].keys()
            }
            summary[subset] = mean_metrics
            table.add_row([
                subset_display[subset], 'all(类别平均)', total_gt, total_pred_geom,
                f"{mean_metrics['precision']:.4f}", f"{mean_metrics['recall']:.4f}", f"{mean_metrics['f1']:.4f}",
                f"{mean_metrics['ap50']:.4f}", f"{mean_metrics['ap75']:.4f}", f"{mean_metrics['map50_95']:.4f}"
            ])
        else:
            summary[subset] = None
            table.add_row([
                subset_display[subset], 'all(类别平均)', total_gt, total_pred_geom,
                '-', '-', '-', '-', '-', '-'
            ])

    return table, summary


if __name__ == '__main__':
    model = RTDETR(MODEL_PATH)
    result = model.val(
        data=DATA_YAML,
        split=SPLIT,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        # save_json=True,  # 如果需要官方 COCO metrics 可开启
        project=PROJECT,
        name=NAME,
    )

    if model.task == 'detect':
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = (
            preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image
        )

        n_l, n_p, n_g, flops = model_info(model.model)

        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)

        model_info_table = PrettyTable()
        model_info_table.title = 'Model Info'
        model_info_table.field_names = [
            'GFLOPs', 'Parameters', '前处理时间/一张图', '推理时间/一张图', '后处理时间/一张图',
            'FPS(前处理+模型推理+后处理)', 'FPS(推理)', 'Model File Size'
        ]
        model_info_table.add_row([
            f'{flops:.1f}', f'{n_p:,}',
            f'{preprocess_time_per_image / 1000:.6f}s',
            f'{inference_time_per_image / 1000:.6f}s',
            f'{postprocess_time_per_image / 1000:.6f}s',
            f'{1000 / all_time_per_image:.2f}',
            f'{1000 / inference_time_per_image:.2f}',
            f'{get_weight_size(MODEL_PATH)}MB'
        ])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = 'Model Metrice'
        model_metrice_table.field_names = [
            'Class Name', 'Precision', 'Recall', 'F1-Score', 'mAP50', 'mAP75', 'mAP50-95'
        ]
        for idx, cls_name in enumerate(model_names):
            model_metrice_table.add_row([
                cls_name,
                f'{result.box.p[idx]:.4f}',
                f'{result.box.r[idx]:.4f}',
                f'{result.box.f1[idx]:.4f}',
                f'{result.box.ap50[idx]:.4f}',
                f'{result.box.all_ap[idx, 5]:.4f}',
                f'{result.box.ap[idx]:.4f}'
            ])
        model_metrice_table.add_row([
            'all(平均数据)',
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f'{np.mean(result.box.f1):.4f}',
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f'{np.mean(result.box.all_ap[:, 5]):.4f}',
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)

        special_table = None
        if ENABLE_SPECIAL_EVAL:
            print('\n正在计算裂缝专项指标（会进行一次额外的低置信度预测，以构建完整 PR/AP 曲线）...')

            # 重新加载一个新的模型实例，避免复用 val() 后已 fuse 的模型
            special_model = RTDETR(MODEL_PATH)

            special_table, _ = evaluate_special_targets(
                model=special_model,
                data_yaml=DATA_YAML,
                split=SPLIT,
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                class_names=model_names,
            )
            print(special_table)

        output_txt = Path(result.save_dir) / 'paper_data.txt'
        with output_txt.open('w+', encoding='utf-8') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))
            if special_table is not None:
                f.write('\n')
                f.write(str(special_table))
                f.write('\n\n')
                f.write(
                    'Special metric definition:\n'
                    f'- High-AR: max(w/h, h/w) >= {HIGH_AR_THRESHOLD:g}\n'
                    f'- Small: GT/pred box area < {SMALL_AREA_MAX:.0f} px^2 in original-image coordinates\n'
                    f'- AP: IoU 0.50:0.95, step 0.05; 101-point interpolation\n'
                    '- P/R/F1: best-F1 operating point on the IoU=0.50 PR curve\n'
                )

        print('-' * 20, f'结果已保存至 {output_txt} ...', '-' * 20)
