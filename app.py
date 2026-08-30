import gradio as gr
from PIL import Image
from collections import Counter
from pathlib import Path
import csv
import json
import os
import tempfile
import time
import uuid

# 默认权重路径
DEFAULT_MODEL_PATH = "./examples/best.pt"


TABLE_HEADERS = ["序号", "类别", "置信度", "x1", "y1", "x2", "y2", "宽", "高"]
EMPTY_SUMMARY = "暂无检测结果。"
WAITING_STATUS = "状态：请上传图片后点击 **开始检测**。"

# 统一导入逻辑
try:
    from ultralytics import YOLO, RTDETR
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    print("❌ 严重错误: 未安装 ultralytics 库。请运行 'pip install ultralytics'")

def get_model_path(model_file):
    """兼容 Gradio 上传文件对象和 Examples 中的字符串路径。"""
    if model_file is None:
        return None
    return model_file.name if hasattr(model_file, "name") else str(model_file)

def get_class_name(names, class_id):
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return names[class_id]
    return str(class_id)

def build_detection_rows(result):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = getattr(result, "names", {}) or {}
    xyxy = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()

    rows = []
    for idx, (box, confidence, class_id) in enumerate(zip(xyxy, confidences, classes), start=1):
        class_id = int(class_id)
        x1, y1, x2, y2 = [round(float(value), 2) for value in box]
        rows.append([
            idx,
            get_class_name(names, class_id),
            round(float(confidence), 4),
            x1,
            y1,
            x2,
            y2,
            round(x2 - x1, 2),
            round(y2 - y1, 2),
        ])
    return rows

def make_summary(rows, model_type, model_path, conf_threshold, total_seconds, speed):
    model_name = Path(model_path).name if model_path else "未选择"
    class_counter = Counter(row[1] for row in rows)
    class_summary = "，".join(f"{name}: {count}" for name, count in class_counter.items()) or "无"
    max_confidence = max((row[2] for row in rows), default=0)

    speed = speed or {}
    preprocess_ms = float(speed.get("preprocess", 0))
    inference_ms = float(speed.get("inference", 0))
    postprocess_ms = float(speed.get("postprocess", 0))

    return "\n".join([
        "### 检测摘要",
        f"- 模型：`{model_type}` / `{model_name}`",
        f"- 置信度阈值：`{conf_threshold:.2f}`",
        f"- 检测目标数：**{len(rows)}**",
        f"- 类别分布：{class_summary}",
        f"- 最高置信度：`{max_confidence:.4f}`",
        f"- 总耗时：`{total_seconds:.3f}s`",
        f"- 推理耗时：预处理 `{preprocess_ms:.1f}ms`，模型 `{inference_ms:.1f}ms`，后处理 `{postprocess_ms:.1f}ms`",
    ])

def create_export_files(output_image, rows, summary):
    export_dir = Path(tempfile.mkdtemp(prefix="rtdetr_demo_"))
    stem = f"detection_{uuid.uuid4().hex[:8]}"
    image_path = export_dir / f"{stem}.png"
    csv_path = export_dir / f"{stem}.csv"
    json_path = export_dir / f"{stem}.json"

    Image.fromarray(output_image).save(image_path)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(TABLE_HEADERS)
        writer.writerows(rows)

    detections = [dict(zip(TABLE_HEADERS, row)) for row in rows]
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump({"summary": summary, "detections": detections}, json_file, ensure_ascii=False, indent=2)

    return [str(image_path), str(csv_path), str(json_path)]

def patch_legacy_rtdetr_decoder(model):
    """为旧版 RT-DETR 权重补齐新版 decoder 推理需要的 DFL 属性。"""
    patched = False
    model_modules = getattr(getattr(model, "model", None), "modules", None)
    if model_modules is None:
        return patched

    for module in model.model.modules():
        if module.__class__.__name__ != "RTDETRDecoder":
            continue

        defaults = {
            "reg_max": 16,
            "dfl_scale": 1.0,
            "dfl_loss_gain": 1.5,
            "dfl": None,
        }
        for attr_name, default_value in defaults.items():
            if not hasattr(module, attr_name):
                setattr(module, attr_name, default_value)
                patched = True

    return patched

def detect_objects(image, model_file, conf_threshold, model_type):
    """
    通用检测函数：适配 YOLO 和 RT-DETR
    """
    if not HAS_ULTRALYTICS:
        return image, "状态：未安装 ultralytics 库，无法执行检测。", EMPTY_SUMMARY, [], None

    if image is None:
        return None, "状态：请先上传待检测图片。", EMPTY_SUMMARY, [], None

    # --- 1. 模型加载逻辑 ---
    # 如果用户没有上传模型文件，则尝试使用默认路径
    if model_file is None:
        if os.path.exists(DEFAULT_MODEL_PATH):
            model_path = DEFAULT_MODEL_PATH
            print(f"ℹ️ 未上传模型，正在使用默认模型: {model_path}")
        else:
            print(f"⚠️ 警告: 未上传模型且未找到默认模型文件 ({DEFAULT_MODEL_PATH})")
            return image, f"状态：未上传模型，且未找到默认模型文件 `{DEFAULT_MODEL_PATH}`。", EMPTY_SUMMARY, [], None
    else:
        model_path = get_model_path(model_file)

    try:
        start_time = time.perf_counter()
        print(f"🔄 正在加载 {model_type} 模型: {model_path}")

        # 根据用户选择的类型加载不同的模型类
        if model_type == "YOLO":
            model = YOLO(model_path)
        elif model_type == "RT-DETR":
            model = RTDETR(model_path)
        else:
            # 默认回退
            model = RTDETR(model_path)

        patched_legacy_decoder = patch_legacy_rtdetr_decoder(model)
        if patched_legacy_decoder:
            print("ℹ️ 已为旧版 RT-DETR 权重补齐 decoder DFL 兼容属性")

        # --- 2. 统一推理逻辑 ---
        results = model.predict(source=image, conf=conf_threshold, verbose=False)
        result = results[0]
        total_seconds = time.perf_counter() - start_time

        # --- 3. 绘图逻辑 ---
        res_plotted = result.plot()
        output_image = res_plotted[..., ::-1].copy() # BGR -> RGB

        rows = build_detection_rows(result)
        summary = make_summary(
            rows=rows,
            model_type=model_type,
            model_path=model_path,
            conf_threshold=conf_threshold,
            total_seconds=total_seconds,
            speed=getattr(result, "speed", {}),
        )
        export_files = create_export_files(output_image, rows, summary)
        if rows:
            status = f"状态：检测完成，共发现 {len(rows)} 个目标，结果文件已生成。"
        else:
            status = "状态：检测完成，未发现高于当前置信度阈值的目标，结果文件已生成。"
        if patched_legacy_decoder:
            status += " 已应用旧版 RT-DETR 权重兼容补丁。"

        return output_image, status, summary, rows, export_files

    except Exception as e:
        print(f"❌ 推理过程中出错: {e}")
        return image, f"状态：推理过程中出错：`{e}`", EMPTY_SUMMARY, [], None

# --- 界面布局代码 ---

custom_css = """
.gradio-container {background-color: #ffffff}
h1 {text-align: center; margin-bottom: 10px}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🚀 通用目标检测 DEMO")
    gr.Markdown("<p style='text-align:center; margin-bottom:20px; color:#666'>系统默认使用 <strong>RT-DETR</strong> 模型。您可以直接上传图片开始检测，或在下方折叠栏中更换设置。</p>")

    with gr.Row():
        # --- 左侧列：输入区域 ---
        with gr.Column(scale=1):
            input_image = gr.Image(label="上传待检测图片", type="pil", height=400)

            # --- 改进点：折叠栏设计 ---
            with gr.Accordion("🛠️ 模型设置 (可选)", open=False):
                # 优先默认选择 RT-DETR
                model_type_selector = gr.Radio(
                    choices=["RT-DETR", "YOLO"],
                    value="RT-DETR",
                    label="1. 选择模型架构类型"
                )

                # 上传功能
                model_input = gr.File(
                    label="2. 上传自定义权重 (.pt)",
                    file_types=[".pt"]
                )

            # 置信度滑块放在折叠栏外，因为这是常用调整项
            conf_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.25,
                label="置信度阈值 (Confidence)"
            )

            # 按钮区域
            with gr.Row():
                clear_btn = gr.Button("清空 (Clear)", variant="secondary")
                submit_btn = gr.Button("开始检测 (Submit)", variant="primary")

        # --- 右侧列：输出区域 ---
        with gr.Column(scale=1):
            output_image = gr.Image(label="检测结果", interactive=False, height=400)
            status_text = gr.Markdown(WAITING_STATUS)
            summary_text = gr.Markdown(EMPTY_SUMMARY)

    with gr.Row():
        with gr.Column(scale=3):
            result_table = gr.Dataframe(
                headers=TABLE_HEADERS,
                label="检测明细",
                interactive=False
            )
        with gr.Column(scale=1):
            export_output = gr.File(
                label="下载检测结果",
                file_count="multiple"
            )

    # --- 底部：示例图片 ---
    gr.Examples(
        examples=[
            ["./examples/example1.jpg", "./examples/rtdetr-r18.pt", 0.25, "RT-DETR"],
            ["./examples/example1.jpg", "./examples/yolov8m.pt", 0.25, "YOLO"],
            ["./examples/example1.jpg", "./examples/best.pt", 0.25, "RT-DETR"],
            ["./examples/example5.jpg", "./examples/rtdetr-r18.pt", 0.5, "RT-DETR"],
            ["./examples/example5.jpg", "./examples/yolov8m.pt", 0.5, "YOLO"],
            ["./examples/example5.jpg", "./examples/best.pt", 0.5, "RT-DETR"],
        ],
        inputs=[input_image, model_input, conf_slider, model_type_selector],
        outputs=[output_image, status_text, summary_text, result_table, export_output],
        fn=detect_objects,
        cache_examples=False,
        label="快速示例"
    )

    # --- 交互逻辑绑定 ---
    submit_btn.click(
        fn=detect_objects,
        inputs=[input_image, model_input, conf_slider, model_type_selector],
        outputs=[output_image, status_text, summary_text, result_table, export_output]
    )

    clear_btn.click(
        fn=lambda: (None, None, 0.25, "RT-DETR", None, WAITING_STATUS, EMPTY_SUMMARY, [], None),
        inputs=None,
        outputs=[
            input_image,
            model_input,
            conf_slider,
            model_type_selector,
            output_image,
            status_text,
            summary_text,
            result_table,
            export_output,
        ]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
