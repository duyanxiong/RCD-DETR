"""Public RT-DETR evaluation entry point.

This script evaluates the release TorchScript artifact (or the original .pt file
inside the private source tree) and writes a human-readable report plus JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from prettytable import PrettyTable


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE_MODEL = PROJECT_DIR / "models" / "best.torchscript"
DEFAULT_PRIVATE_MODEL = PROJECT_DIR.parent / "examples" / "best.pt"
DEFAULT_DATA = PROJECT_DIR / "configs" / "rdd2022.yaml"


def default_model_path() -> Path:
    """Prefer the public artifact, while keeping local private-tree use convenient."""
    return DEFAULT_RELEASE_MODEL if DEFAULT_RELEASE_MODEL.exists() else DEFAULT_PRIVATE_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the released RT-DETR model and export paper-style metrics."
    )
    parser.add_argument("--model", type=Path, default=default_model_path(), help=".torchscript or private .pt model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Ultralytics dataset YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=640, help="square evaluation image size")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu; default lets Ultralytics choose")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--name", default="val")
    parser.add_argument("--save-json", action="store_true", help="also save COCO-format predictions")
    parser.add_argument("--no-plots", action="store_true", help="disable validation plots")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_release_metadata(model_path: Path) -> dict[str, Any]:
    candidates = (
        model_path.with_suffix(".metadata.json"),
        model_path.parent / "model_metadata.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(value) for _, value in sorted(names.items(), key=lambda item: int(item[0]))]
    return [str(value) for value in names] if isinstance(names, (list, tuple)) else []


def warn_on_class_order_mismatch(data_path: Path, metadata: dict[str, Any]) -> None:
    import yaml

    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    dataset_names = normalize_names(dataset.get("names"))
    model_names = normalize_names(metadata.get("names"))
    if dataset_names and model_names and dataset_names != model_names:
        print(
            "WARNING: dataset class order differs from the class order embedded in the model.\n"
            f"  dataset: {dataset_names}\n"
            f"  model:   {model_names}\n"
            "Numeric evaluation still uses class IDs, but per-class labels may be misleading. "
            "Confirm the label-ID mapping before publishing results.",
            file=sys.stderr,
        )


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: Any, digits: int = 4) -> str:
    number = finite_number(value)
    return "-" if number is None else f"{number:.{digits}f}"


def metric_value(values: Any, position: int) -> float | None:
    try:
        return finite_number(values[position])
    except (IndexError, TypeError):
        return None


def ordered_names(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, dict):
        return sorted((int(index), str(name)) for index, name in names.items())
    return [(index, str(name)) for index, name in enumerate(names)]


def apply_legacy_checkpoint_compatibility(model: object) -> None:
    """Allow the private source tree to evaluate older non-DFL .pt checkpoints."""
    pytorch_model = getattr(model, "model", model)
    modules = getattr(pytorch_model, "modules", None)
    if modules is None:
        return
    for module in modules():
        if module.__class__.__name__ == "RTDETRDecoder":
            if not hasattr(module, "use_dfl"):
                module.use_dfl = False
            if not hasattr(module, "reg_max"):
                module.reg_max = 16
            if not hasattr(module, "dfl_scale"):
                module.dfl_scale = 1.0


def build_class_rows(result: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    box = result.box
    class_indices = getattr(box, "ap_class_index", None)
    if class_indices is None or len(class_indices) == 0:
        class_indices = np.arange(len(getattr(box, "ap", [])))
    positions = {int(class_index): position for position, class_index in enumerate(class_indices)}

    rows: list[dict[str, Any]] = []
    for class_index, class_name in ordered_names(result.names):
        position = positions.get(class_index)
        all_ap = getattr(box, "all_ap", [])
        ap75 = None
        if position is not None:
            try:
                ap75 = finite_number(all_ap[position, 5])
            except (IndexError, TypeError):
                pass
        rows.append(
            {
                "class_id": class_index,
                "class_name": class_name,
                "precision": metric_value(getattr(box, "p", []), position) if position is not None else None,
                "recall": metric_value(getattr(box, "r", []), position) if position is not None else None,
                "f1": metric_value(getattr(box, "f1", []), position) if position is not None else None,
                "map50": metric_value(getattr(box, "ap50", []), position) if position is not None else None,
                "map75": ap75,
                "map50_95": metric_value(getattr(box, "ap", []), position) if position is not None else None,
            }
        )

    results_dict = result.results_dict
    all_ap = np.asarray(getattr(box, "all_ap", []))
    overall = {
        "class_name": "all (mean)",
        "precision": finite_number(results_dict.get("metrics/precision(B)")),
        "recall": finite_number(results_dict.get("metrics/recall(B)")),
        "f1": finite_number(np.mean(getattr(box, "f1", []))) if len(getattr(box, "f1", [])) else None,
        "map50": finite_number(results_dict.get("metrics/mAP50(B)")),
        "map75": finite_number(np.mean(all_ap[:, 5])) if all_ap.ndim == 2 and all_ap.shape[1] > 5 else None,
        "map50_95": finite_number(results_dict.get("metrics/mAP50-95(B)")),
    }
    return rows, overall


def make_model_table(model_path: Path, result: Any, metadata: dict[str, Any]) -> tuple[PrettyTable, dict[str, Any]]:
    speed = result.speed
    preprocess = finite_number(speed.get("preprocess")) or 0.0
    inference = finite_number(speed.get("inference")) or 0.0
    postprocess = finite_number(speed.get("postprocess")) or 0.0
    total = preprocess + inference + postprocess
    info = {
        "gflops": finite_number(metadata.get("gflops")),
        "parameters": metadata.get("parameters"),
        "preprocess_ms_per_image": preprocess,
        "inference_ms_per_image": inference,
        "postprocess_ms_per_image": postprocess,
        "fps_end_to_end": 1000.0 / total if total > 0 else None,
        "fps_inference": 1000.0 / inference if inference > 0 else None,
        "artifact_size_mb": model_path.stat().st_size / 1024 / 1024,
        "source_checkpoint_size_mb": finite_number(metadata.get("source_checkpoint_size_mb")),
    }

    table = PrettyTable()
    table.title = "Model Info"
    table.field_names = [
        "GFLOPs",
        "Parameters",
        "Preprocess/image",
        "Inference/image",
        "Postprocess/image",
        "FPS (end-to-end)",
        "FPS (inference)",
        "Artifact size",
    ]
    parameters = info["parameters"]
    table.add_row(
        [
            format_number(info["gflops"], 1),
            f"{int(parameters):,}" if parameters is not None else "-",
            f"{preprocess / 1000:.6f}s",
            f"{inference / 1000:.6f}s",
            f"{postprocess / 1000:.6f}s",
            format_number(info["fps_end_to_end"], 2),
            format_number(info["fps_inference"], 2),
            f"{info['artifact_size_mb']:.1f}MB",
        ]
    )
    return table, info


def make_metrics_table(rows: list[dict[str, Any]], overall: dict[str, Any]) -> PrettyTable:
    table = PrettyTable()
    table.title = "Detection Metrics"
    table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75", "mAP50-95"]
    for row in [*rows, overall]:
        table.add_row(
            [
                row["class_name"],
                format_number(row["precision"]),
                format_number(row["recall"]),
                format_number(row["f1"]),
                format_number(row["map50"]),
                format_number(row["map75"]),
                format_number(row["map50_95"]),
            ]
        )
    return table


def run(args: argparse.Namespace) -> Path:
    model_path = require_file(args.model, "model")
    data_path = require_file(args.data, "dataset YAML")
    metadata = load_release_metadata(model_path)
    warn_on_class_order_mismatch(data_path, metadata)

    # Delayed import keeps --help and release checks usable before dependencies are installed.
    import torch
    from ultralytics import RTDETR
    from ultralytics.models.rtdetr.val import RTDETRValidator

    class ReleaseRTDETRValidator(RTDETRValidator):
        """Normalize exported-backend output to the tuple used by PyTorch RT-DETR."""

        def postprocess(self, predictions):
            if torch.is_tensor(predictions):
                predictions = (predictions, )
            return super().postprocess(predictions)

    if model_path.suffix.lower() in {".pt", ".yaml", ".yml"}:
        model = RTDETR(str(model_path))
    else:
        # Ultralytics 8.0.201 can validate exported backends, but its RTDETR
        # convenience constructor unnecessarily rejects their file suffixes.
        # Calling the shared model initializer preserves RTDETR's validator map.
        from ultralytics.engine.model import Model

        model = RTDETR.__new__(RTDETR)
        Model.__init__(model, model=str(model_path), task="detect")
    apply_legacy_checkpoint_compatibility(model)
    val_args: dict[str, Any] = {
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project.expanduser().resolve()),
        "name": args.name,
        "exist_ok": True,
        "save_json": args.save_json,
        "plots": not args.no_plots,
    }
    if args.device is not None:
        val_args["device"] = args.device
    result = model.val(validator=ReleaseRTDETRValidator, **val_args)

    model_table, model_info = make_model_table(model_path, result, metadata)
    class_rows, overall = build_class_rows(result)
    metrics_table = make_metrics_table(class_rows, overall)

    print("\n" + str(model_table))
    print(metrics_table)

    save_dir = Path(result.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_dir / "paper_data.txt"
    report_path.write_text(f"{model_table}\n{metrics_table}\n", encoding="utf-8")

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "model": str(model_path),
            "data": str(data_path),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
        },
        "model": model_info,
        "classes": class_rows,
        "overall": overall,
        "release_metadata": metadata,
    }
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReports saved to: {report_path} and {metrics_path}")
    return save_dir


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
