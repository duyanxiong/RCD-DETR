"""Maintainer-only exporter from the private PyTorch checkpoint to TorchScript."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


PUBLIC_DIR = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = PUBLIC_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the private checkpoint into the public release directory.")
    parser.add_argument("--weights", type=Path, default=PRIVATE_ROOT / "examples" / "best.pt")
    parser.add_argument("--output", type=Path, default=PUBLIC_DIR / "models" / "best.torchscript")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA index or cpu")
    parser.add_argument("--force", action="store_true", help="replace an existing release artifact")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_legacy_checkpoint_compatibility(model: object) -> None:
    """Fill inference-only attributes added after this checkpoint was trained."""
    pytorch_model = getattr(model, "model", model)
    modules = getattr(pytorch_model, "modules", None)
    if modules is None:
        return
    for module in modules():
        if module.__class__.__name__ == "RTDETRDecoder":
            # Older non-DFL RT-DETR checkpoints predate these runtime flags. The
            # current forward method needs them, but no new tensors are required.
            if not hasattr(module, "use_dfl"):
                module.use_dfl = False
            if not hasattr(module, "reg_max"):
                module.reg_max = 16
            if not hasattr(module, "dfl_scale"):
                module.dfl_scale = 1.0


def torch_device(value: str):
    import torch

    return torch.device(f"cuda:{value}" if value.isdigit() else value)


def prepare_portable_fixed_shape_export(model: object, imgsz: int, device_value: str) -> None:
    """Replace traced device creation with movable fixed-shape buffers.

    The legacy exporter traces ``x.device`` as a fixed device. Precomputing the
    fixed-size positional embeddings and anchors as registered buffers keeps
    them colocated with the TorchScript module when the archive is loaded.
    """
    import torch
    from ultralytics.nn.modules.head import RTDETRDecoder
    from ultralytics.nn.modules.transformer import AIFI, TransformerEncoderLayer

    device = torch_device(device_value)
    # The caller passes RTDETRDetectionModel. Do not unwrap its internal
    # ``.model`` Sequential because the task-level forward resolves skip links.
    pytorch_model = model.to(device).eval()
    aifi_modules = [module for module in pytorch_model.modules() if isinstance(module, AIFI)]
    decoder_modules = [module for module in pytorch_model.modules() if isinstance(module, RTDETRDecoder)]

    input_shapes: dict[object, tuple[int, int, int]] = {}
    decoder_shapes: dict[object, list[tuple[int, int]]] = {}
    handles = []
    for module in aifi_modules:
        handles.append(
            module.register_forward_pre_hook(
                lambda current, inputs: input_shapes.__setitem__(current, tuple(int(x) for x in inputs[0].shape[1:]))
            )
        )
    for module in decoder_modules:
        handles.append(
            module.register_forward_pre_hook(
                lambda current, inputs: decoder_shapes.__setitem__(
                    current,
                    [(int(feature.shape[2]), int(feature.shape[3])) for feature in inputs[0]],
                )
            )
        )
    try:
        with torch.inference_mode():
            pytorch_model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    finally:
        for handle in handles:
            handle.remove()

    for module in aifi_modules:
        channels, height, width = input_shapes[module]
        position = module.build_2d_sincos_position_embedding(width, height, channels).to(device)
        module.register_buffer("_release_pos_embed", position, persistent=True)

    def portable_forward(self, x):
        channels, height, width = x.shape[1:]
        position = self._release_pos_embed.to(dtype=x.dtype)
        encoded = TransformerEncoderLayer.forward(
            self,
            x.flatten(2).permute(0, 2, 1),
            pos=position,
        )
        return encoded.permute(0, 2, 1).view([-1, channels, height, width]).contiguous()

    AIFI.forward = portable_forward

    for module in decoder_modules:
        anchors, valid_mask = module._generate_anchors(
            decoder_shapes[module],
            dtype=torch.float32,
            device=device,
        )
        module.register_buffer("_release_anchors", anchors, persistent=True)
        module.register_buffer("_release_valid_mask", valid_mask, persistent=True)

    def portable_generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        del shapes, grid_size, device, eps
        return self._release_anchors.to(dtype=dtype), self._release_valid_mask

    RTDETRDecoder._generate_anchors = portable_generate_anchors


def main() -> int:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists (use --force to replace it): {output}")

    # Import the private implementation only for this one-time maintainer export.
    sys.path.insert(0, str(PRIVATE_ROOT))
    from ultralytics import RTDETR
    from ultralytics.utils.torch_utils import model_info

    model = RTDETR(str(weights))
    apply_legacy_checkpoint_compatibility(model.model)
    _, parameters, _, gflops = model_info(model.model, imgsz=args.imgsz)
    prepare_portable_fixed_shape_export(model.model, args.imgsz, args.device)
    exported = Path(
        model.export(format="torchscript", imgsz=args.imgsz, batch=1, device=args.device, optimize=False)
    ).resolve()
    if not exported.is_file():
        raise RuntimeError(f"export did not create the expected artifact: {exported}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    if exported != output:
        shutil.move(str(exported), str(output))

    names = getattr(model.model, "names", {})
    if isinstance(names, dict):
        names = {str(key): value for key, value in names.items()}
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": "torchscript",
        "derived_from": weights.name,
        "imgsz": args.imgsz,
        "parameters": int(parameters),
        "gflops": float(gflops),
        "names": names,
        "source_checkpoint_size_mb": weights.stat().st_size / 1024 / 1024,
        "artifact_size_mb": output.stat().st_size / 1024 / 1024,
        "runtime_test_device": args.device,
        "sha256": sha256(output),
    }
    metadata_path = output.parent / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # A real forward pass verifies both source independence and executable output.
    import torch

    device = torch_device(args.device)
    released_model = torch.jit.load(str(output), map_location=device).eval()
    with torch.inference_mode():
        prediction = released_model(torch.zeros(1, 3, args.imgsz, args.imgsz, device=device))
    expected_shape = (1, 300, 4 + len(names))
    if tuple(prediction.shape) != expected_shape:
        raise RuntimeError(f"unexpected release output shape: {tuple(prediction.shape)} != {expected_shape}")
    print(f"release model: {output}")
    print(f"metadata:      {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
