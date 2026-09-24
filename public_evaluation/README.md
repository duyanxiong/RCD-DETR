# RCD-DETR    dataset: RDD2022 (chinese subset) 

它可以复现 `examples/best.pt` 对应模型在 RDD2022 上的 Precision、Recall、F1、mAP50、mAP75、mAP50-95 和推理速度，并生成推理表格结果。



> English summary:  Install the dependencies, configure an RDD2022 dataset YAML, and run `python val.py` to reproduce the reported detection metrics.

## 目录内容

```text
public_evaluation/
├── configs/rdd2022.yaml       # 数据集配置模板
├── models/
│   ├── best.torchscript       # 发布模型（）
│   └── model_metadata.json    # 参数量、GFLOPs、哈希等
├── tools/
│   ├── export_for_release.py  # 维护者在私有完整工程中执行一次
│   └── check_release.py       # 发布前泄露与完整性检查
├── val.py                     # 公开评测入口
└── requirements.txt
```

## 使用方法

建议使用 Python 3.9–3.11。先按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与本机 CPU/CUDA 匹配的 PyTorch，再安装其余依赖：

```bash
pip install -r requirements.txt
```

准备 YOLO 检测格式的数据集(本项目中按照实验实际使用数据集划分方式划分）：

```text
RDD2022/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

将 `configs/rdd2022.yaml` 中的 `path` 改为数据集路径，然后运行：

```bash
python val.py --data configs/rdd2022.yaml --split test --imgsz 640 --batch 8 --device 0
```

没有 CUDA 时使用 `--device cpu`。结果保存在 `results/val/`：

- `paper_data.txt`：便于直接查看或粘贴到实验记录的表格；
- `metrics.json`：结构化的逐类别与总体指标；
- Ultralytics 生成的 PR 曲线、混淆矩阵等图片。

可通过 `python val.py --help` 查看全部参数。测速结果会受 GPU、CUDA、批量大小、预热状态和后台负载影响；精度复现时请保持 `imgsz=640`、`split=test` 及相同测试集版本。

