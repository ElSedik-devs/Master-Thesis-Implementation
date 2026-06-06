# Thesis RetinaNet Bracelet Detection Implementation

This repository contains the implementation code for a thesis project on bracelet detection using RetinaNet with a ResNet-50 FPN backbone.

## Contents

- `retinanet_model.py`: reusable dataset, model, training, and inference utilities.
- `evaluate_retinanet.py`: evaluation and error-analysis utilities for RetinaNet predictions.
- `pdf_utils.py`: utilities for extracting scanned PDF pages and splitting COCO annotations.
- `annotations/`: COCO-style train, validation, and combined annotations.
- `eval_outputs/`: CSV evaluation summaries and per-image analysis outputs.
- `*.ipynb`: experiment and workflow notebooks.

Large generated assets, extracted page images, model checkpoints, local caches, and archives are intentionally excluded from Git. Store those separately or use Git LFS / a release artifact if they need to be shared.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

The code expects image folders and COCO annotation paths to be provided locally. Checkpoints are not committed because individual `.pth` files exceed GitHub's normal file-size limit.

