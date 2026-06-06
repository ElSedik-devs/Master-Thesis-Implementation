# evaluate_retinanet.py
"""
Evaluation and error analysis for the RetinaNet bracelet detector.

Key entry point: evaluate_model(...)
"""

import os
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

import pandas as pd
import numpy as np
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from retinanet_model import RetinaNetBracelets, CocoDetectionSingleClass, detection_collate


# ---------- helpers for attributes ----------

def _size_category(area_ratio: float) -> str:
    """
    Buckets based on box_area / image_area.
    - small  : < 1% of page
    - medium : 1–5%
    - large  : >= 5%
    Thresholds can be adjusted for different object-scale definitions.
    """
    if area_ratio < 0.01:
        return "small"
    elif area_ratio < 0.05:
        return "medium"
    else:
        return "large"


def _near_border(cx: float, cy: float, W: int, H: int, frac: float = 0.10) -> bool:
    """
    True if the box center is within `frac` of the nearest image border.
    Example: frac=0.10 -> within 10% of width/height.
    """
    dist_left = cx
    dist_right = W - cx
    dist_top = cy
    dist_bottom = H - cy
    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
    threshold = frac * min(W, H)
    return bool(min_dist < threshold)


def _elongated(w: float, h: float, ratio_thresh: float = 4.0) -> bool:
    """
    Very elongated / thin shapes (e.g. side-view bracelet edges).
    """
    if w <= 0 or h <= 0:
        return False
    r = max(w / h, h / w)
    return bool(r >= ratio_thresh)


# ---------- matching & per-image analysis ----------

def _analyse_single_image(
    image_id: int,
    image_path: str,
    image_size: Tuple[int, int],
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_thresh: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Greedy matching per image.

    Returns:
      rows   : list of dicts (one row per TP / FP / FN instance)
      summary: dict with per-image counts
    """
    W, H = image_size
    rows: List[Dict[str, Any]] = []

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]

    # No GTs -> everything is FP
    if num_gt == 0:
        for j in range(num_pred):
            x1, y1, x2, y2 = pred_boxes[j].tolist()
            area = (x2 - x1) * (y2 - y1)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            area_ratio = area / (W * H) if W * H > 0 else 0.0
            rows.append(
                dict(
                    image_id=image_id,
                    image_path=image_path,
                    instance_type="FP",
                    score=float(pred_scores[j]),
                    iou=None,
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    area=float(area),
                    area_ratio=float(area_ratio),
                    size_cat=_size_category(area_ratio),
                    near_border=_near_border(cx, cy, W, H),
                    elongated=_elongated(x2 - x1, y2 - y1),
                )
            )

        summary = dict(
            image_id=image_id,
            image_path=image_path,
            num_gt=0,
            num_pred=num_pred,
            num_tp=0,
            num_fp=num_pred,
            num_fn=0,
        )
        return rows, summary

    # At least one GT
    ious = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]

    gt_matched = torch.zeros(num_gt, dtype=torch.bool)
    order = torch.argsort(pred_scores, descending=True)

    tp_flags = torch.zeros(num_pred, dtype=torch.bool)
    matched_iou = torch.zeros(num_pred, dtype=torch.float32)

    for j in order:
        iou_row = ious[j]
        best_iou, best_idx = iou_row.max(dim=0)
        if best_iou >= iou_thresh and not gt_matched[best_idx]:
            tp_flags[j] = True
            gt_matched[best_idx] = True
            matched_iou[j] = best_iou

    # Predictions (TP/FP)
    for j in range(num_pred):
        x1, y1, x2, y2 = pred_boxes[j].tolist()
        area = (x2 - x1) * (y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area_ratio = area / (W * H) if W * H > 0 else 0.0
        instance_type = "TP" if tp_flags[j] else "FP"
        rows.append(
            dict(
                image_id=image_id,
                image_path=image_path,
                instance_type=instance_type,
                score=float(pred_scores[j]),
                iou=float(matched_iou[j]) if instance_type == "TP" else None,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                area=float(area),
                area_ratio=float(area_ratio),
                size_cat=_size_category(area_ratio),
                near_border=_near_border(cx, cy, W, H),
                elongated=_elongated(x2 - x1, y2 - y1),
            )
        )

    # Missed GTs (FN)
    fn_indices = (~gt_matched).nonzero(as_tuple=False).flatten().tolist()
    for k in fn_indices:
        x1, y1, x2, y2 = gt_boxes[k].tolist()
        area = (x2 - x1) * (y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        area_ratio = area / (W * H) if W * H > 0 else 0.0
        rows.append(
            dict(
                image_id=image_id,
                image_path=image_path,
                instance_type="FN",
                score=None,
                iou=None,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                area=float(area),
                area_ratio=float(area_ratio),
                size_cat=_size_category(area_ratio),
                near_border=_near_border(cx, cy, W, H),
                elongated=_elongated(x2 - x1, y2 - y1),
            )
        )

    summary = dict(
        image_id=image_id,
        image_path=image_path,
        num_gt=int(num_gt),
        num_pred=int(num_pred),
        num_tp=int(tp_flags.sum().item()),
        num_fp=int((~tp_flags).sum().item()),
        num_fn=int(len(fn_indices)),
    )
    return rows, summary


# ---------- global metrics (mAP, recall, mean IoU) ----------

def _compute_pr_metrics(
    df_instances: pd.DataFrame,
    total_gt: int,
) -> Dict[str, float]:
    """
    Uses all predictions (TP + FP) sorted by score to build a
    precision–recall curve at IoU=0.5, then integrates AP.
    """
    pred_df = df_instances[df_instances["instance_type"].isin(["TP", "FP"])].copy()
    if pred_df.empty or total_gt == 0:
        return {"mAP@0.5": 0.0, "Recall@0.5": 0.0, "mean_iou@0.5": 0.0}

    pred_df = pred_df.sort_values("score", ascending=False)

    tp_cumsum = (pred_df["instance_type"] == "TP").astype(int).cumsum()
    fp_cumsum = (pred_df["instance_type"] == "FP").astype(int).cumsum()

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / max(1, total_gt)

    # AP as area under the precision–recall curve
    mrec = np.concatenate(([0.0], recall.to_numpy(), [1.0]))
    mpre = np.concatenate(([0.0], precision.to_numpy(), [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    ap = 0.0
    for i in range(1, len(mrec)):
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]

    recall_at_05 = float(recall.iloc[-1])

    # Mean IoU of true positives
    tp_iou = pred_df.loc[pred_df["instance_type"] == "TP", "iou"]
    mean_iou = float(tp_iou.mean()) if len(tp_iou) > 0 else 0.0

    return {
        "mAP@0.5": float(ap),
        "Recall@0.5": float(recall_at_05),
        "mean_iou@0.5": mean_iou,
    }


# ---------- optional: COCOeval sanity check ----------

def _run_coco_eval(test_ann_path: str, results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Builds a COCO-style detections list and runs COCOeval.
    Uses all retained predictions, so score_thresh_for_metrics should be set appropriately.
    """
    coco_gt = COCO(test_ann_path)

    # Some CVAT-exported COCO files may not contain optional top-level fields
    # like "info" or "licenses", but pycocotools.loadRes() may expect them.
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    coco_dets = []
    for r in results:
        image_id = int(r["image_id"])
        boxes = r["pred_boxes"].cpu().numpy()
        scores = r["scores"].cpu().numpy()
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = b
            coco_dets.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s),
                }
            )

    if len(coco_dets) == 0:
        return {}

    coco_dt = coco_gt.loadRes(coco_dets)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "COCO_AP": float(stats[0]),
        "COCO_AP50": float(stats[1]),
        "COCO_AP75": float(stats[2]),
        "COCO_AR100": float(stats[8]),
    }


# ---------- main public function ----------

def evaluate_model(
    checkpoint_path: str,
    test_img_dir: str,
    test_ann_path: str,
    score_thresh_for_metrics: float = 0.05,
    iou_thresh: float = 0.5,
    device: str = None,
    run_coco_eval: bool = True,
):
    """
    Full evaluation + error analysis.

    Returns:
      metrics      : dict with mAP@0.5, Recall@0.5, mean_iou@0.5 (+ optional COCO stats)
      df_instances : per-instance DataFrame (TP/FP/FN + attributes)
      df_per_image : per-image summary DataFrame
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    assert os.path.isfile(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    assert os.path.isdir(test_img_dir), f"Image dir not found: {test_img_dir}"
    assert os.path.isfile(test_ann_path), f"Annotation JSON not found: {test_ann_path}"

    # 1) Build model and load weights
    runner = RetinaNetBracelets(
        train_img_dir=test_img_dir,
        train_ann=test_ann_path,
        val_img_dir=None,
        val_ann=None,
        out_dir=os.path.dirname(checkpoint_path) or ".",
        model_variant="v2",
        device=device,
    )
    runner.build_model(num_classes=2)
    runner.load(checkpoint_path)
    model = runner.model
    model.eval()

    # 2) Test dataset / loader using the deterministic COCO dataset
    test_ds = CocoDetectionSingleClass(test_img_dir, test_ann_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=detection_collate,
    )

    id_to_filename = {img_id: info["file_name"] for img_id, info in test_ds.coco.imgs.items()}

    all_results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    per_image_summaries: List[Dict[str, Any]] = []

    total_gt = 0

    with torch.inference_mode():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            img = images[0].to(device_t)
            target = {k: v.to(device_t) for k, v in targets[0].items()}
            image_id = int(target["image_id"].item())
            H, W = img.shape[1:]  # C x H x W

            out = model([img])[0]
            scores = out["scores"]
            keep = scores >= score_thresh_for_metrics
            scores = scores[keep]
            pred_boxes = out["boxes"][keep]

            gt_boxes = target["boxes"].cpu()
            total_gt += gt_boxes.shape[0]

            image_path = os.path.join(test_img_dir, id_to_filename[image_id])

            all_results.append(
                dict(
                    image_id=image_id,
                    pred_boxes=pred_boxes.cpu(),
                    scores=scores.cpu(),
                )
            )

            rows, summary = _analyse_single_image(
                image_id=image_id,
                image_path=image_path,
                image_size=(W, H),
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes.cpu(),
                pred_scores=scores.cpu(),
                iou_thresh=iou_thresh,
            )
            all_rows.extend(rows)
            per_image_summaries.append(summary)

    df_instances = pd.DataFrame(all_rows)
    df_per_image = pd.DataFrame(per_image_summaries)

    metrics = _compute_pr_metrics(df_instances, total_gt=total_gt)

    if run_coco_eval:
        coco_metrics = _run_coco_eval(test_ann_path, all_results)
        metrics.update(coco_metrics)

    # Nice compact summary
    print("=== Bracelet detection on TEST set ===")
    print(f"  #images:   {len(test_ds)}")
    print(f"  #GT boxes: {total_gt}")
    print(f"  mAP@0.5:   {metrics['mAP@0.5']:.3f}")
    print(f"  Recall@0.5:{metrics['Recall@0.5']:.3f}")
    print(f"  mean IoU:  {metrics['mean_iou@0.5']:.3f}")
    if run_coco_eval and "COCO_AP50" in metrics:
        print(f"  COCO AP50:{metrics['COCO_AP50']:.3f}")

    print("\nFN counts by size bucket:")
    if not df_instances.empty:
        fn_by_size = df_instances[df_instances["instance_type"] == "FN"].groupby("size_cat").size()
        print(fn_by_size.to_string() if not fn_by_size.empty else "  (no FN)")
        fn_by_border = df_instances[df_instances["instance_type"] == "FN"].groupby("near_border").size()
        print("\nFN counts by near_border:")
        print(fn_by_border.to_string() if not fn_by_border.empty else "  (no FN)")

    return metrics, df_instances, df_per_image


# ---------- visualization helpers ----------

def plot_image_with_detections(
    image_path: str,
    df_instances: pd.DataFrame,
    score_min: float = 0.0,
    figsize: Tuple[int, int] = (10, 14),
):
    """
    Draw:
      - FN (missed GTs) in dashed lime.
      - TP predictions in blue.
      - FP predictions in red.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image

    img = Image.open(image_path).convert("RGB")

    df_img = df_instances[df_instances["image_path"] == image_path]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img, cmap="gray")
    ax.axis("off")

    # FNs
    for _, row in df_img[df_img["instance_type"] == "FN"].iterrows():
        rect = patches.Rectangle(
            (row["x1"], row["y1"]),
            row["x2"] - row["x1"],
            row["y2"] - row["y1"],
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)

    # TP / FP preds
    for _, row in df_img[df_img["instance_type"].isin(["TP", "FP"])].iterrows():
        if row["score"] is not None and row["score"] < score_min:
            continue
        color = "blue" if row["instance_type"] == "TP" else "red"
        rect = patches.Rectangle(
            (row["x1"], row["y1"]),
            row["x2"] - row["x1"],
            row["y2"] - row["y1"],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        if row["score"] is not None:
            ax.text(
                row["x1"],
                row["y1"] - 2,
                f"{row['instance_type']} {row['score']:.2f}",
                color=color,
                fontsize=8,
                backgroundcolor="black",
            )

    plt.tight_layout()
    return fig, ax


def show_worst_images(df_per_image: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Quick listing of pages with most FNs / highest miss-rate.
    The returned rows can be passed to plot_image_with_detections.
    """
    if df_per_image.empty:
        return df_per_image

    df = df_per_image.copy()
    df["fn_rate"] = df["num_fn"] / df["num_gt"].replace(0, 1)
    df["tp_rate"] = df["num_tp"] / df["num_gt"].replace(0, 1)

    worst = df.sort_values(["num_fn", "fn_rate"], ascending=[False, False]).head(top_k)
    return worst[["image_id", "image_path", "num_gt", "num_tp", "num_fp", "num_fn", "fn_rate", "tp_rate"]]
