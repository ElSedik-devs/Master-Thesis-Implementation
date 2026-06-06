# retinanet_bracelets.py
import os, json, math, random
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm


# ------------------------ Dataset ------------------------
class CocoDetectionSingleClass(Dataset):
    """
    COCO detection dataset, single foreground class (label=1).
    Images are referenced by file_name from the COCO JSON and loaded from img_dir.
    """
    def __init__(self, img_dir: str, ann_file: str):
        assert os.path.exists(img_dir), f"Missing dir: {img_dir}"
        assert os.path.exists(ann_file), f"Missing ann: {ann_file}"
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info   = self.coco.loadImgs(img_id)[0]
        path   = os.path.join(self.img_dir, info["file_name"])

        # Load (RGB) with PIL
        img = Image.open(path).convert("RGB")
        W, H = img.size

        # To tensor [0,1], CxHxW
        img_t = to_tensor(img)

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns    = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # single class
            areas.append(a.get("area", float((x2 - x1) * (y2 - y1))))
            iscrowd.append(int(a.get("iscrowd", 0)))

        # Empty-target safe tensors
        if len(boxes) == 0:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas_t  = torch.zeros((0,), dtype=torch.float32)
            crowd_t  = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t  = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas_t  = torch.tensor(areas, dtype=torch.float32)
            crowd_t  = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "image_id": torch.tensor([img_id]),
            "area":     areas_t,
            "iscrowd":  crowd_t,
        }
        return img_t, target


def detection_collate(batch):
    # Required by torchvision detection models
    return tuple(zip(*batch))


# ------------------------ Main Class ------------------------
class RetinaNetBracelets:
    """
    Reusable RetinaNet (ResNet50 FPN[v2 preferred]) trainer/inferencer for 'bracelet' detection.

    Key choices:
      - Uses ImageNet weights for the backbone (avoids COCO num_classes lock).
      - Works with both torchvision v2 model and legacy v1 fallback.
      - Single foreground class: set num_classes to 2 (background + bracelet) for compatibility
        with the original training script.
    """

    def __init__(
        self,
        train_img_dir: str,
        train_ann: str,
        val_img_dir: Optional[str] = None,
        val_ann: Optional[str] = None,
        out_dir: str = "checkpoints",
        model_variant: str = "v2",  # "v2" | "v1"
        anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
        anchor_aspect_ratios: Optional[Tuple[Tuple[float, ...], ...]] = None,
        device: Optional[str] = None,
    ):
        self.train_img_dir = train_img_dir
        self.val_img_dir   = val_img_dir
        self.train_ann     = train_ann
        self.val_ann       = val_ann
        self.out_dir       = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_variant = model_variant

        # Anchors (optional): tune if objects are small.
        self.anchor_generator = None
        if anchor_sizes is not None or anchor_aspect_ratios is not None:
            sizes = anchor_sizes if anchor_sizes is not None else ((16, 32, 64, 128, 256),)
            ratios = anchor_aspect_ratios if anchor_aspect_ratios is not None else ((0.5, 1.0, 2.0),)
            self.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

        self.model = None
        self.train_loader = None
        self.val_loader = None

    # --------------- Data ----------------
    def build_loaders(self, batch_size: int = 2, num_workers: int = 0):
        train_ds = CocoDetectionSingleClass(self.train_img_dir, self.train_ann)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=detection_collate, persistent_workers=(num_workers > 0)
        )

        self.val_loader = None
        if self.val_img_dir and self.val_ann and os.path.exists(self.val_img_dir) and os.path.exists(self.val_ann):
            val_ds = CocoDetectionSingleClass(self.val_img_dir, self.val_ann)
            self.val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=detection_collate, persistent_workers=(num_workers > 0)
            )

    # --------------- Model ----------------
    def build_model(self, num_classes: int = 2, freeze_backbone: bool = False):
        """
        num_classes default = 2 (background + 1 foreground class) to match the original training setup.
        """
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2

        model = None
        if self.model_variant == "v2":
            # Prefer v2; if not available in the env, fallback to v1.
            try:
                from torchvision.models.detection import retinanet_resnet50_fpn_v2
                model = retinanet_resnet50_fpn_v2(
                    weights=None,                         # IMPORTANT: no detection weights (to avoid COCO=91 lock)
                    weights_backbone=weights_backbone,    # ImageNet backbone
                    num_classes=num_classes,
                    anchor_generator=self.anchor_generator
                )
            except Exception:
                self.model_variant = "v1"  # fallback
        if model is None:
            # v1 fallback
            from torchvision.models.detection import retinanet_resnet50_fpn
            model = retinanet_resnet50_fpn(
                weights=None,
                weights_backbone=weights_backbone,
                num_classes=num_classes,
                anchor_generator=self.anchor_generator
            )

        if freeze_backbone:
            for p in model.backbone.parameters():
                p.requires_grad = False

        self.model = model.to(self.device)

    # --------------- Train ----------------
    def train(
        self,
        epochs: int = 12,
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        step_size: int = 5,
        gamma: float = 0.1,
        clip_grad: float = 5.0,
        print_each: int = 1,
        save_each_epoch: bool = True,
    ) -> str:
        assert self.model is not None, "Call build_model() first."
        assert self.train_loader is not None, "Call build_loaders() first."

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_val = float("inf")
        best_path = os.path.join(self.out_dir, "retinanet_best.pth")

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.model.train()
            running = 0.0
            for images, targets in tqdm(self.train_loader, desc=f"[Epoch {epoch}/{epochs}] Train"):
                images  = [im.to(self.device) for im in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)  # dict of losses
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                if clip_grad and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(params, clip_grad)
                optimizer.step()

                running += float(loss.item())

            scheduler.step()
            train_loss = running / max(1, len(self.train_loader))

            # ---- "Validation loss" (sanity check) ----
            val_loss = None
            if self.val_loader is not None:
                self.model.train()  # detection losses require train mode
                v_running = 0.0
                with torch.no_grad():
                    for images, targets in tqdm(self.val_loader, desc=f"[Epoch {epoch}/{epochs}] Val"):
                        images  = [im.to(self.device) for im in images]
                        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                        v_loss_dict = self.model(images, targets)
                        v_loss = sum(v_loss_dict.values())
                        v_running += float(v_loss.item())
                val_loss = v_running / max(1, len(self.val_loader))
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            # ---- Save ----
            if save_each_epoch:
                torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"retinanet_epoch{epoch}.pth"))
            if (val_loss is not None) and (val_loss < best_val):
                best_val = val_loss
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✅ Saved new best (val_loss={best_val:.4f})")

        # final save
        final_path = os.path.join(self.out_dir, "retinanet_final.pth")
        torch.save(self.model.state_dict(), final_path)
        print("✅ Training finished.")
        print("Saved:", final_path)
        return best_path if os.path.exists(best_path) else final_path

    # --------------- I/O ----------------
    def save(self, path: str):
        assert self.model is not None
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights to {path}")

    def load(self, path: str):
        assert self.model is not None, "Call build_model() first to create the module structure."
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        print(f"Loaded weights from {path}")

    # --------------- Inference ----------------
    @torch.inference_mode()
    def predict(self, image_paths: List[str], score_thresh: float = 0.4, max_detections: int = 100):
        assert self.model is not None
        self.model.eval()
        outputs: List[Dict[str, Any]] = []
    
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            tensor = to_tensor(img).to(self.device)
            out = self.model([tensor])[0]
    
            keep = (out["scores"] >= score_thresh).nonzero(as_tuple=False).squeeze(1).tolist()
            boxes  = out["boxes"][keep][:max_detections].detach().cpu().numpy().tolist()
            scores = out["scores"][keep][:max_detections].detach().cpu().numpy().tolist()
            labels = out["labels"][keep][:max_detections].detach().cpu().numpy().astype(int).tolist()
    
            outputs.append({"path": p, "boxes": boxes, "scores": scores, "labels": labels})
        return outputs

    # --------------- Visualization ----------------
    def draw(self, image_path: str, det: Dict[str, Any], label_name: str = "bracelet", score_fmt: str = "{:.2f}"):
        from PIL import ImageDraw
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b, s in zip(det["boxes"], det["scores"]):
            x1, y1, x2, y2 = map(float, b)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
            draw.text((x1 + 2, y1 + 2), f"{label_name} {score_fmt.format(float(s))}", fill=(0, 255, 0))
        return img

    # --------------- Folder prediction helper ----------------
    @torch.inference_mode()
    def predict_folder(
        self,
        images_dir: str,
        out_json: str,
        out_vis_dir: Optional[str] = None,
        score_thresh: float = 0.4,
        batch_size: int = 8,
    ):
        import glob, json as _json
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        if out_vis_dir:
            os.makedirs(out_vis_dir, exist_ok=True)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp", "*.webp")
        all_imgs: List[str] = []
        for e in exts:
            all_imgs.extend(glob.glob(os.path.join(images_dir, e)))
        all_imgs.sort()

        all_results = []
        for i in range(0, len(all_imgs), batch_size):
            chunk = all_imgs[i:i + batch_size]
            preds = self.predict(chunk, score_thresh=score_thresh)
            all_results.extend(preds)

            if out_vis_dir:
                for p, det in zip(chunk, preds):
                    vis = self.draw(p, det)
                    base = os.path.splitext(os.path.basename(p))[0]
                    vis.save(os.path.join(out_vis_dir, f"{base}_det.jpg"))

        with open(out_json, "w") as f:
            _json.dump(all_results, f)
        print(f"Done. Saved detections for {len(all_imgs)} images to {out_json}"
              + (f" and overlays to {out_vis_dir}" if out_vis_dir else ""))

