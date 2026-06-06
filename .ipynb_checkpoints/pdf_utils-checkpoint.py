import fitz  # pip install pymupdf
import os
import json
import random
from typing import List, Tuple

class PDFToImages:
    def __init__(self, pdf_path: str, out_dir: str):
        self.pdf_path = pdf_path
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def convert(self) -> List[str]:
        """
        Extract the original embedded image from each page (if present)
        and save it with no recompression. Returns list of saved paths.
        """
        saved: List[str] = []
        doc = fitz.open(self.pdf_path)
        for i, page in enumerate(doc):
            images = page.get_images(full=True)
            if not images:
                print(f"Page {i+1}: no image found, skipping.")
                continue
            xref = images[0][0]  # usually 1 image per scanned page
            img = doc.extract_image(xref)
            ext = img["ext"]
            img_bytes = img["image"]

            out_path = os.path.join(self.out_dir, f"page_{i+1:04d}.{ext}")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            saved.append(out_path)
            print(f"Saved page {i+1} → {out_path}")
        print("All done!")
        return saved

    def convert_sampled(self, n: int, out_dir: str) -> List[str]:
        """
        Extract and save every n-th page (1-indexed: pages n, 2n, 3n, ...).
        Uses the same lossless extraction logic. Returns list of saved paths.

        Args:
            n: sample step, must be >= 1
            out_dir: directory to save the sampled images
        """
        if n < 1:
            raise ValueError("n must be >= 1")

        os.makedirs(out_dir, exist_ok=True)

        saved: List[str] = []
        doc = fitz.open(self.pdf_path)
        for i, page in enumerate(doc, start=1):  # i is 1-based page number
            if i % n != 0:
                continue
            images = page.get_images(full=True)
            if not images:
                print(f"Page {i}: no image found, skipping.")
                continue
            xref = images[0][0]
            img = doc.extract_image(xref)
            ext = img["ext"]
            img_bytes = img["image"]

            out_path = os.path.join(out_dir, f"page_{i:04d}.{ext}")
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            saved.append(out_path)
            print(f"Saved sampled page {i} → {out_path}")
        print("Sampling done!")
        return saved
        
def split_coco_annotations(
    json_path: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Split a single COCO detection JSON into train/val JSON files by image id.
    Only the JSON is split; images stay in the same folder.

    Args:
        json_path: Path to the original COCO JSON (from CVAT export).
        train_ratio: Fraction of images to assign to the training split (0 < r < 1).
        seed: RNG seed for reproducibility.

    Returns:
        (train_json_path, val_json_path)
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"COCO file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if "images" not in data or "annotations" not in data:
        raise ValueError("Invalid COCO JSON: missing 'images' or 'annotations'.")

    random.seed(seed)
    img_ids = [img["id"] for img in data["images"]]
    random.shuffle(img_ids)

    split_idx = int(len(img_ids) * train_ratio)
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])

    def subset(ids):
        imgs = [img for img in data["images"] if img["id"] in ids]
        anns = [ann for ann in data["annotations"] if ann["image_id"] in ids]
        # Keep categories untouched
        return {"images": imgs, "annotations": anns, "categories": data.get("categories", [])}

    train_json = subset(train_ids)
    val_json = subset(val_ids)

    base = os.path.dirname(json_path)
    train_path = os.path.join(base, "annotations_train.json")
    val_path = os.path.join(base, "annotations_val.json")

    with open(train_path, "w") as f:
        json.dump(train_json, f)
    with open(val_path, "w") as f:
        json.dump(val_json, f)

    print(f"Split done → Train images: {len(train_json['images'])} | Val images: {len(val_json['images'])}")
    return train_path, val_path


