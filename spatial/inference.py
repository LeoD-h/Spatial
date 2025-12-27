import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import requests

from .data import CLASS_NAMES


def load_model(model_path: Path):
    """
    Delayed import to avoid pulling torch when unused.
    """
    from ultralytics import YOLO

    return YOLO(str(model_path))


def predict_image(
    model,
    source: Path,
    save_dir: Path,
    conf: float = 0.10,
    iou: float = 0.45,
) -> Tuple[Path, List[Dict]]:
    """
    Runs inference on a single image and writes an annotated copy.
    Returns the output path and a list of detections with confidences.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(source=str(source), conf=conf, iou=iou, verbose=False)
    res = results[0]

    annotated = res.plot()  # BGR numpy array
    out_path = save_dir / f"{Path(source).stem}_pred.jpg"
    cv2.imwrite(str(out_path), annotated)

    detections: List[Dict] = []
    if res.boxes is not None and len(res.boxes) > 0:
        for cls_id, score in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            cid = int(cls_id)
            detections.append(
                {
                    "class_id": cid,
                    "class_name": CLASS_NAMES.get(cid, str(cid)),
                    "confidence": float(score),
                }
            )

    return out_path, detections


def download_image(url: str) -> Path:
    """
    Downloads an image to a temporary file and returns its path.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    suffix = Path(url).suffix or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.flush()
    return Path(tmp.name)


def evaluate_batch(
    model,
    images: Iterable[Path],
    save_dir: Path,
    label_dir: Optional[Path] = None,
    conf: float = 0.25,
    iou: float = 0.45,
) -> Dict:
    """
    Runs predictions on a set of images and aggregates simple statistics.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    stats = {cid: 0 for cid in CLASS_NAMES.keys()}
    total = 0
    detected = 0
    correct = 0
    verifiable = 0
    per_image: List[Dict] = []

    for img_path in images:
        total += 1
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            verbose=False,
        )
        res = results[0]
        annotated = res.plot()
        out_path = save_dir / f"{img_path.stem}_pred.jpg"
        cv2.imwrite(str(out_path), annotated)

        top_class = None
        top_conf = None
        if res.boxes is not None and len(res.boxes) > 0:
            detected += 1
            top_idx = int(res.boxes.cls[0])
            top_class = top_idx
            top_conf = float(res.boxes.conf[0])
            stats[top_idx] = stats.get(top_idx, 0) + 1

        true_class = None
        if label_dir:
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, "r") as f:
                    line = f.readline().strip()
                    if line:
                        true_class = int(line.split()[0])
                        verifiable += 1
                        if top_class is not None and top_class == true_class:
                            correct += 1

        per_image.append(
            {
                "image": str(img_path),
                "prediction": top_class,
                "prediction_name": CLASS_NAMES.get(top_class, None)
                if top_class is not None
                else None,
                "confidence": top_conf,
                "ground_truth": true_class,
                "annotated_path": str(out_path),
            }
        )

    detection_rate = (detected / total) * 100 if total else 0.0
    accuracy = (correct / verifiable) * 100 if verifiable else 0.0

    return {
        "total_images": total,
        "detected_images": detected,
        "detection_rate": detection_rate,
        "accuracy": accuracy,
        "counts": stats,
        "verifiable": verifiable,
        "details": per_image,
        "output_dir": str(save_dir),
    }
