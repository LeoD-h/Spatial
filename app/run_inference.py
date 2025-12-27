import argparse
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed from app/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial.data import CLASS_NAMES
from spatial.inference import download_image, evaluate_batch, load_model, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the Spatial model.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/galaxy_model_v2_expert.pt"),
        help="Path to the trained YOLO weights.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Local image to analyse.",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Remote image URL to download and analyse.",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("data/processed/galaxy_expert/val/images"),
        help="Folder used when running batch predictions.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/processed/galaxy_expert/val/labels"),
        help="Ground truth labels folder for stats (optional).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of images to sample from folder for batch stats (0 to skip).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help="Where to save annotated images.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)

    if args.image or args.url:
        if args.url:
            local_path = download_image(args.url)
        else:
            local_path = args.image

        out_path, detections = predict_image(
            model,
            local_path,
            args.output_dir,
            conf=args.conf,
            iou=args.iou,
        )

        print(f"Annotated image saved to: {out_path}")
        if not detections:
            print("No detections.")
        else:
            print("Detections:")
            for det in detections:
                print(
                    f"- {det['class_name']} (id={det['class_id']}) "
                    f"conf={det['confidence']:.2f}"
                )
        return

    if args.count > 0:
        images = list(Path(args.folder).glob("*.jpg"))
        if not images:
            raise SystemExit(f"No images found in {args.folder}")

        sampled = random.sample(images, k=min(len(images), args.count))
        label_dir = args.labels if args.labels.exists() else None
        summary = evaluate_batch(
            model,
            sampled,
            save_dir=args.output_dir,
            label_dir=label_dir,
            conf=args.conf,
            iou=args.iou,
        )

        print(
            f"Processed {summary['total_images']} images "
            f"-> detection rate {summary['detection_rate']:.1f}%"
        )
        if label_dir:
            print(f"Accuracy (when GT available): {summary['accuracy']:.2f}%")
        print("Counts per class:")
        for cid, count in summary["counts"].items():
            print(f"- {CLASS_NAMES.get(cid, cid)}: {count}")
        print(f"Annotated outputs in: {summary['output_dir']}")
        return

    raise SystemExit(
        "Provide an --image, --url or set --count with a folder to run batch stats."
    )


if __name__ == "__main__":
    main()
