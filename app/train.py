import argparse
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed from app/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spatial.data import prepare_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Spatial YOLO model.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path("data/raw/images_training_rev1.zip"),
        help="Path to images_training_rev1.zip",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=Path("data/raw/training_solutions_rev1.csv"),
        help="Path to training_solutions_rev1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/galaxy_expert"),
        help="Where to write the YOLO-ready dataset.",
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=12000,
        help="Number of samples to keep (<=0 for full dataset).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--base-weights",
        type=str,
        default="yolov8n.pt",
        help="Starting checkpoint for training.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--run-name",
        type=str,
        default="galaxy_fast_expert",
        help="Name used by Ultralytics to create the run folder.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("models"),
        help="Where to copy the final best.pt file.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build the dataset without running training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_size = None if args.dataset_size is None or args.dataset_size <= 0 else args.dataset_size

    dataset_yaml, dataset_root = prepare_dataset(
        zip_path=args.zip_path,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        dataset_size=dataset_size,
        val_split=args.val_split,
    )

    if args.prepare_only:
        print("Dataset prepared. Skipping training as requested.")
        return

    from ultralytics import YOLO

    model = YOLO(args.base_weights)
    results = model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        name=args.run_name,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    dest = args.artifacts_dir / f"{args.run_name}_best.pt"
    if best_path.exists():
        shutil.copy2(best_path, dest)
        print(f"Training complete. Best model copied to {dest}")
    else:
        print("Training finished but no best.pt was found to copy.")


if __name__ == "__main__":
    main()
