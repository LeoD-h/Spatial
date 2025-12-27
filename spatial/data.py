import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

# Class mapping reused across training and inference
CLASS_NAMES: Dict[int, str] = {
    0: "elliptique",
    1: "spirale",
    2: "profil",
    3: "artefact",
}


def _assign_class(row) -> int:
    """
    Reproduces the heuristic from the original notebook to build a single label
    from Galaxy Zoo probabilities.
    """
    if row["Class1.3"] > 0.4:
        return 3
    if row["Class2.1"] > 0.5:
        return 2
    if row["Class1.2"] > 0.5 and (row["Class4.1"] > 0.4 or row["Class3.1"] > 0.4):
        return 1
    if row["Class1.1"] > 0.5:
        return 0

    main_probs = {"0": row["Class1.1"], "1": row["Class1.2"], "3": row["Class1.3"]}
    return int(max(main_probs, key=main_probs.get))


def _extract_images(zip_path: Path, extract_to: Path) -> Path:
    """
    Unpacks the raw image archive if needed and returns the directory containing
    the JPEG files.
    """
    extract_to.mkdir(parents=True, exist_ok=True)
    images_dir = extract_to / "images_training_rev1"
    if images_dir.exists():
        return images_dir

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    return images_dir


def prepare_dataset(
    zip_path: Path,
    labels_csv: Path,
    output_dir: Path,
    dataset_size: Optional[int] = 12000,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[Path, Path]:
    """
    Builds a YOLO-ready dataset from the Galaxy Zoo archive.

    Args:
        zip_path: Path to images_training_rev1.zip.
        labels_csv: Path to training_solutions_rev1.csv.
        output_dir: Where the YOLO train/val folders will be written.
        dataset_size: Number of samples to keep (None for full dataset).
        val_split: Fraction used for validation.
        seed: Random seed for reproducibility.

    Returns:
        dataset_yaml: Path to the generated dataset.yaml.
        output_dir: The directory containing train/ and val/ folders.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    images_root = _extract_images(zip_path, output_dir.parent / "galaxy_data")
    df = pd.read_csv(labels_csv)

    if dataset_size is not None:
        df = df.sample(n=min(len(df), dataset_size), random_state=seed)

    train_df, val_df = train_test_split(df, test_size=val_split, random_state=seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for sub in ("train/images", "train/labels", "val/images", "val/labels"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    def _process_split(split_df: pd.DataFrame, split_name: str) -> int:
        processed = 0
        for _, row in split_df.iterrows():
            img_id = str(int(row["GalaxyID"]))
            img_name = f"{img_id}.jpg"
            src_path = images_root / img_name
            if not src_path.exists():
                continue

            shutil.copy2(src_path, output_dir / split_name / "images" / img_name)
            with open(output_dir / split_name / "labels" / f"{img_id}.txt", "w") as f:
                # Single box covering the full frame as in the notebook
                f.write(f"{_assign_class(row)} 0.5 0.5 0.6 0.6\n")
            processed += 1
        return processed

    train_count = _process_split(train_df, "train")
    val_count = _process_split(val_df, "val")

    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("names:\n")
        for idx, name in CLASS_NAMES.items():
            f.write(f"  {idx}: {name}\n")

    print(
        f"Dataset ready in {output_dir} "
        f"({train_count} train / {val_count} val images, val split={val_split})"
    )

    return dataset_yaml, output_dir
