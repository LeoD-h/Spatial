import argparse
import random
import threading
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, TOP, BOTTOM, Button, Entry, Frame, Label, StringVar, Tk, filedialog
from tkinter import ttk

from PIL import Image, ImageTk

from spatial.data import CLASS_NAMES
from spatial.inference import download_image, evaluate_batch, load_model, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface graphique pour Spatial.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/galaxy_model_v2_expert.pt"),
        help="Poids du modele YOLO.",
    )
    parser.add_argument(
        "--dataset-images",
        type=Path,
        default=Path("data/processed/galaxy_expert/val/images"),
        help="Images du set de validation pour les tests rapides.",
    )
    parser.add_argument(
        "--dataset-labels",
        type=Path,
        default=Path("data/processed/galaxy_expert/val/labels"),
        help="Labels (optionnels) pour calculer la precision en batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/gui"),
        help="Ou sauvegarder les images annotees.",
    )
    return parser.parse_args()


class SpatialApp:
    def __init__(self, root: Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.model_path = args.model
        self.model = load_model(self.model_path)
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_label = None
        self.current_photo = None

        root.title("Spatial - Galaxy detector")
        root.geometry("900x700")

        top_frame = Frame(root)
        top_frame.pack(side=TOP, fill=BOTH, pady=10)

        Button(
            top_frame,
            text="Image aleatoire du test set",
            command=self._random_val_image,
        ).pack(side=LEFT, padx=5)

        Button(
            top_frame,
            text="Choisir image locale",
            command=self._pick_local_image,
        ).pack(side=LEFT, padx=5)

        Button(
            top_frame,
            text="Choisir un modele",
            command=self._pick_model,
        ).pack(side=LEFT, padx=5)

        url_frame = Frame(root)
        url_frame.pack(side=TOP, fill=BOTH, pady=5)
        Label(url_frame, text="URL:").pack(side=LEFT)
        self.url_var = StringVar()
        Entry(url_frame, textvariable=self.url_var, width=60).pack(side=LEFT, padx=5)
        Button(url_frame, text="Analyser URL", command=self._predict_url).pack(
            side=LEFT, padx=5
        )

        batch_frame = Frame(root)
        batch_frame.pack(side=TOP, fill=BOTH, pady=5)
        Label(batch_frame, text="Batch (val set) :").pack(side=LEFT)
        self.batch_var = StringVar(value="30")
        Entry(batch_frame, textvariable=self.batch_var, width=6).pack(side=LEFT, padx=5)
        Button(batch_frame, text="Lancer stats", command=self._run_batch).pack(
            side=LEFT, padx=5
        )

        self.status_var = StringVar(value="Pret.")
        Label(root, textvariable=self.status_var).pack(side=BOTTOM, pady=5)

        self.image_label = Label(root)
        self.image_label.pack(fill=BOTH, expand=True, pady=10)

        self.log_var = StringVar(value="")
        self.log_label = Label(root, textvariable=self.log_var, justify=LEFT)
        self.log_label.pack(side=BOTTOM, pady=5)

        self.model_var = StringVar(value=f"Modele: {self.model_path}")
        Label(root, textvariable=self.model_var).pack(side=BOTTOM, pady=2)

    def _set_status(self, msg: str) -> None:
        self.status_var.set(msg)
        self.root.update_idletasks()

    def _show_image(self, img_path: Path) -> None:
        img = Image.open(img_path)
        img.thumbnail((850, 550))
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.current_photo)

    def _log_detections(self, detections) -> None:
        if not detections:
            self.log_var.set("Aucune detection.")
            return
        lines = [
            f"{det['class_name']} (id={det['class_id']}) conf={det['confidence']:.2f}"
            for det in detections
        ]
        self.log_var.set("\n".join(lines))

    def _predict_path(self, path: Path) -> None:
        self._set_status(f"Analyse de {path.name}...")
        out_path, dets = predict_image(
            self.model,
            path,
            self.output_dir,
            conf=0.25,
            iou=0.45,
        )
        self._show_image(out_path)
        self._log_detections(dets)
        self._set_status("Termine.")

    def _pick_model(self):
        path = filedialog.askopenfilename(
            title="Choisir un modele (.pt)",
            filetypes=[("PyTorch weights", "*.pt")],
        )
        if path:
            self._set_status("Chargement du modele...")
            self.model_path = Path(path)
            self.model = load_model(self.model_path)
            self.model_var.set(f"Modele: {self.model_path}")
            self._set_status("Modele charge.")
    def _run_threaded(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    def _random_val_image(self):
        images = list(self.args.dataset_images.glob("*.jpg"))
        if not images:
            self._set_status(f"Aucune image dans {self.args.dataset_images}")
            return
        self._run_threaded(self._predict_path, random.choice(images))

    def _pick_local_image(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.jpg *.png *.jpeg")],
        )
        if path:
            self._run_threaded(self._predict_path, Path(path))

    def _predict_url(self):
        url = self.url_var.get().strip()
        if not url:
            self._set_status("Merci de coller une URL.")
            return

        def _task():
            self._set_status("Telechargement en cours...")
            try:
                img_path = download_image(url)
            except Exception as exc:
                self._set_status(f"Echec du telechargement: {exc}")
                return
            self._predict_path(img_path)

        self._run_threaded(_task)

    def _run_batch(self):
        try:
            count = int(self.batch_var.get())
        except ValueError:
            self._set_status("Nombre invalide.")
            return

        images = list(self.args.dataset_images.glob("*.jpg"))
        if not images:
            self._set_status(f"Aucune image dans {self.args.dataset_images}")
            return

        sampled = random.sample(images, k=min(len(images), count))

        def _task():
            self._set_status(f"Batch de {len(sampled)} images...")
            label_dir = self.args.dataset_labels if self.args.dataset_labels.exists() else None
            summary = evaluate_batch(
                self.model,
                sampled,
                save_dir=self.output_dir,
                label_dir=label_dir,
                conf=0.25,
                iou=0.45,
            )

            lines = [
                f"Images: {summary['total_images']}",
                f"Detection rate: {summary['detection_rate']:.1f}%",
            ]
            if label_dir:
                lines.append(f"Accuracy: {summary['accuracy']:.2f}%")
            lines.append("Repartition:")
            for cid, count in summary["counts"].items():
                lines.append(f"- {CLASS_NAMES.get(cid, cid)}: {count}")
            lines.append(f"Outputs: {summary['output_dir']}")
            self.log_var.set("\n".join(lines))
            self._set_status("Batch termine.")

        self._run_threaded(_task)


def main():
    args = parse_args()
    root = Tk()
    SpatialApp(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
