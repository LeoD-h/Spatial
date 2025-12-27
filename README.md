# Spatial

Scripts Python inspirés du notebook `Spatial_Test.ipynb` pour préparer le dataset Galaxy Zoo, entraîner YOLOv8 et tester le modèle via CLI ou interface graphique.

## Organisation

- `app/` : exécutables
  - `train.py` : préparation dataset + entraînement YOLO (CUDA/NVIDIA par défaut).
  - `run_inference.py` : inférences (image locale, URL ou batch avec stats).
  - `gui.py` : interface Tk.
- `run.py` : point d’entrée unique pour lancer l’interface graphique (racine).
- `spatial/` : utilitaires (préparation dataset, prédiction, stats).
- `TestNotebook/` : notebooks d’origine.
- `data/raw/` : données brutes (zip + csv).
- `data/processed/` : dataset YOLO généré (train/val).
- `models/` : poids entraînés.
- `outputs/` : images annotées produites par scripts/GUI.
- `misc/` : divers.

## Prérequis

- Python 3.9+
- `pip install -r requirements.txt`
- Placer les données brutes dans `data/raw/` :
  - `images_training_rev1.zip`
  - `training_solutions_rev1.csv`
- Poids fournis dans `models/galaxy_model_v2_expert.pt` (ou utilisez vos propres poids).

## Préparer et entraîner

```bash
python app/train.py \
  --zip-path data/raw/images_training_rev1.zip \
  --labels-csv data/raw/training_solutions_rev1.csv \
  --dataset-size 12000 \
  --epochs 40 --batch 64 --img-size 416 \
  --device 0  # GPU NVIDIA par défaut (mettre 'cpu' si besoin)
```

- Le dataset YOLO est écrit dans `data/processed/galaxy_expert`.
- Le meilleur modèle est copié dans `models/galaxy_fast_expert_best.pt`.
- Ajouter `--prepare-only` pour ne créer que le dataset.

## Inférence en ligne de commande

- Image locale :

```bash
python app/run_inference.py --model models/galaxy_model_v2_expert.pt --image path/to/image.jpg
```

- Image distante :

```bash
python app/run_inference.py --url https://.../galaxy.jpg
```

- Batch + stats sur le jeu de validation (ex. 30 images aléatoires) :

```bash
python app/run_inference.py --count 30 \
  --folder data/processed/galaxy_expert/val/images \
  --labels data/processed/galaxy_expert/val/labels
```

Les images annotées sont sauvegardées dans `outputs/predictions`.

## Interface graphique

```bash
python run.py
```

Fonctionnalités dans la fenêtre :
- Choisir un modèle (.pt).
- Image aléatoire du jeu de validation (nécessite `data/processed/...`).
- Sélection d'une image locale.
- Analyse d'une URL collée.
- Batch sur le jeu de validation (nombre libre, ex. 30) avec statistiques.

Notes :
- Assurer un modèle disponible (`models/galaxy_model_v2_expert.pt` ou vos poids entraînés).
- Générer le dataset avec `app/train.py --prepare-only` si besoin pour les tests val.
- Les images annotées générées par la GUI sont écrites dans `outputs/gui`.

## Notes

- Entraînement optimisé pour GPU NVIDIA (CUDA). Passer `--device cpu` si vous n’avez pas de GPU.
- Classes : `elliptique`, `spirale`, `profil`, `artefact`.
