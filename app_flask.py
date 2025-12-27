"""
Interface Flask pour Spatial - Détection de galaxies
"""
import io
import base64
import random
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import numpy as np

from spatial.data import CLASS_NAMES
from spatial.inference import download_image, load_model, predict_image

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Configuration
MODEL_PATH = Path("models/galaxy_model_v2_expert.pt")
VAL_IMAGES_DIR = Path("data/processed/galaxy_expert/val/images")
OUTPUT_DIR = Path("outputs/flask")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger le modèle au démarrage
print("Chargement du modèle...")
model = load_model(MODEL_PATH)
print("Modèle chargé avec succès!")


def image_to_base64(image_path):
    """Convertir une image en base64 pour l'affichage HTML"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html', class_names=CLASS_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """Prédiction sur une image uploadée"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Lire l'image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Sauvegarder temporairement
        temp_path = OUTPUT_DIR / "temp_input.jpg"
        img.save(temp_path)
        
        # Prédiction
        output_path, detections = predict_image(model, temp_path, OUTPUT_DIR)
        
        # Convertir les détections au format attendu par le frontend
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                'class': det['class_name'],
                'confidence': f"{det['confidence']:.2%}"
            })
        
        # Convertir l'image de sortie en base64
        output_b64 = image_to_base64(output_path)
        
        return jsonify({
            'success': True,
            'image': output_b64,
            'detections': formatted_detections,
            'count': len(formatted_detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Prédiction sur une image depuis URL"""
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'URL non fournie'}), 400
        
        # Télécharger l'image
        temp_path = download_image(url)
        
        # Prédiction
        output_path, detections = predict_image(model, temp_path, OUTPUT_DIR)
        
        # Convertir les détections au format attendu par le frontend
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                'class': det['class_name'],
                'confidence': f"{det['confidence']:.2%}"
            })
        
        # Convertir l'image de sortie en base64
        output_b64 = image_to_base64(output_path)
        
        return jsonify({
            'success': True,
            'image': output_b64,
            'detections': formatted_detections,
            'count': len(formatted_detections)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/random_test', methods=['POST'])
def random_test():
    """Prédiction sur une image aléatoire du dataset de test"""
    try:
        # Trouver toutes les images de validation
        val_images = list(VAL_IMAGES_DIR.glob("*.jpg"))
        if not val_images:
            return jsonify({'error': 'Aucune image de validation trouvée'}), 404
        
        # Sélectionner une image aléatoire
        random_image = random.choice(val_images)
        
        # Lire le vrai label si disponible
        label_dir = Path("data/processed/galaxy_expert/val/labels")
        label_path = label_dir / f"{random_image.stem}.txt"
        true_class = None
        true_class_name = None
        
        if label_path.exists():
            with open(label_path, "r") as f:
                line = f.readline().strip()
                if line:
                    true_class = int(line.split()[0])
                    true_class_name = CLASS_NAMES.get(true_class, "Inconnu")
        
        # Prédiction
        output_path, detections = predict_image(model, random_image, OUTPUT_DIR)
        
        # Convertir les détections au format attendu par le frontend
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                'class': det['class_name'],
                'confidence': f"{det['confidence']:.2%}"
            })
        
        # Convertir l'image de sortie en base64
        output_b64 = image_to_base64(output_path)
        
        return jsonify({
            'success': True,
            'image': output_b64,
            'detections': formatted_detections,
            'count': len(formatted_detections),
            'filename': random_image.name,
            'true_class': true_class_name
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Spatial Galaxy Detector - Interface Web")
    print("="*60)
    print(f"Modèle: {MODEL_PATH}")
    print(f"Classes détectables: {', '.join(CLASS_NAMES.values())}")
    print("\nOuvrez votre navigateur à: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

