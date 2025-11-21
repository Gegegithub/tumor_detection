# Détection de Tumeurs Cérébrales par Deep Learning

Projet de classification d'images IRM du cerveau pour détecter automatiquement la présence de tumeurs cérébrales à l'aide de techniques de Deep Learning.

## Aperçu du projet

Ce projet explore l'application du Deep Learning à l'imagerie médicale pour la classification de tumeurs cérébrales à partir d'images IRM. Deux approches distinctes ont été comparées :

1. **CNN personnalisé** : Un réseau de neurones convolutionnel conçu et entraîné from scratch
2. **Transfer Learning avec ResNet50** : Utilisation d'un modèle pré-entraîné sur ImageNet

Le projet met en évidence les avantages et limitations de chaque approche dans un contexte de données limitées, typique de l'imagerie médicale.

## Dataset

Le dataset contient des images IRM du cerveau classées en **4 catégories** :

- **Glioma** : Tumeur cérébrale de type gliome
- **Meningioma** : Tumeur de type méningiome
- **Pituitary** : Tumeur de l'hypophyse
- **No Tumor** : Absence de tumeur

**Répartition** :
- Training : 2870 images
- Test : 394 images
- Taille des images : 224x224 pixels

## Architecture des modèles

### CNN Personnalisé (From Scratch)

Architecture composée de **5 blocs convolutionnels** suivis de couches fully-connected :

```
Conv2D (32) → MaxPooling2D
Conv2D (64) → MaxPooling2D
Conv2D (128) → MaxPooling2D
Conv2D (256) → MaxPooling2D
Conv2D (512) → MaxPooling2D
Flatten → Dense (128) → BatchNorm → Dropout (0.5) → Dense (4, softmax)
```

**Techniques de régularisation** :
- Batch Normalization & Dropout
- Early Stopping & ReduceLROnPlateau
- Data Augmentation (rotation, zoom, translation)

### Transfer Learning - ResNet50

```
ResNet50 (pré-entraîné, gelé)
GlobalAveragePooling2D
Dense (128) → BatchNorm → Dropout (0.5)
Dense (4, softmax)
```

**Stratégie** :
- Couches ResNet50 gelées (poids ImageNet conservés)
- Entraînement uniquement de la tête de classification
- Data Augmentation renforcée (rotation ±10°, zoom, flip)

## Résultats

| Modèle | Test Accuracy | Test Loss |
|--------|---------------|-----------|
| **CNN Personnalisé** | 64.72% | 1.2796 |
| **ResNet50 (Transfer Learning)** | 70.81% | 1.2345 |

**Observations** :
- Le modèle ResNet50 converge plus rapidement et généralise mieux
- Le CNN personnalisé souffre de surapprentissage malgré les techniques de régularisation
- Le Transfer Learning démontre sa supériorité avec des données limitées
- Les performances reflètent les défis typiques du Deep Learning en imagerie médicale

## Technologies utilisées

- **Deep Learning** : TensorFlow, Keras
- **Modèles** : CNN personnalisé, ResNet50
- **Data Science** : NumPy, Pandas, Scikit-learn
- **Visualisation** : Matplotlib, Seaborn
- **Expérimentation** : ClearML
- **Environnement** : Google Colab, Jupyter Notebook

## Structure du projet

```
tumor_detection/
├── custom_cnn.ipynb           # CNN personnalisé from scratch
├── ResNet50_2.ipynb           # Transfer Learning avec ResNet50
├── README.md                  # Documentation
└── dataset_brain tumor/       # Dataset d'images IRM
    ├── Training/              # Données d'entraînement (4 classes)
    └── Testing/               # Données de test (4 classes)
```

## Apprentissages clés

### Défis rencontrés

**Surapprentissage du CNN personnalisé** :
- Dataset de petite taille limitant l'apprentissage from scratch
- Solution : combinaison de Dropout, Batch Normalization, Early Stopping et Data Augmentation

**Généralisation limitée** :
- Le modèle from scratch reste limité malgré l'augmentation de données
- Le Transfer Learning s'impose comme solution efficace pour les petits datasets

### Compétences développées

- Conception et implémentation d'architectures CNN personnalisées
- Maîtrise du Transfer Learning et fine-tuning
- Techniques de régularisation pour éviter l'overfitting
- Expérimentation systématique et tracking avec ClearML
- Compréhension des enjeux du Deep Learning en imagerie médicale

## Perspectives d'amélioration

- Fine-tuning progressif des couches ResNet50
- Test d'architectures alternatives (EfficientNet, Vision Transformers)
- Gestion du déséquilibre de classes (class weights, focal loss)
- Explicabilité des prédictions (Grad-CAM, SHAP)
- Déploiement via API (Flask/FastAPI)
