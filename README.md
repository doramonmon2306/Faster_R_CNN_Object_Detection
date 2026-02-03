# 🖼️ Faster R-CNN Object Detection System 🖼️

## 🚀 VERSION EN FRANÇAIS 🚀

Ce projet présente une implémentation d’un système de **détection d’objets basé sur Faster R-CNN**, entraîné et évalué sur le jeu de données **PASCAL VOC 2012**. Le modèle est **pré-entraîné sur COCO** puis **fine-tuné sur VOC 2012**.

### 🔬 Méthodologie
- Utilisation de **Faster R-CNN** avec un backbone **MobileNetV3 + FPN** (Torchvision)
- Prétraitement et augmentation des données (ColorJitter, transformations géométriques)
- Fine-tuning à partir de poids pré-entraînés COCO
- Évaluation basée sur les métriques **Mean Average Precision**

### 🛠️ Outils et bibliothèques
PyTorch, Torchvision, TorchMetrics, NumPy, TensorBoard

### 📊 Résultats
- Meilleures performances obtenues sur le jeu de validation sont **mAP (IoU 0.5:0.95)** : **0.3**, **mAP_0.5** : **0.5**, **mAP_0.75** : **0.2**
- Suivi des pertes d’entraînement et des métriques de validation via TensorBoard
- Sauvegarde automatique du meilleur modèle selon le score **Mean Average Precision**

Ce projet illustre un pipeline complet de détection d’objets, allant du prétraitement des données à l’évaluation.


## 🚀 ENGLISH VERSION 🚀

This project presents an implementation of an **object detection system based on Faster R-CNN**, trained and evaluated on the **PASCAL VOC 2012** dataset.  
The model is **pretrained on COCO** and **fine-tuned on VOC 2012**.

### 🔬 Methodology
- Use of **Faster R-CNN** with a **MobileNetV3 + FPN** backbone (Torchvision)
- Data preprocessing and augmentation (ColorJitter, geometric transforms)
- Fine-tuning from COCO pretrained weights
- Performance evaluation using **Mean Average Precision**

### 🛠️ Tools and Libraries
PyTorch, Torchvision, TorchMetrics, NumPy, TensorBoard

### 📊 Results
- Best validation performances are **mAP (IoU 0.5:0.95)** : **0.3**, **mAP_0.5** : **0.5**, **mAP_0.75** : **0.2**
- Monitoring of training loss and validation metrics via TensorBoard
- Automatic saving of the best-performing model based on Mean Average Precision

This project demonstrates a **complete end-to-end object detection pipeline**, from data preprocessing to evaluation.


