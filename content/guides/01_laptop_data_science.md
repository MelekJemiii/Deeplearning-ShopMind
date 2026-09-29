---
title: Choisir un laptop pour la data science et le machine learning
topic: use_case_guide
device: laptop
lang: fr
---
# Choisir un laptop pour la data science et le machine learning

La data science combine manipulation de données (pandas, SQL), visualisation et entraînement de modèles. Les besoins matériels dépendent surtout de la taille des données et du type de modèles : du machine learning classique (scikit-learn) demande peu de ressources, alors que le deep learning (PyTorch, TensorFlow) profite fortement d'une carte graphique.

## Mémoire vive (RAM)

La RAM est le composant le plus important pour la data science. Les données chargées avec pandas sont stockées en mémoire, souvent avec un surcoût important par rapport à la taille du fichier.

- 8 Go : insuffisant, le système ralentit dès que plusieurs outils sont ouverts (navigateur, IDE, Jupyter).
- 16 Go : minimum recommandé pour des jeux de données moyens.
- 32 Go : confortable pour de gros jeux de données, Docker et plusieurs notebooks en parallèle.

Privilégiez un modèle dont la RAM est extensible (emplacements SO-DIMM) plutôt que soudée.

## Processeur (CPU)

Un processeur récent de milieu ou haut de gamme avec au moins 6 cœurs convient : Intel Core i5/i7 ou Core Ultra 5/7, AMD Ryzen 5/7. Plus de cœurs accélèrent le prétraitement des données et l'entraînement des modèles classiques qui parallélisent sur CPU.

## Carte graphique (GPU)

Pour le deep learning, un GPU NVIDIA est fortement conseillé, car la majorité des frameworks utilisent CUDA, la technologie de calcul de NVIDIA. Les GPU AMD et Intel sont moins bien supportés.

- La quantité de mémoire vidéo (VRAM) limite la taille des modèles : 6 Go minimum, 8 Go ou plus conseillé.
- Pour du machine learning classique uniquement, un GPU dédié n'est pas nécessaire.

Alternative : utiliser des GPU en ligne (Google Colab, Kaggle) et choisir un laptop plus léger et moins cher.

## Stockage

Un SSD NVMe est indispensable : il accélère le chargement des données, des bibliothèques et du système. Prévoyez 512 Go minimum, 1 To si vous stockez des jeux de données localement.

## Écran et autonomie

Un écran de 15 pouces en Full HD minimum facilite le travail avec plusieurs fenêtres. Les laptops équipés d'un GPU puissant ont généralement une autonomie plus faible et sont plus lourds : c'est le compromis principal à accepter.

## Système d'exploitation

Linux est très utilisé en data science ; sous Windows, WSL2 permet d'utiliser un environnement Linux. Vérifiez la compatibilité du matériel si vous comptez installer Linux directement.

## Résumé

Priorités dans l'ordre : RAM (16 Go minimum), GPU NVIDIA si deep learning, SSD NVMe 512 Go, CPU 6 cœurs ou plus.
