# Classification des Sentiments avec BERT

## Table des Matières
1. [Introduction](#introduction)
2. [Données](#données)
3. [Entraînement du Modèle](#entraînement-du-modèle)
4. [Évaluation](#évaluation)
5. [Résultats](#résultats)
6. [Usage](#usage)
7. [Conclusion](#conclusion)

## Introduction
Ce projet vise à démontrer l'utilisation de BERT pour la classification des sentiments dans un ensemble de données de textes. BERT est un modèle de langage puissant développé par Google, et nous ajoutons une couche linéaire sur le modèle pré-entraîné pour adapter le modèle à notre tâche spécifique de classification.

## Données
Les données utilisées dans ce projet proviennent de fichiers CSV contenant des textes et des sentiments associés. Les sentiments peuvent être "positif", "négatif" ou "neutre". Les données ont été nettoyées et équilibrées pour améliorer les performances du modèle.

### Nettoyage des Données
Les étapes de nettoyage des données incluent :
- Suppression des caractères spéciaux et des ponctuations.
- Conversion des textes en minuscules.
- Suppression des mots vides (stop words).
- Lemmatisation des mots pour réduire les mots à leur forme de base.

### Rééquilibrage des Classes
Les données sont rééquilibrées pour éviter le déséquilibre des classes en utilisant la technique de suréchantillonnage. Cela implique la duplication des exemples des classes minoritaires pour obtenir une distribution uniforme des classes.

### Description des Données
Les colonnes des fichiers CSV incluent :
- `text` : Le texte à analyser.
- `sentiment` : Le sentiment associé au texte (positif, négatif, neutre).

## Entraînement du Modèle
Nous utilisons le modèle BERT pré-entraîné et ajoutons une couche linéaire pour la classification des sentiments. Le modèle est entraîné sur les données de formation et évalué sur les données de test.

### Préparation des Données
Les textes sont tokenisés en utilisant le tokenizer de BERT et les données sont transformées en tenseurs PyTorch pour l'entraînement.

### Hyperparamètres
- **Nombre d'époques** : 10
- **Taux d'apprentissage** : 3e-5
- **Taux de dropout** : 0.5
- **Taille de batch** : 16 pour l'entraînement, 8 pour l'évaluation

### Entraînement
L'entraînement est effectué en utilisant l'optimiseur AdamW et une fonction de perte de type CrossEntropyLoss. Un scheduler est utilisé pour ajuster le taux d'apprentissage au cours des époques.

## Évaluation
Le modèle est évalué en utilisant les métriques de précision (accuracy) et de perte (loss) sur l'ensemble de test. Les prédictions sont comparées aux étiquettes réelles pour calculer ces métriques.

## Résultats
Les résultats de l'entraînement montrent une diminution de la perte d'entraînement au fil des époques, mais la perte de validation reste élevée, indiquant un possible surapprentissage. Les mesures de précision et de perte sont rapportées pour chaque époque.

### Exemple de Résultats
