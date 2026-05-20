# Prédiction du Risque Cardiovasculaire via Machine Learning

##  Description du Projet
Ce projet consiste en la conception et l'implémentation d'une application interactive permettant la prédiction du risque cardiaque à partir de données médicales, il a été développé dans le cadre d'un stage d'initiation au sein du Centre Hospitalier Universitaire (CHU) Hassan II de Fès.

L'application fournit une interface simple pour estimer rapidement le risque de maladies cardiovasculaires, accessible aussi bien aux professionnels de santé qu'aux patients.

##  Fonctionnalités Principales
L'application propose deux parcours utilisateurs distincts :
**Profil Patient :** Permet la saisie des données médicales pour obtenir une estimation personnalisée du risque cardiovasculaire.L'utilisateur peut également consulter des recommandations adaptées et comparer ses résultats aux normes médicales de référence.
**Profil Médecin :** Permet d'exploiter les données pour générer une prédiction, consulter l'historique des évaluations des patients, effectuer des recherches par identifiant et analyser les variables influentes dans le résultat.

##  Technologies et Outils
**Langage :** Python.
**Interface Web :** Streamlit
**Manipulation des données :** Pandas, NumPy
**Visualisation :** Matplotlib, Seaborn
**Machine Learning :** Scikit-learn
**Sauvegarde du modèle :** Joblib

##  Données et Modélisation
Le modèle a été entraîné sur le jeu de données public *Heart Disease Dataset*. 

Plusieurs algorithmes d'apprentissage supervisé ont été implémentés et comparés pour la classification:
* K-Nearest Neighbors (KNN) 
* Logistic Regression
* Decision Tree
* Random Forest 

**Résultats :** Le modèle final retenu est le **Random Forest**, car il offre les meilleures performances globales et une robustesse accrue.
*  **Accuracy (Précision globale) :** 88.04%
*  **F1-Score :** 89.52% 

##  Architecture du Code
Le code source est organisé en deux fichiers distincts pour une meilleure lisibilité:
* `model_training.py` : Script dédié au nettoyage des données, à l'entraînement des modèles de Machine Learning, à l'optimisation des hyperparamètres et à la sauvegarde du modèle final.
* `interface.py` : Script gérant l'interface utilisateur Streamlit, le chargement du modèle sauvegardé, et la logique d'interaction pour les profils patient et médecin.
