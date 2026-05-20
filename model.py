import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Charger les donnees
df = pd.read_csv('heart.csv')

#-----------------------------------------------Analyse exploratoire -----------------------------------------------

pd.set_option('display.max_columns', None)       # pour afficher toutes les colonnes
pd.set_option('display.width', None)             #pour enlever la coupure de texte

print(df.head())     #afficher les premières lignes par defaut 5

print("\n Nombre de lignes et colonnes :", df.shape)    # Dimensions du dataset

print("\n",df.info())     # Noms des colonnes et types

# Vérifier s'il y a des valeurs nulles
print(df.isnull().sum())

print(df.describe())
print(df.describe(include='object'))  # pour les colonnes de type string (object)

numeric_columns = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Remplacer les 0 dans 'Cholesterol' et 'RestingBP' avec la médiane de leurs valeurs non-nulles
for col in ['Cholesterol', 'RestingBP']:
    median_val = df[df[col] != 0][col].median()  # Calculer la médiane des valeurs non nulles pour la colonne spécifique
    df[col] = df[col].replace(0, median_val)     # Remplacer les 0 par la médiane calculée

print(df[['Cholesterol', 'RestingBP']].describe().loc['min'])

print(f"\nNombre de lignes dupliquées : {df.duplicated().sum()}")   # Vérifier les doublons

print("Valeurs uniques :", df['HeartDisease'].unique())
print(df['HeartDisease'].value_counts())

for col in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x=col, hue='HeartDisease', fill=True, palette='Set1')
    plt.title(f'Distribution de {col} selon HeartDisease')
    plt.xlabel(col)
    plt.grid(True)
    plt.show()

for col in categorical_columns:
    sns.countplot(x=col, hue='HeartDisease', data=df)
    plt.title(f'{col} vs HeartDisease')
    plt.xticks(rotation=45)
    plt.show()

 #-----------------------------------------------Model Training & Evaluation--------------------------------------------

# Copier le DataFrame original pour le modifier
data = df.copy()

# On encode les colonnes comme Sex, ChestPainType, etc. en nombres
from sklearn.preprocessing import LabelEncoder
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data.drop('HeartDisease', axis=1)  # Variables explicatives
y = data['HeartDisease']        # Cible

#matrice de correlation
corr_matrix = data.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,          # affiche la valeur dans chaque case
    fmt=".2f",           # deux décimales
    cmap="coolwarm",     # dégradé bleu‑rouge
    center=0,            # zéro au milieu de la palette
    linewidths=0.5
)
plt.title("Matrice de corrélation des variables")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
# Séparer les données en train et test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on train
X_test = scaler.transform(X_test)        # Use same scaling on test (sans .fit !)

#---------------------------------KNN---------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Choix du meilleur (k) pour le modèle KNN en se basant sur l'accuracy
k_values = list(range(1, 21))  # On teste k de 1 à 20
accuracies_k = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies_k.append(acc)

# Afficher le graphe
plt.figure(figsize=(10,5))
plt.plot(k_values, accuracies_k, marker='o', linestyle='dashed', color='blue')
plt.title('Choix du meilleur k')
plt.xlabel('Valeur de k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Afficher le meilleur k
best_k = k_values[np.argmax(accuracies_k)]
print(f"\n Meilleure valeur de k = {best_k}")

model_knn =KNeighborsClassifier(n_neighbors=best_k)

#---------------------------------regression logistic ---------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
f1_scores = []

for c in C_values:
    model = LogisticRegression(C=c, solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1_scores.append(f1_score(y_test, y_pred))

plt.plot(C_values, f1_scores, marker='o', color='green')
plt.title("Choix du meilleur C pour la régression logistique")
plt.xlabel("Valeur de C")
plt.ylabel("F1 Score")
plt.xscale("log")
plt.grid(True)
plt.show()

best_c = C_values[np.argmax(f1_scores)]
print(f"\n Meilleure valeur de C = {best_c}")

# --- Interpreter les Coefficients de la Regression Logistique---
print("\n les Coefficients de la Regression Logistique")
model_lr = LogisticRegression(C=best_c, solver='liblinear', max_iter=1000)
model_lr.fit(X_train, y_train)
# Get the coefficients and feature names
coefficients = model_lr.coef_[0]
feature_names = X.columns

# Create a DataFrame for better readability
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by the absolute value of coefficients to see the most impactful features
coefficients_df['Abs_Coefficient'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

print(coefficients_df)

# Visualize the coefficients
plt.figure(figsize=(12, 7))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df, palette='coolwarm', hue='Feature', legend=False)
plt.title('Logistic Regression Feature Coefficients (on Scaled Data)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

#--------------------------------- arbre de decision ---------------------------------

from sklearn.tree import DecisionTreeClassifier
# Essayer des profondeurs de 1 à 10
depths = list(range(1, 11))
accuracies = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Affichage de la courbe
plt.figure(figsize=(10, 6))
plt.plot(depths, accuracies, marker='o', color='purple')
plt.title('Validation de max_depth')
plt.xlabel('Profondeur (max_depth)')
plt.ylabel('Accuracy sur le test set')
plt.grid(True)
plt.xticks(depths)
plt.show()

best_depth = depths[np.argmax(accuracies)]
print(f"\n Meilleure Profondeur (max_depth) = {best_depth}")

model_dt=DecisionTreeClassifier(max_depth=best_depth)

#--------------------------------- random forest ---------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# Trouver le meilleur nombre d'estimators
estimators = [10, 50, 100, 150, 200]
f1_scores = []

for n in estimators:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

plt.plot(estimators, f1_scores, marker='o', color='orange')
plt.title("Choix du meilleur n_estimators pour Random Forest")
plt.xlabel("n_estimators")
plt.ylabel("F1 Score")
plt.grid(True)
plt.show()

best_n = estimators[np.argmax(f1_scores)]
print(f"\n Meilleur n_estimators : {best_n}")

model_rf=RandomForestClassifier(n_estimators=best_n, random_state=42)
model_rf.fit(X_train, y_train)

models = {
    "KNN" : model_knn ,
    "Logistic Regression" : model_lr,
    "Decision Tree" : model_dt,
    "Random Forest": model_rf
}

#-----------------------------------------------Évaluation finale de tous les modèles----------------------------------

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, jaccard_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# On choisit les scores à calculer pour cross validation
scores_cv = {
    "Accuracy": make_scorer(accuracy_score),
    "F1-Score": make_scorer(f1_score),
    "Recall": make_scorer(recall_score),
    "Précision": make_scorer(precision_score),
    "Jaccard": make_scorer(jaccard_score)
}

cv_results = []   # Tableau pour stocker les résultats des scores pour cross validation

# ✅ Cross-validation sur X_train seulement
for name, model in models.items():
    scores = {}
    for metric_name, scorer in scores_cv.items():
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer)
        scores[metric_name] = cv_score.mean()
    scores["Modèle"] = name
    cv_results.append(scores)

# Résultats sous forme de DataFrame
cv_df = pd.DataFrame(cv_results).set_index("Modèle")
cv_df = cv_df.sort_values(by="F1-Score", ascending=False)

# Affichage
print("\n📊 Résultats Cross-Validation (moyennes des 5 folds) :\n")
print(cv_df)

# 📈 Affichage graphique
cv_df.plot(kind="bar", figsize=(12, 6))
plt.title("Performances moyennes en cross-validation (cv=5)")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

def evaluate_model(model, X_train, y_train, X_test, y_test, name=""):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📌 Résultats pour {name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 🎯 Visualisation arbre de décision
    if isinstance(model, DecisionTreeClassifier):
        from sklearn.tree import plot_tree
        plt.figure(figsize=(20, 10))
        plot_tree(model_dt, filled=True, feature_names=X.columns, class_names=['NoDisease', 'Disease'], rounded=True)
        plt.show()

# 🔍 Feature importance (si applicable)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importances_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("\n🔎 Importances des features :")
        print(importances_df)

        # Graphique
        plt.figure(figsize=(8, 5))
        sns.barplot(x='Importance', y='Feature', data=importances_df, hue='Feature', palette='viridis')
        plt.title(f'Feature Importances pour {name}')
        plt.grid(True)
        plt.show()

    # Cross-validation F1
    cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

    return {
        "Modèle": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Jaccard": jaccard_score(y_test, y_pred),
        "CV F1": cv_f1
    }

results = []

for name, model in models.items():
    res = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    results.append(res)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)  # ou selon "CV F1"
print(results_df)

# ---------------- Graphique en barres comparatif ----------------
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = range(len(results_df))

plt.bar(index, results_df["F1-Score"], width=bar_width, label='F1-Test', color='skyblue')
plt.bar([i + bar_width for i in index], results_df["CV F1"], width=bar_width, label='F1-CV', color='orange')

plt.xlabel('Modèle')
plt.ylabel('F1-Score')
plt.title('Comparaison des modèles')
plt.xticks([i + bar_width / 2 for i in index], results_df["Modèle"])
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

import joblib
joblib.dump(model_rf, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
data.to_csv("heart_clean.csv", index=False)

importances = model_rf.feature_importances_

# Calcul des importances
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Top 5 avec pourcentage
total_imp = feat_imp["Importance"].sum()
feat_imp["Importance (%)"] = (feat_imp["Importance"] / total_imp * 100).round(2)

top_feat_imp = feat_imp.head(11)
joblib.dump(top_feat_imp, "top_features.pkl")

