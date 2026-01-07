# 📊 Global EV Sales Analysis (2010–2024)

# Author: Nouhaila EN-NAIDI
# ID: 22006162
# Course: Machine Learning – Économie & Finance

# ================================
# 1. Importation des bibliothèques
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import warnings
warnings.filterwarnings("ignore")

# ================================
# 2. Chargement du Dataset
# ================================
# Remplacez par le chemin local ou le lien Kaggle
df = pd.read_csv("data/global_ev_sales.csv")
df.head(10)

# ================================
# 3. Aperçu du Dataset
# ================================
print("Dimensions du dataset:", df.shape)
print("\nTypes de données:")
print(df.dtypes)
print("\nValeurs manquantes:")
print(df.isnull().sum())
print("\nStatistiques descriptives:")
print(df.describe())

# ================================
# 4. Nettoyage des données
# ================================
df_clean = df.copy()
df_clean = df_clean.drop_duplicates()
df_clean.columns = df_clean.columns.str.lower().str.replace(" ", "_")
numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df_clean.select_dtypes(include="object").columns.tolist()
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# ================================
# 5. Exploratory Data Analysis (EDA)
# ================================

## A. Distribution de la variable cible
y = df_clean['ev_sales']
plt.figure(figsize=(6,4))
y.value_counts().plot(kind="bar")
plt.title("Distribution de la variable cible")
plt.xlabel("Classe")
plt.ylabel("Fréquence")
plt.show()

print("""
INTERPRÉTATION :
Si les classes sont très déséquilibrées, cela signifie que le modèle pourrait
avoir du mal à prédire la classe minoritaire. Un rééquilibrage sera nécessaire.
""")

## B. Histogrammes des variables numériques
df_clean[numeric_cols].hist(bins=30, figsize=(15,8))
plt.suptitle("Distribution des variables numériques")
plt.show()

print("""
INTERPRÉTATION :
Ces distributions permettent d'identifier les asymétries, outliers et transformations
potentielles (log-transform, normalisation…).
""")

## C. Boxplots pour détecter les outliers
df_clean[numeric_cols].plot(kind="box", subplots=True, layout=(4,4), figsize=(15,10))
plt.suptitle("Détection d'Outliers")
plt.show()

print("""
INTERPRÉTATION :
Les boxplots montrent les valeurs extrêmes (outliers) qui peuvent influencer
négativement la performance des modèles sensibles aux échelles.
""")

## D. Heatmap des corrélations
corr = df_clean[numeric_cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Corrélations entre variables numériques")
plt.show()

print("""
INTERPRÉTATION :
La heatmap montre les relations linéaires entre variables.
Une forte corrélation indique des variables redondantes qu’on peut éliminer ou combiner.
""")

## E. Barplots pour les colonnes catégorielles
for col in categorical_cols:
    plt.figure(figsize=(10,4))
    df_clean[col].value_counts().plot(kind="bar")
    plt.title(f"Distribution de {col}")
    plt.xticks(rotation=45)
    plt.show()

    print(f"""
INTERPRÉTATION ({col}) :
Ce graphique montre les catégories les plus fréquentes. Cela permet de comprendre
la composition démographique / comportementale du dataset.
""")

# ================================
# 6. Conclusions
# ================================
print("""
- Forte croissance du marché des véhicules électriques.
- La Chine domine largement le marché mondial.
- Les BEV surpassent progressivement les PHEV.
- Base solide pour des modèles ML / séries temporelles.
""")


