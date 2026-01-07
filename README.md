# 📊 Global EV Sales Analysis (2010–2024)

**Author**: Nouhaila EN-NAIDI  
**ID**: 22006162  
**Course**: Machine Learning – Économie & Finance  

---

## 📌 Project Overview

This project analyzes the global evolution of **electric vehicle (EV) sales** from **2010 to 2024**, using real-world data sourced from Kaggle.  
The goal is to extract meaningful insights on market growth, regional dynamics, and technology adoption.

---

## 🎯 Objectives

- Analyze global EV sales trends over time  
- Compare adoption rates across countries and regions  
- Study BEV vs PHEV market evolution  
- Prepare clean data for machine learning models  
- Provide business and policy insights related to energy transition  

---

## 📂 Dataset Information

- **Source**: Kaggle  
- **Dataset**: Global EV Sales 2010–2024  
- **Period**: 2010 → 2024  
- **Granularity**: Country / Year / Vehicle Type  

---

## 🛠️ Methods & Techniques

- Exploratory Data Analysis (EDA)  
- Data Cleaning & Feature Engineering  
- Time Series Analysis  
- Correlation Analysis  
- Preparation for Machine Learning (Regression / Forecasting / Clustering)  

---

## 📊 Exploratory Data Analysis (EDA)

### 1️⃣ Distribution de la variable cible
![Distribution cible](<img width="570" height="396" alt="téléchargement" src="https://github.com/user-attachments/assets/b5f7bd98-d626-4b9e-bc8b-e9207cc40d0e" />
)

**INTERPRÉTATION**:  
Si les classes sont très déséquilibrées, le modèle pourrait avoir du mal à prédire la classe minoritaire. Un rééquilibrage sera nécessaire.

---

### 2️⃣ Histogrammes des variables numériques
![Histogrammes variables numériques](images/numeric_histograms.png)
**INTERPRÉTATION**:  
Ces distributions permettent d'identifier les asymétries, outliers et transformations potentielles (log-transform, normalisation…).

---

### 3️⃣ Boxplots pour détecter les outliers
![Boxplots](images/boxplots.png)

**INTERPRÉTATION**:  
Les boxplots montrent les valeurs extrêmes (outliers) qui peuvent influencer négativement la performance des modèles sensibles aux échelles.

---

### 4️⃣ Heatmap des corrélations
![Heatmap corrélations](images/heatmap_corr.png)

**INTERPRÉTATION**:  
La heatmap montre les relations linéaires entre variables. Une forte corrélation indique des variables redondantes qu’on peut éliminer ou combiner.

---

### 5️⃣ Barplots pour les colonnes catégorielles
#### Exemple : Region
![Barplot Region](images/barplot_region.png)

**INTERPRÉTATION (region)**:  
Ce graphique montre les catégories les plus fréquentes, permettant de comprendre la composition démographique / comportementale du dataset.

#### Exemple : Type de véhicule
![Barplot Type](images/barplot_type.png)

**INTERPRÉTATION (type)**:  
Ce graphique montre la répartition BEV vs PHEV, ce qui est utile pour suivre les tendances technologiques.

---

## 📈 Key Insights

- Forte croissance du marché des véhicules électriques depuis 2015  
- La Chine domine le marché mondial des EV  
- Les véhicules BEV surpassent progressivement les PHEV  
- Accélération significative après 2020 grâce aux politiques publiques  

---

## ▶️ Project Structure


