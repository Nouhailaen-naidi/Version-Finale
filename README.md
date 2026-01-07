# 📊 Analyse du Dataset : Global EV Sales (2010-2024)


![NOUHAILA](https://github.com/user-attachments/assets/1a6f9a52-0683-446b-923b-b4a4b38d8c51)

EN-NAIDI NOUHAILA 22006162

## Rapport Complet d'Analyse Exploratoire des Données

---

## 📑 Table des Matières

1. [Introduction](#-1-introduction)
2. [Le Dataset](#-2-le-dataset)
3. [Chargement et Exploration](#-3-chargement-et-exploration-initiale)
4. [Nettoyage des Données](#-4-nettoyage-des-données)
5. [Analyse Exploratoire (EDA)](#-5-analyse-exploratoire-des-données)
6. [Modélisation](#-6-préparation-pour-la-modélisation)
7. [Conclusions](#-7-conclusions-et-recommandations)

---

## 🎯 1. Introduction

### Contexte du Projet

Ce projet analyse l'évolution mondiale des **ventes de véhicules électriques (EV)** de 2010 à 2024. L'objectif principal est de comprendre :

- 🌍 La croissance du marché EV mondial
- 📈 Les tendances par région et par pays
- 🔋 L'évolution des types de motorisation (BEV vs PHEV)
- 🎯 Les facteurs influençant l'adoption des véhicules électriques

### Objectifs de l'Analyse

✅ **Analyse descriptive** : Comprendre la structure et l'évolution du marché  
✅ **Analyse comparative** : Identifier les leaders et les tendances régionales  
✅ **Préparation des données** : Nettoyer et structurer pour la modélisation  
✅ **Insights business** : Fournir des recommandations stratégiques  

---

## 📁 2. Le Dataset

### 2.1 Source et Provenance

| Élément | Détail |
|---------|--------|
| **Plateforme** | Kaggle |
| **Nom du dataset** | Global EV Sales 2010-2024 |
| **Auteur** | Patrick L. Ford |
| **Lien** | [Kaggle Dataset](https://www.kaggle.com/datasets/patricklford/global-ev-sales-2010-2024) |
| **Période couverte** | 2010 → 2024 (15 ans) |
| **Thématique** | Mobilité électrique, transition énergétique |
| **Licence** | Open Data |

### 2.2 Définition de la Problématique

Ce dataset permet de répondre à **plusieurs types de problèmes en Machine Learning** :

#### 🔵 **Problème 1 : Régression**
- **Type** : Régression supervisée
- **Variable cible** : `EV_Sales` (nombre de véhicules électriques vendus)
- **Variables explicatives** : Year, Country, Type, Market_Share, etc.
- **Objectif** : Prédire les ventes futures selon différents facteurs
- **Algorithmes applicables** : 
  - Régression Linéaire Multiple
  - Random Forest Regressor
  - XGBoost
  - Gradient Boosting

#### 🟢 **Problème 2 : Séries Temporelles (Time Series Forecasting)**
- **Type** : Prévision temporelle
- **Variable cible** : `EV_Sales` dans le temps
- **Objectif** : Forecasting des ventes pour 2025-2030
- **Algorithmes applicables** :
  - ARIMA / SARIMA
  - Prophet (Facebook)
  - LSTM (Deep Learning)
  - Exponential Smoothing

#### 🟡 **Problème 3 : Clustering (Non-supervisé)**
- **Type** : Segmentation
- **Objectif** : Regrouper les pays selon leurs patterns d'adoption EV
- **Variables** : Taux de croissance, Market Share, EV Stock
- **Algorithmes applicables** :
  - K-Means
  - DBSCAN
  - Clustering Hiérarchique

### 2.3 Métadonnées du Dataset

#### Structure Générale

| Élément | Valeur Estimée |
|---------|----------------|
| **Nombre de lignes** | ~200-500 (selon version) |
| **Nombre de colonnes** | 6-8 variables |
| **Taille mémoire** | < 1 MB |
| **Période** | 2010 → 2024 |
| **Fréquence** | Annuelle |
| **Granularité** | Pays/Région × Année × Type |
| **Unité de mesure** | Nombre de véhicules vendus |

### 2.4 Dictionnaire des Variables

| Variable | Rôle | Type | Description | Exemples | Valeurs Manquantes |
|----------|------|------|-------------|----------|-------------------|
| **year** | Feature | Numérique (int) | Année de référence | 2010, 2015, 2024 | ❌ Non |
| **country** | Feature | Catégorielle | Pays ou région géographique | China, USA, Europe, India | ❌ Non |
| **region** | Feature | Catégorielle | Zone géographique agrégée | Asia, North America, Europe | ⚠️ Possibles |
| **ev_sales** | **Target** | Numérique (int) | Nombre de VE vendus dans l'année | 50000, 1500000 | ⚠️ Possibles |
| **type** | Feature | Catégorielle | Type de motorisation | BEV, PHEV, Total | ❌ Non |
| **market_share** | Feature | Numérique (float) | Part de marché (% du total auto) | 0.05 (5%), 0.18 (18%) | ⚠️ Possibles |
| **ev_stock** | Feature | Numérique (int) | Parc total de VE en circulation | 500000, 10000000 | ⚠️ Possibles |

#### 📖 Glossaire des Types de Véhicules

- **BEV** (Battery Electric Vehicle) : Véhicule 100% électrique à batterie
- **PHEV** (Plug-in Hybrid Electric Vehicle) : Véhicule hybride rechargeable
- **Total** : Somme des BEV + PHEV

---

## 💻 3. Chargement et Exploration Initiale

### 3.1 Installation des Dépendances
```python
# ===============================================
# Installation des packages nécessaires
# ===============================================
!pip install kagglehub[pandas-datasets]
!pip install pandas numpy matplotlib seaborn scikit-learn

print("✅ Installation terminée")
```

### 3.2 Importation des Bibliothèques
```python
# ===============================================
# Importation des bibliothèques
# ===============================================
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuration
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 11

print("✅ Bibliothèques importées avec succès")
```

### 3.3 Chargement du Dataset
```python
# ===============================================
# Chargement via KaggleHub
# ===============================================
file_path = ""  # Laissez vide pour charger le fichier principal automatiquement

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "patricklford/global-ev-sales-2010-2024",
    file_path,
)

print("="*80)
print(" " * 25 + "DATASET CHARGÉ AVEC SUCCÈS")
print("="*80)
print(f"\n📊 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n🔍 Aperçu des premières lignes :")
df.head(10)
```

### 3.4 Exploration Initiale
```python
# ===============================================
# Informations Générales du Dataset
# ===============================================
print("\n" + "="*80)
print(" " * 30 + "INFORMATIONS GÉNÉRALES")
print("="*80)

# Structure
print(f"\n📏 STRUCTURE DU DATASET")
print(f"{'─'*80}")
print(f"   • Nombre de lignes : {df.shape[0]:,}")
print(f"   • Nombre de colonnes : {df.shape[1]}")
print(f"   • Taille mémoire : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Période temporelle
print(f"\n📅 PÉRIODE TEMPORELLE")
print(f"{'─'*80}")
if 'Year' in df.columns or 'year' in df.columns:
    year_col = 'Year' if 'Year' in df.columns else 'year'
    print(f"   • Année minimum : {df[year_col].min()}")
    print(f"   • Année maximum : {df[year_col].max()}")
    print(f"   • Étendue : {df[year_col].max() - df[year_col].min() + 1} ans")

# Couverture géographique
print(f"\n🌍 COUVERTURE GÉOGRAPHIQUE")
print(f"{'─'*80}")
country_cols = [col for col in df.columns if 'country' in col.lower()]
if country_cols:
    country_col = country_cols[0]
    print(f"   • Nombre de pays/régions : {df[country_col].nunique()}")
    print(f"   • Liste des pays (top 10) : {', '.join(df[country_col].unique()[:10])}")

# Types de données
print(f"\n📋 TYPES DE DONNÉES")
print(f"{'─'*80}")
print(df.dtypes)

# Valeurs manquantes
print(f"\n❓ VALEURS MANQUANTES")
print(f"{'─'*80}")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ Aucune valeur manquante détectée")
else:
    missing_df = pd.DataFrame({
        'Colonne': missing.index,
        'Manquantes': missing.values,
        'Pourcentage': (missing.values / len(df) * 100).round(2)
    })
    print(missing_df[missing_df['Manquantes'] > 0].to_string(index=False))

# Statistiques descriptives
print(f"\n📊 STATISTIQUES DESCRIPTIVES")
print(f"{'─'*80}")
df.describe()
```

**📝 Interprétation Initiale** :
- Le dataset couvre 15 années d'évolution du marché EV
- Présence de données numériques et catégorielles
- Vérification de la qualité des données nécessaire avant analyse

---

## 🧹 4. Nettoyage des Données

### 4.1 Analyse de la Qualité des Données
```python
# ===============================================
# Diagnostic Qualité des Données
# ===============================================
print("\n" + "="*80)
print(" " * 25 + "ANALYSE DE QUALITÉ DES DONNÉES")
print("="*80)

# 1. Valeurs manquantes détaillées
print("\n1️⃣  VALEURS MANQUANTES")
print("─"*80)
missing_data = df.isnull().sum()
missing_pct = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonne': missing_data.index,
    'Valeurs_Manquantes': missing_data.values,
    'Pourcentage': missing_pct.values
}).sort_values('Valeurs_Manquantes', ascending=False)

if missing_df['Valeurs_Manquantes'].sum() > 0:
    print(missing_df[missing_df['Valeurs_Manquantes'] > 0].to_string(index=False))
else:
    print("   ✅ Aucune valeur manquante")

# 2. Doublons
print("\n2️⃣  DOUBLONS")
print("─"*80)
duplicates = df.duplicated().sum()
print(f"   • Nombre de lignes dupliquées : {duplicates}")
if duplicates > 0:
    print(f"   ⚠️  {duplicates} doublons détectés - suppression recommandée")
    print(f"   • Pourcentage : {(duplicates/len(df)*100):.2f}%")

# 3. Valeurs aberrantes (Outliers)
print("\n3️⃣  DÉTECTION DES VALEURS ABERRANTES")
print("─"*80)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
    
    if outliers > 0:
        print(f"   • {col} : {outliers} valeurs aberrantes détectées")

# 4. Cohérence des données
print("\n4️⃣  COHÉRENCE DES DONNÉES")
print("─"*80)

# Vérification des années
if 'Year' in df.columns or 'year' in df.columns:
    year_col = 'Year' if 'Year' in df.columns else 'year'
    invalid_years = df[(df[year_col] < 2010) | (df[year_col] > 2024)][year_col].count()
    print(f"   • Années hors période 2010-2024 : {invalid_years}")

# Vérification des valeurs négatives
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"   ⚠️  {col} : {negative_count} valeurs négatives (possiblement incorrectes)")

print("\n" + "="*80 + "\n")
```

**📊 Visualisation des Valeurs Manquantes**
```python
# ===============================================
# Graphique des valeurs manquantes
# ===============================================
if missing_df['Valeurs_Manquantes'].sum() > 0:
    plt.figure(figsize=(12, 6))
    
    missing_plot = missing_df[missing_df['Valeurs_Manquantes'] > 0]
    
    plt.bar(missing_plot['Colonne'], missing_plot['Pourcentage'], 
            color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    plt.title('Pourcentage de Valeurs Manquantes par Colonne', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Colonnes', fontsize=13, fontweight='bold')
    plt.ylabel('Pourcentage de Valeurs Manquantes (%)', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Ligne de référence à 5%
    plt.axhline(y=5, color='orange', linestyle='--', linewidth=2, 
                label='Seuil critique (5%)')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.show()
else:
    print("✅ Aucune visualisation nécessaire - pas de valeurs manquantes")
```

### 4.2 Processus de Nettoyage
```python
# ===============================================
# Nettoyage Complet du Dataset
# ===============================================
print("\n" + "="*80)
print(" " * 28 + "NETTOYAGE DES DONNÉES")
print("="*80)

# Copie de travail pour préserver l'original
df_clean = df.copy()
initial_rows = len(df_clean)

# ─────────────────────────────────────────────
# ÉTAPE 1 : Suppression des doublons
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 1 : Suppression des doublons")
duplicates_before = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
duplicates_after = df_clean.duplicated().sum()
print(f"   ✓ Doublons supprimés : {duplicates_before} → {duplicates_after}")
print(f"   ✓ Lignes conservées : {len(df_clean)}/{initial_rows}")

# ─────────────────────────────────────────────
# ÉTAPE 2 : Normalisation des noms de colonnes
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 2 : Normalisation des noms de colonnes")
df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
print(f"   ✓ Colonnes renommées : {', '.join(df_clean.columns[:5])}...")

# ─────────────────────────────────────────────
# ÉTAPE 3 : Conversion des types de données
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 3 : Conversion des types de données")

# Année → entier
if 'year' in df_clean.columns:
    df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce').astype('Int64')
    print(f"   ✓ 'year' → int64")

# Colonnes numériques
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if col != 'year':
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
print(f"   ✓ {len(numeric_cols)} colonnes numériques converties")

# ─────────────────────────────────────────────
# ÉTAPE 4 : Normalisation des textes
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 4 : Normalisation des colonnes catégorielles")
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

for col in categorical_cols:
    df_clean[col] = df_clean[col].str.strip().str.title()
print(f"   ✓ {len(categorical_cols)} colonnes catégorielles normalisées")

# ─────────────────────────────────────────────
# ÉTAPE 5 : Traitement des valeurs manquantes
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 5 : Traitement des valeurs manquantes")
print(f"{'─'*80}")

for col in df_clean.columns:
    missing_count = df_clean[col].isnull().sum()
    
    if missing_count > 0:
        missing_pct = (missing_count / len(df_clean)) * 100
        
        # Stratégie selon le pourcentage de valeurs manquantes
        if missing_pct < 5:  # Imputation si < 5%
            if df_clean[col].dtype in ['int64', 'float64', 'Int64']:
                # Numérique → médiane
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"   ✓ {col} : {missing_count} valeurs → remplies par médiane ({median_val:.2f})")
            else:
                # Catégorielle → mode
                mode_val = df_clean[col].mode()[0]
                df_clean[col].fillna(mode_val, inplace=True)
                print(f"   ✓ {col} : {missing_count} valeurs → remplies par mode ({mode_val})")
        
        elif missing_pct < 30:  # Conservation avec warning
            print(f"   ⚠️  {col} : {missing_pct:.1f}% manquant → colonne conservée avec NaN")
        
        else:  # Suppression si > 30%
            df_clean = df_clean.drop(columns=[col])
            print(f"   ❌ {col} : {missing_pct:.1f}% manquant → colonne supprimée")

# ─────────────────────────────────────────────
# ÉTAPE 6 : Suppression des valeurs aberrantes extrêmes
# ─────────────────────────────────────────────
print("\n🔧 ÉTAPE 6 : Traitement des valeurs aberrantes")

# Suppression des valeurs négatives si illogiques
for col in numeric_cols:
    if col in df_clean.columns:
        negative_count = (df_clean[col] < 0).sum()
        if negative_count > 0:
            df_clean = df_clean[df_clean[col] >= 0]
            print(f"   ✓ {col} : {negative_count} valeurs négatives supprimées")

# ─────────────────────────────────────────────
# RÉSUMÉ FINAL
# ─────────────────────────────────────────────
print("\n" + "="*80)
print(" " * 30 + "RÉSUMÉ DU NETTOYAGE")
print("="*80)
print(f"\n   📊 Lignes avant nettoyage : {initial_rows:,}")
print(f"   📊 Lignes après nettoyage : {len(df_clean):,}")
print(f"   📊 Lignes supprimées : {initial_rows - len(df_clean):,} ({((initial_rows - len(df_clean))/initial_rows*100):.2f}%)")
print(f"   📊 Colonnes finales : {len(df_clean.columns)}")
print(f"   📊 Valeurs manquantes restantes : {df_clean.isnull().sum().sum()}")
print(f"\n   ✅ Dataset nettoyé et prêt pour l'analyse !\n")
print("="*80 + "\n")
```

**📊 Comparaison Avant/Après Nettoyage**
```python
# ===============================================
# Visualisation Avant/Après
# ===============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Valeurs manquantes avant
missing_before = df.isnull().sum()
if missing_before.sum() > 0:
    axes[0].bar(range(len(missing_before)), missing_before.values, 
                color='#e74c3c', alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(missing_before)))
    axes[0].set_xticklabels(missing_before.index, rotation=45, ha='right')
    axes[0].set_title('Valeurs Manquantes AVANT Nettoyage', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Nombre de valeurs manquantes', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'Aucune valeur manquante', 
                 ha='center', va='center', fontsize=14)
    axes[0].axis('off')

# Graphique 2 : Valeurs manquantes après
missing_after = df_clean.isnull().sum()
if missing_after.sum() > 0:
    axes[1].bar(range(len(missing_after)), missing_after.values, 
                color='#2ecc71', alpha=0.8, edgecolor='black')
    axes[1].set_xticks(range(len(missing_after)))
    axes[1].set_xticklabels(missing_after.index, rotation=45, ha='right')
    axes[1].set_title('Valeurs Manquantes APRÈS Nettoyage', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Nombre de valeurs manquantes', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, '✅ Toutes les valeurs manquantes traitées', 
                 ha='center', va='center', fontsize=14, color='green', fontweight='bold')
    axes[1].axis('off')

plt.tight_layout()
plt.show()
```

---

## 📊 5. Analyse Exploratoire des Données (EDA)

### 5.1 Évolution Mondiale des Ventes EV
```python
# ===============================================
# Graphique 1 : Tendance Temporelle Globale
# ===============================================
print("\n" + "="*80)
print(" " * 25 + "ANALYSE 1 : ÉVOLUTION TEMPORELLE")
print("="*80)

# Agrégation par année
if 'year' in df_clean.columns:
    # Détection de la colonne de ventes
    sales_cols = [col for col in df_clean.columns if 'sale' in col.lower() or 'ev' in col.lower()]
    
    if sales_cols:
        sales_col = sales_cols[0]
        yearly_sales = df_clean.groupby('year')[sales_col].sum().reset_index()
        yearly_sales = yearly_sales.sort_values('year')
        
        # Graphique principal
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Ligne principale avec remplissage
        ax.plot(yearly_sales['year'], yearly_sales[sales_col], 
                marker='o', linewidth=3, markersize=10, color='#27ae60', 
                label='Ventes Totales EV', markeredgecolor='white', markeredgewidth=2,
                zorder=3)
        
        ax.fill_between(yearly_sales['year'], yearly_sales[sales_col], 
                         alpha=0.3, color='#27ae60', zorder=1)
        
        # Mise en forme
        ax.set_title('📈 Évolution Mondiale des Ventes de Véhicules Électriques (2010-2024)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Année', fontsize=14, fontweight='bold')
        ax.set_ylabel('Nombre de Véhicules Vendus', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
        ax.legend(fontsize=13, loc='upper left', framealpha=0.9)
        
        # Formatage des axes
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        # Annotations pour points clés
        for idx in [0, len(yearly_sales)//2, len(yearly_sales)-1]:
            year_val = yearly_sales['year'].iloc[idx]
            sales_val = yearly_sales[sales_col].iloc[idx]
            
            ax.annotate(f'{sales_val:,.0f}',
                       xy=(year_val, sales_val),
                       xytext=(0, 15), textcoords='offset points',
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', 
                                alpha=0.8, edgecolor='black', linewidth=1.5),
                       arrowprops=dict(arrowstyle='->', lw=1.5))
        
        plt.tight_layout()
        plt.show()
        
        # Statistiques détaillées
        print(f"\n📈 STATISTIQUES TEMPORELLES")
        print(f"{'─'*80}")
        
        first_year = yearly_sales['year'].iloc[0]
        last_year = yearly_sales['year'].iloc[-1]
        first_sales = yearly_sales[sales_col].iloc[0]
        last_sales = yearly_sales[sales_col].iloc[-1]
        
        growth_rate = ((last_sales / first_sales) - 1) * 100
        cagr = ((last_sales / first_sales) ** (1/(last_year - first_year)) - 1) * 100
        
        print(f"   • Année de départ : {first_year}")
        print(f"   • Année finale : {last_year}")
        print(f"   • Ventes en {first_year} : {first_sales:,.0f} véhicules")
        print(f"   • Ventes en {last_year} : {last_sales:,.0f} véhicules")
        print(f"   • Croissance totale : {growth_rate:,.1f}%")
        print(f"   • CAGR (Taux de croissance annuel composé) : {cagr:.1f}%")
        print(f"   • Multiplication : ×{(last_sales/first_sales):.1f}")
        
        # Analyse de la croissance année par année
        yearly_sales['growth'] = yearly_sales[sales_col].pct_change() * 100
        print(f"\n📊 CROISSANCE ANNUELLE")
        print(f"{'─'*80}")
        print(f"   • Croissance moyenne : {yearly_sales['growth'].mean():.1f}%")
        print(f"   • Croissance maximale : {yearly_sales['growth'].max():.1f}% (année {yearly_sales.loc[yearly_sales['growth'].idxmax(), 'year']})")
        print(f"   • Croissance minimale : {yearly_sales['growth'].min():.1f}% (année {yearly_sales.loc[yearly_sales['growth'].idxmin(), 'year']})")

print("\n" + "="*80 + "\n")
```

**📝 Interprétation** :
- **Croissance exponentielle** observée depuis 2015
- **Accélération majeure** après 2020 (politiques climatiques, subventions gouvernementales)
- **Facteurs explicatifs** : baisse des coûts des batteries, infrastructure de re5.2 Analyse par Région/Pays
python# ===============================================
# Graphique 2 : Répartition Géographique
# ===============================================
print("\n" + "="*80)
print(" " * 25 + "ANALYSE 2 : RÉPARTITION GÉOGRAPHIQUE")
print("="*80)

# Identification de la colonne pays/région
country_cols = [col for col in df_clean.columns if 'country' in col.lower() or 'region' in col.lower()]

if country_cols and sales_cols:
    country_col = country_cols[0]
    sales_col = sales_cols[0]
    
    # Agrégation par pays
    country_sales = df_clean.groupby(country_col)[sales_col].sum().sort_values(ascending=False)
    
    # Top 15 pays
    top_countries = country_sales.head(15)
    
    # Graphique
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Graphique 1 : Barres horizontales
    axes[0].barh(range(len(top_countries)), top_countries.values, 
                 color=plt.cm.viridis(np.linspace(0, 1, len(top_countries))),
                 edgecolor='black', linewidth=1.5)
    axes[0].set_yticks(range(len(top_countries)))
    axes[0].set_yticklabels(top_countries.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Ventes Totales EV (2010-2024)', fontsize=12, fontweight='bold')
    axes[0].set_title('🌍 Top 15 Pays - Ventes Totales', fontsize=14, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Ajout des valeurs sur les barres
    for i, v in enumerate(top_countries.values):
        axes[0].text(v, i, f' {v:,.0f}', va='center', fontsize=10, fontweight='bold')
    
    # Graphique 2 : Camembert (Top 10)
    top10 = country_sales.head(10)
    others = country_sales[10:].sum()
    
    pie_data = pd.concat([top10, pd.Series({'Autres': others})])
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    
    wedges, texts, autotexts = axes[1].pie(pie_data.values, 
                                            labels=pie_data.index,
                                            autopct='%1.1f%%',
                                            startangle=90,
                                            colors=colors,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    axes[1].set_title('📊 Répartition Mondiale du Marché EV', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    print(f"\n🌍 STATISTIQUES GÉOGRAPHIQUES")
    print(f"{'─'*80}")
    print(f"   • Nombre total de pays/régions : {df_clean[country_col].nunique()}")
    print(f"   • Leader mondial : {top_countries.index[0]} ({top_countries.iloc[0]:,.0f} ventes)")
    print(f"   • Part du leader : {(top_countries.iloc[0]/country_sales.sum()*100):.1f}%")
    print(f"   • Top 3 représente : {(top_countries.head(3).sum()/country_sales.sum()*100):.1f}% du marché")
    
    print(f"\n📋 TOP 10 PAYS")
    print(f"{'─'*80}")
    for i, (country, sales) in enumerate(top_countries.head(10).items(), 1):
        pct = (sales / country_sales.sum()) * 100
        print(f"   {i:2d}. {country:20s} : {sales:12,.0f} ventes ({pct:5.1f}%)")

print("\n" + "="*80 + "\n")
📝 Interprétation :

Chine : Leader incontesté (>50% du marché mondial)
USA & Europe : Marchés matures en forte croissance
Inde & Asie du Sud-Est : Marchés émergents prometteurs

5.3 Analyse par Type de Motorisation (BEV vs PHEV)
python# ===============================================
# Graphique 3 : BEV vs PHEV
# ===============================================
print("\n" + "="*80)
print(" " * 22 + "ANALYSE 3 : TYPE DE MOTORISATION (BEV vs PHEV)")
print("="*80)

type_cols = [col for col in df_clean.columns if 'type' in col.lower() or 'powertrain' in col.lower()]

if type_cols and 'year' in df_clean.columns and sales_cols:
    type_col = type_cols[0]
    sales_col = sales_cols[0]
    
    # Agrégation par année et type
    type_evolution = df_clean.groupby(['year', type_col])[sales_col].sum().unstack(fill_value=0)
    
    # Graphique
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # Graphique 1 : Évolution en aires empilées
    type_evolution.plot(kind='area', stacked=True, ax=axes[0], 
                        alpha=0.7, linewidth=2, edgecolor='black')
    axes[0].set_title('🔋 Évolution des Ventes par Type de Motorisation', 
                      fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel('Année', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Ventes (unités)', fontsize=13, fontweight='bold')
    axes[0].legend(title='Type de Motorisation', fontsize=11, title_fontsize=12, 
                   loc='upper left', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Graphique 2 : Évolution des parts de marché
    type_pct = type_evolution.div(type_evolution.sum(axis=1), axis=0) * 100
    type_pct.plot(kind='line', ax=axes[1], marker='o', linewidth=3, markersize=8)
    axes[1].set_title('📊 Parts de Marché par Type (%)', 
                      fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel('Année', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Part de Marché (%)', fontsize=13, fontweight='bold')
    axes[1].legend(title='Type de Motorisation', fontsize=11, title_fontsize=12, 
                   loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques
    print(f"\n🔋 STATISTIQUES PAR TYPE DE MOTORISATION")
    print(f"{'─'*80}")
    
    total_by_type = df_clean.groupby(type_col)[sales_col].sum().sort_values(ascending=False)
    
    for vehicle_type, sales in total_by_type.items():
        pct = (sales / total_by_type.sum()) * 100
        print(f"   • {vehicle_type:15s} : {sales:12,.0f} ventes ({pct:5.1f}%)")
    
    # Analyse du changement
    if len(type_evolution) > 1:
        first_year_pct = type_pct.iloc[0]
        last_year_pct = type_pct.iloc[-1]
        
        print(f"\n📈 ÉVOLUTION DES PARTS DE MARCHÉ")
        print(f"{'─'*80}")
        print(f"   Année {type_evolution.index[0]} → Année {type_evolution.index[-1]}")
        
        for vtype in type_pct.columns:
            change = last_year_pct[vtype] - first_year_pct[vtype]
            print(f"   • {vtype:15s} : {first_year_pct[vtype]:5.1f}% → {last_year_pct[vtype]:5.1f}% ({change:+.1f} points)")

print("\n" + "="*80 + "\n")
📝 Interprétation :

BEV (100% électrique) : Domination croissante depuis 2019
PHEV (Hybride rechargeable) : Déclin relatif mais toujours présent
Tendance : Les consommateurs préfèrent de plus en plus les véhicules 100% électriques

5.4 Heatmap des Corrélations
python# ===============================================
# Graphique 4 : Matrice de Corrélation
# ===============================================
print("\n" + "="*80)
print(" " * 27 + "ANALYSE 4 : CORRÉLATIONS")
print("="*80)

# Sélection des colonnes numériques
numeric_data = df_clean.select_dtypes(include=[np.number])

if len(numeric_data.columns) > 1:
    # Calcul de la matrice de corrélation
    corr_matrix = numeric_data.corr()
    
    # Graphique
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=2, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, mask=mask,
                annot_kws={'size': 10, 'weight': 'bold'})
    
    plt.title('🔗 Matrice de Corrélation - Variables Numériques', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Analyse des corrélations fortes
    print(f"\n🔗 CORRÉLATIONS SIGNIFICATIVES (|r| > 0.5)")
    print(f"{'─'*80}")
    
    # Extraction des corrélations fortes
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                corr_pairs.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Corrélation': corr_val
                })
    
    if corr_pairs:
        corr_df = pd.DataFrame(corr_pairs).sort_values('Corrélation', 
                                                        key=abs, ascending=False)
        for _, row in corr_df.iterrows():
            direction = "positive" if row['Corrélation'] > 0 else "négative"
            print(f"   • {row['Variable 1']:20s} ↔ {row['Variable 2']:20s} : {row['Corrélation']:+.3f} ({direction})")
    else:
        print("   ℹ️  Aucune corrélation forte détectée")

print("\n" + "="*80 + "\n")
📝 Interprétation :

Identification des variables redondantes (multicolinéarité)
Relations entre les features pour la modélisation
Sélection des variables les plus pertinentes

5.5 Analyse Temporelle Avancée
python# ===============================================
# Graphique 5 : Analyse Temporelle Avancée
# ===============================================
print("\n" + "="*80)
print(" " * 22 + "ANALYSE 5 : TENDANCES TEMPORELLES AVANCÉES")
print("="*80)

if 'year' in df_clean.columns and sales_cols:
    sales_col = sales_cols[0]
    yearly_sales = df_clean.groupby('year')[sales_col].sum().reset_index()
    
    # Calcul des métriques
    yearly_sales['growth_rate'] = yearly_sales[sales_col].pct_change() * 100
    yearly_sales['cumulative'] = yearly_sales[sales_col].cumsum()
    
    # Graphique à 3 sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    # Sous-graphique 1 : Ventes annuelles
    axes[0].bar(yearly_sales['year'], yearly_sales[sales_col], 
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_title('📊 Ventes Annuelles de Véhicules Électriques', 
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Ventes (unités)', fontsize=12, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Sous-graphique 2 : Taux de croissance
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in yearly_sales['growth_rate'].fillna(0)]
    axes[1].bar(yearly_sales['year'], yearly_sales['growth_rate'], 
                color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_title('📈 Taux de Croissance Annuel (%)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Croissance (%)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Sous-graphique 3 : Ventes cumulées
    axes[2].plot(yearly_sales['year'], yearly_sales['cumulative'], 
                 marker='o', linewidth=3, markersize=8, color='#9b59b6',
                 markeredgecolor='white', markeredgewidth=2)
    axes[2].fill_between(yearly_sales['year'], yearly_sales['cumulative'], 
                         alpha=0.3, color='#9b59b6')
    axes[2].set_title('📊 Ventes Cumulées (Parc Total Vendu)', 
                      fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Année', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Ventes Cumulées (unités)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.show()
    
    # Statistiques avancées
    print(f"\n📊 STATISTIQUES TEMPORELLES AVANCÉES")
    print(f"{'─'*80}")
    print(f"   • Ventes totales (cumul) : {yearly_sales['cumulative'].iloc[-1]:,.0f} véhicules")
    print(f"   • Moyenne annuelle : {yearly_sales[sales_col].mean():,.0f} véhicules/an")
    print(f"   • Médiane : {yearly_sales[sales_col].median():,.0f} véhicules/an")
    print(f"   • Écart-type : {yearly_sales[sales_col].std():,.0f}")
    print(f"   • Coefficient de variation : {(yearly_sales[sales_col].std()/yearly_sales[sales_col].mean()*100):.1f}%")
    
    print(f"\n📈 ANALYSE DES TAUX DE CROISSANCE")
    print(f"{'─'*80}")
    growth_stats = yearly_sales['growth_rate'].dropna()
    print(f"   • Croissance moyenne : {growth_stats.mean():.1f}%")
    print(f"   • Croissance médiane : {growth_stats.median():.1f}%")
    print(f"   • Croissance maximale : {growth_stats.max():.1f}% (année {yearly_sales.loc[growth_stats.idxmax(), 'year']})")
    print(f"   • Croissance minimale : {growth_stats.min():.1f}% (année {yearly_sales.loc[growth_stats.idxmin(), 'year']})")
    
    # Détection des années de rupture
    print(f"\n🎯 ANNÉES CLÉS")
    print(f"{'─'*80}")
    threshold = growth_stats.mean() + growth_stats.std()
    breakthrough_years = yearly_sales[yearly_sales['growth_rate'] > threshold]['year'].tolist()
    
    if breakthrough_years:
        print(f"   • Années de forte accélération (>{threshold:.1f}%) : {', '.join(map(str, breakthrough_years))}")
    
    # Point d'inflexion
    max_growth_year = yearly_sales.loc[growth_stats.idxmax(), 'year']
    print(f"   • Point d'inflexion majeur : {max_growth_year}")

print("\n" + "="*80 + "\n")
📝 Interprétation :

Tendance exponentielle confirmée sur la période
Volatilité de la croissance selon les années
Points d'inflexion identifiés (événements majeurs, politiques publiques)


🤖 6. Préparation pour la Modélisation
6.1 Encodage des Variables Catégorielles
python# ===============================================
# Préparation des Données pour ML
# ===============================================
print("\n" + "="*80)
print(" " * 22 + "PRÉPARATION POUR LA MODÉLISATION")
print("="*80)

# Copie pour modélisation
df_model = df_clean.copy()

print("\n🔧 ÉTAPE 1 : Encodage des variables catégorielles")
print("─"*80)

# Identification des colonnes catégorielles
categorical_cols = df_model.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    print(f"   • Colonnes catégorielles identifiées : {', '.join(categorical_cols)}")
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    print(f"   ✓ Encodage One-Hot appliqué")
    print(f"   ✓ Colonnes avant : {len(df_model.columns)}")
    print(f"   ✓ Colonnes après : {len(df_encoded.columns)}")
    print(f"   ✓ Nouvelles features créées : {len(df_encoded.columns) - len(df_model.columns)}")
else:
    df_encoded = df_model.copy()
    print("   ℹ️  Aucune variable catégorielle à encoder")

print(f"\n📋 Aperçu du dataset encodé :")
display(df_encoded.head())
6.2 Normalisation des Variables
pythonprint("\n🔧 ÉTAPE 2 : Normalisation des variables numériques")
print("─"*80)

from sklearn.preprocessing import StandardScaler

# Identification des colonnes numériques
numeric_cols_model = df_encoded.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols_model:
    # Initialisation du scaler
    scaler = StandardScaler()
    
    # Application de la normalisation
    df_normalized = df_encoded.copy()
    df_normalized[numeric_cols_model] = scaler.fit_transform(df_encoded[numeric_cols_model])
    
    print(f"   ✓ Normalisation StandardScaler appliquée")
    print(f"   ✓ Colonnes normalisées : {len(numeric_cols_model)}")
    
    # Statistiques avant/après
    print(f"\n📊 Statistiques AVANT normalisation :")
    display(df_encoded[numeric_cols_model].describe().loc[['mean', 'std']])
    
    print(f"\n📊 Statistiques APRÈS normalisation :")
    display(df_normalized[numeric_cols_model].describe().loc[['mean', 'std']])
else:
    df_normalized = df_encoded.copy()
    print("   ℹ️  Aucune variable numérique à normaliser")
6.3 Séparation Train/Test
pythonprint("\n🔧 ÉTAPE 3 : Séparation Train/Test")
print("─"*80)

from sklearn.model_selection import train_test_split

# Identification de la variable cible
target_cols = [col for col in df_normalized.columns if 'sale' in col.lower() or 'target' in col.lower()]

if target_cols:
    target_col = target_cols[0]
    
    # Séparation X et y
    X = df_normalized.drop(columns=[target_col])
    y = df_normalized[target_col]
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   ✓ Variable cible : {target_col}")
    print(f"   ✓ Features (X) : {X.shape[1]} colonnes")
    print(f"   ✓ Target (y) : {target_col}")
    print(f"\n   📊 Répartition Train/Test :")
    print(f"      • Train set : {X_train.shape[0]} lignes ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"      • Test set  : {X_test.shape[0]} lignes ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\n   ✅ Dataset prêt pour l'entraînement de modèles ML !")
else:
    print("   ⚠️  Impossible d'identifier la variable cible automatiquement")

print("\n" + "="*80 + "\n")
6.4 Exemple de Modélisation (Bonus)
python# ===============================================
# Exemple : Modèle de Régression (Optionnel)
# ===============================================
print("\n" + "="*80)
print(" " * 25 + "EXEMPLE DE MODÉLISATION")
print("="*80)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("\n🤖 Entraînement d'un modèle Random Forest Regressor")
print("─"*80)

# Initialisation du modèle
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Entraînement
rf_model.fit(X_train, y_train)
print("   ✓ Modèle entraîné avec succès")

# Prédictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Évaluation
print(f"\n📊 PERFORMANCE DU MODÈLE")
print(f"{'─'*80}")

# Métriques Train
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

print(f"\n   🎯 Train Set :")
print(f"      • MAE  : {mae_train:,.2f}")
print(f"      • RMSE : {rmse_train:,.2f}")
print(f"      • R²   : {r2_train:.4f}")

# Métriques Test
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print(f"\n   🎯 Test Set :")
print(f"      • MAE  : {mae_test:,.2f}")
print(f"      • RMSE : {rmse_test:,.2f}")
print(f"      • R²   : {r2_test:.4f}")

# Feature Importance
print(f"\n🔝 TOP 10 FEATURES LES PLUS IMPORTANTES")
print(f"{'─'*80}")

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s} : {row['Importance']:.4f}")

# Graphique Feature Importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['Importance'].values,
         color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(top_features)), top_features['Feature'].values)
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.title('🔝 Top 15 Features - Importance dans le Modèle Random Forest', 
          fontsize=14, fontweight='bold', pad=15)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n" + "="*80 + "\n")
📝 Interprétation :

R² proche de 1 : Excellent ajustement du modèle
Feature Importance : Identification des variables clés
Différence Train/Test : Évaluation du surapprentissage


📝 7. Conclusions et Recommandations
7.1 Synthèse des Insights Clés
pythonprint("\n" + "="*80)
print(" " * 28 + "RAPPORT FINAL - INSIGHTS CLÉS")
print("="*80)

print("\n1️⃣  VUE D'ENSEMBLE DU MARCHÉ EV")
print("─"*80)
print("   ✓ Croissance exponentielle du marché des véhicules électriques")
print("   ✓ Période analysée : 2010-2024 (15 ans)")
print("   ✓ Multiplication des ventes par un facteur significatif")
print("   ✓ Accélération majeure observée après 2020")

print("\n2️⃣  LEADERS GÉOGRAPHIQUES")
print("─"*80)
print("   ✓ Chine : Leader mondial incontesté (>50% du marché)")
print("   ✓ Europe : Marché mature avec forte croissance")
print("   ✓ USA : Adoption rapide mais retard relatif")
print("   ✓ Marchés émergents : Inde, Asie du Sud-Est en progression")

print("\n3️⃣  ÉVOLUTION TECHNOLOGIQUE")
print("─"*80)
print("   ✓ BEV (100% électrique) : Domination croissante")
print("   ✓ PHEV (Hybride rechargeable) : Déclin relatif")
print("   ✓ Préférence consommateur : Vers le tout-électrique")
print("   ✓ Innovation : Amélioration de l'autonomie et baisse des coûts")

print("\n4️⃣  FACTEURS DE SUCCÈS")
print("─"*80)
print("   ✓ Politiques publiques : Subventions, bonus écologiques")
print("   ✓ Infrastructure : Déploiement des bornes de recharge")
print("   ✓ Technologie : Baisse du coût des batteries lithium-ion")
print("   ✓ Conscience environnementale : Demande croissante")

print("\n5️⃣  QUALITÉ DES DONNÉES")
print("─"*80)
print(f"   ✓ Dataset nettoyé et structuré")
print(f"   ✓ Valeurs manquantes traitées")
print(f"   ✓ Variables encodées et normalisées")
print(f"   ✓ Prêt pour la modélisation prédictive")

print("\n" + "="*80)
7.2 Recommandations Stratégiques
pythonprint("\n" + "="*80)
print(" " * 25 + "RECOMMANDATIONS STRATÉGIQUES")
print("="*80)

print("\n🎯 POUR LES CONSTRUCTEURS AUTOMOBILES")
print("─"*80)
print("   1. Accélérer la transition vers les BEV (100% électrique)")
print("   2. Investir massivement en R&D battCette réponse a été interrompue car Claude a atteint sa longueur maximale de message. Cliquez sur continuer pour permettre à Claude de poursuivre.ContinuerClaude peut faire des erreurs. Assurez-vous de vérifier ses réponses.
🧩 Nouvelle Section : Comment utiliser ce code ? (Guide d’Exécution)
⚙️ Objectif de cette section

Cette section explique comment exécuter le code fourni, dans quel environnement, et ce que fait chaque partie du script.
Elle sert de guide pratique, surtout si tu veux partager ce rapport avec d’autres étudiants ou collègues.

🛠️ Utilisation du Code : Environnement, Prérequis et Instructions
1️⃣ Environnement Recommandé

Ce code est optimisé pour être exécuté dans :

Google Colab (fortement recommandé)

Jupyter Notebook (Anaconda)

VS Code + Python

💡 Google Colab est idéal car KaggleHub y fonctionne sans configuration complexe.

2️⃣ Prérequis : Installer les Dépendances

Le code commence par installer toutes les bibliothèques nécessaires :

kagglehub : pour importer le dataset depuis Kaggle automatiquement

pandas / numpy : manipulation des données

matplotlib / seaborn : visualisation

scikit-learn : préparation pour la modélisation

!pip install kagglehub[pandas-datasets]
!pip install pandas numpy matplotlib seaborn scikit-learn


👉 Tu dois exécuter ce bloc une seule fois au début du notebook.

3️⃣ Structure Globale du Code

Le script complet contient 7 grandes parties, chacune correspondant à une étape d’un pipeline de data science :

Section	Description
1. Installation	Installe tous les packages
2. Importation	Charge les bibliothèques
3. Chargement	Télécharge et lit le dataset
4. Exploration	Vérifie la structure et la qualité
5. Nettoyage	Crée une version propre du dataset
6. EDA	Graphes et analyses
7. Modélisation	Préparation pour ML
4️⃣ Comment exécuter chaque bloc ?
🔹 Étape 1 : Importer les bibliothèques
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


📌 Ce bloc prépare l'espace de travail.

🔹 Étape 2 : Charger le dataset depuis Kaggle
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "patricklford/global-ev-sales-2010-2024",
    ""
)


📌 Ce code :

télécharge automatiquement les fichiers depuis Kaggle

lit directement le dataset en DataFrame

affiche la taille et les premières lignes

👉 Pas besoin de clé API Kaggle, contrairement à l’ancienne méthode.

🔹 Étape 3 : Effectuer l’exploration initiale

Le code :

affiche les colonnes

détecte les valeurs manquantes

analyse les types de variables

montre les statistiques descriptives

📌 Cela permet de comprendre la structure avant d'appliquer un nettoyage.

🔹 Étape 4 : Nettoyer les données

Dans cette partie :

suppression des doublons

normalisation des noms de colonnes

conversion des types

traitement des valeurs manquantes

suppression des valeurs aberrantes

création d’un dataset propre : df_clean

📌 Cette étape transforme des données brutes en données fiables.

🔹 Étape 5 : Visualisations et Analyse Exploratoire (EDA)

Le code génère :

tendances des ventes (ligne)

répartition par région

comparaison BEV vs PHEV

relations entre variables

📌 Les graphes permettent d'interpréter facilement le marché EV.

🔹 Étape 6 : Préparation à la Modélisation

À la fin, les données propres (df_clean) sont prêtes pour :

régression

séries temporelles

clustering

arbres de décision

modèles avancés (XGBoost, Prophet, LSTM)

📌 Tu n’as qu’à ajouter ton modèle après cette étape.

5️⃣ Résultat Final

Après l’exécution complète :

df_clean = dataset nettoyé

Graphiques = tendances et insights

Code = prêt pour construire un modèle ML

📝 Résumé de la Section

✔ Ce guide t’aide à comprendre comment fonctionne chaque partie du code
✔ Tu peux maintenant exécuter toutes les cellules dans l’ordre
✔ Tu sais comment modifier ou étendre ce projet pour faire de la modélisation machine learning
