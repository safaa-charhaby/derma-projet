# GlowGuide AI – Skincare Recommendation Web Application

## 🌟 Description générale

**GlowGuide AI** est une application web intelligente de recommandation de soins de la peau. Elle combine :

* 🧠 **Machine Learning (Python / Flask)**
* ⚛️ **Frontend interactif (React + Vite)**

L’objectif de l’application est d’aider un utilisateur à **choisir les ingrédients et produits cosmétiques les plus adaptés** à ses besoins, en fonction :

* de la zone du visage ou du corps
* du type de produit souhaité
* de ses préoccupations cutanées

Toutes les recommandations sont générées automatiquement par un **modèle de machine learning entraîné à partir d’un fichier CSV**.

---

## 🧭 Fonctionnement global de l’application

L’application fonctionne en **4 étapes principales** :

### 1️⃣ Sélection de la zone et du type de produit

L’utilisateur commence par choisir :

* une **zone** (Face, Eyes, Lips, Hair, Body, etc.)
* un **type de produit** (Serum, Cleanser, Sunscreen, Shampoo, etc.)

👉 Ces choix permettent de contextualiser les recommandations.

---

### 2️⃣ Sélection des préoccupations cutanées

L’utilisateur sélectionne ensuite une ou plusieurs **skin concerns** parmi une liste prédéfinie, par exemple :

* Acne Fighting
* Anti-Aging
* Hydrating
* Redness Reducing
* Rosacea
* Dark Spots

Chaque préoccupation est convertie en **feature binaire (0 / 1)** pour le modèle de machine learning.

---

### 3️⃣ Recommandation d’ingrédients (Machine Learning)

Lorsque l’utilisateur valide ses choix :

* Le frontend React envoie une requête `POST` au backend Flask
* Le backend charge un **modèle ML sauvegardé avec `joblib`**
* Le modèle prédit :

  * ✅ les ingrédients **recommandés**
  * ❌ les ingrédients **à éviter**

Les résultats sont affichés sous forme claire et pédagogique.

---

### 4️⃣ Recommandation de produits finis

À partir :

* des ingrédients recommandés
* du type de produit choisi

Le backend filtre une base de données de produits cosmétiques et renvoie une liste de **produits compatibles** (nom, marque, type, ingrédients clés).

---

## 🧠 Machine Learning

## 📊 Description détaillée du dataset

Le jeu de données **SkinSort** contient des informations sur des produits de soin de la peau provenant de plusieurs marques internationales. Chaque ligne correspond à un **produit unique**.

### Colonnes principales

* **Brand** : marque du produit
* **Name** : nom du produit
* **Type** : catégorie du produit (Toner, Sérum, Nettoyant, Crème hydratante, etc.)
* **Country** : pays d’origine de la marque
* **Ingredients** : liste des ingrédients de la formule
* **AfterUse** : bénéfices attendus après utilisation (hydratation, anti-acné, anti-ride, etc.)

Le dataset combine ainsi des **variables textuelles**, **catégorielles** et **multi-label**. Les données brutes ont été nettoyées (minuscules, suppression de ponctuation, normalisation des noms d’ingrédients).

---

## 🧪 Protocole expérimental

### 🔍 Analyse Exploratoire des Données (EDA)

**Exploration générale**
Le dataset a été analysé afin de comprendre sa structure et la qualité des données. Les colonnes critiques (Brand, Ingredients, AfterUse) sont bien renseignées dans la majorité des cas, ce qui a permis de conserver l’essentiel des enregistrements sans perte significative.

**Traitement des variables catégorielles et textuelles**
Les colonnes *Ingredients* et *AfterUse* sont multi-label. Elles ont été transformées en représentations binaires via **One-Hot Encoding**. Chaque ingrédient ou effet est représenté par une colonne indiquant sa présence (1) ou son absence (0).

**Visualisations et statistiques descriptives**

* Répartition par type de produit : dominance des sérums et hydratants
* Distribution du nombre d’ingrédients : majorité entre 20 et 25 ingrédients
* Distribution du nombre d’effets : principalement entre 1 et 5 effets
* Top ingrédients fréquents : glycérine, acide hyaluronique, etc.
* Top effets fréquents : hydratation, anti-âge, anti-acné
* Matrice de co-occurrence des effets mettant en évidence des associations fréquentes (ex. hydratant + éclat)

---

### ⚙️ Prétraitement des données

* Extraction et normalisation des ingrédients
* Regroupement des ingrédients en **catégories fonctionnelles** (hydratant, antioxydant, rétinol, peptides, etc.)
* Encodage one-hot des variables Brand, Type et Country
* Sélection de labels pour certaines approches (F1 > 60 %)
* Séparation train/test via **stratification multi-label itérative (Sechidis)** avec 33 % de données en test

---

### 🧠 Approches testées

1️⃣ **Tous ingrédients** : prédiction de tous les ingrédients individuels
→ Haute dimension, forte imbalance, performances faibles sur ingrédients rares.

2️⃣ **Ingrédients (F1 > 60 %)** : prédiction uniquement des ingrédients bien appris
→ Bonnes performances mais comportement instable en conditions réelles.

3️⃣ **Catégories fonctionnelles (final)** : prédiction de catégories d’ingrédients
→ Réduction du nombre de classes (~25), meilleure stabilité et interprétabilité.
<img width="608" height="682" alt="image" src="https://github.com/user-attachments/assets/43790f19-8fdd-48be-a0d1-9333d2dca4b1" />
 Tableau 1 – Performances comparées des approches. Les valeurs sont indicatives (moyennes globaux) : on note la supériorité du SVM sur le RF et l’amélioration progressive des métriques en allant vers les approches “F1>60 %” et “Catégories”.

---

### 🤖 Modélisation et choix du modèle

Plusieurs algorithmes ont été évalués : **KNN**, **Random Forest**, **SVM**.

* **KNN** : écarté (coût élevé en haute dimension)
* **Random Forest** : robuste mais plus lent et moins performant
* **SVM RBF** : meilleures performances globales et inférence rapide

Le modèle final est un **SVM à noyau RBF encapsulé dans un MultiOutputClassifier**, permettant la prédiction multi-label.

---

### 📈 Optimisation et évaluation

* Recherche d’hyperparamètres via **Random Search** puis **Grid Search**
* Métriques utilisées : F1-score (micro/macro), précision, rappel, Hamming Loss

Résultat final :

* **F1-score global ≈ 0.70**
* **Hamming Loss ≈ 0.16** (84 % des étiquettes bien prédites)
* Temps de prédiction rapide, adapté à une application interactive

👉 Le **SVM RBF + catégories fonctionnelles** a été retenu comme meilleur compromis entre performance, robustesse et interprétabilité.

---

## 📊 Visualisations et courbes d’évaluation

Afin de mieux comprendre les données et d’évaluer les performances du modèle, plusieurs **graphes et visualisations** ont été générés durant l’analyse et l’entraînement.

### 📈 1. Répartition des types de produits

**Objectif :** comprendre la distribution des catégories de produits dans le dataset.

* Type de graphe : diagramme en barres
* Interprétation : permet d’identifier les classes dominantes (ex. Sérum, Crème hydratante)



---

### 🧪 2. Distribution du nombre d’ingrédients par produit

**Objectif :** analyser la complexité des formulations cosmétiques.

* Type de graphe : histogramme
* Observation : la majorité des produits contiennent entre 20 et 25 ingrédients
<img width="610" height="353" alt="image" src="https://github.com/user-attachments/assets/5ca49d53-0e64-45b7-92a3-59e5b10554b3" />


---

### 🔗 3. Matrice de co-occurrence des effets (AfterUse)

**Objectif :** identifier les associations fréquentes entre effets cosmétiques.

* Type de graphe : heatmap
* Exemple : Hydratant + Éclat apparaissent souvent ensemble
<img width="606" height="357" alt="image" src="https://github.com/user-attachments/assets/0eccb1ce-8319-44f5-b16d-3e5ea0627d25" />


---

### 📉 4.  performance du modèle

**Objectif :** évaluer quantitativement le modèle multi-label.

#### a) F1-score par catégorie fonctionnelle

* Permet d’identifier les classes bien ou mal apprises
<img width="1138" height="509" alt="image" src="https://github.com/user-attachments/assets/e976019d-53a3-4ca3-beed-6a73e5f4d5b3" />


#### b) metrics  globale

<img width="1040" height="693" alt="image" src="https://github.com/user-attachments/assets/7955ec29-687c-49c8-9085-58812ebbb664" />

#### c) La courbe ROC: 
**Objectif :** évaluer la capacité d’un modèle à distinguer les classes, et plus la courbe s’approche du coin supérieur gauche, meilleure est la performance.

<img width="454" height="349" alt="image" src="https://github.com/user-attachments/assets/55153a85-40d8-48db-82b9-a94f0b433ed0" />
#### d)La courbe DET 
**Objectif :** comparer plus finement les erreurs des systèmes de détection, surtout quand les erreurs sont rares.
<img width="452" height="333" alt="image" src="https://github.com/user-attachments/assets/17866794-ebbe-45a3-afdf-117e9b068f6f" />

---



### 🏋️ Entraînement du modèle

Le script `train_model.py` :

* charge le CSV
* entraîne un modèle 
* sauvegarde le modèle avec `joblib`


---

### 📦 Contenu du fichier `final_model.joblib`



Ce fichier est chargé **au démarrage du backend Flask**.

---

## 🚀 Backend Flask (API)

Le backend expose deux endpoints principaux :

### 🔹 POST `/predict`

Utilisé pour recommander les ingrédients.

**Entrée :**

```json
{
  "features": [1, 0, 1, 0, ...],
  "product_type": "Serum"
}
```

**Sortie :**

```json
{
  "ingredients": {
    "niacinamide": "Yes",
    "alcohol": "No"
  }
}
```

---

### 🔹 POST `/filter-products`

Utilisé pour recommander des produits finis.

**Entrée :**

```json
{
  "ingredients": {...},
  "product_type": "Serum"
}
```

**Sortie :**

```json
{
  "products": [
    {"name": "Product A", "brand": "Brand X"}
  ]
}
```

---

## ⚛️ Frontend React + Vite

Le frontend :

* gère l’interface utilisateur
* guide l’utilisateur étape par étape
* communique avec Flask via `fetch`
* affiche les résultats de manière visuelle et intuitive

Le fichier `App.jsx` contient toute la logique de navigation, de sélection et d’affichage.
<img width="965" height="575" alt="Capture d&#39;écran 2026-01-18 143222" src="https://github.com/user-attachments/assets/5b547a62-c90a-4449-929d-deff70441fe1" />
<img width="986" height="764" alt="Capture d&#39;écran 2026-01-18 143240" src="https://github.com/user-attachments/assets/b3698298-bde6-44c2-af08-95b9617e0ede" />
<img width="946" height="805" alt="Capture d&#39;écran 2026-01-18 143253" src="https://github.com/user-attachments/assets/10ce79f2-9788-4686-9e0d-c88efff51295" />
<img width="968" height="865" alt="Capture d&#39;écran 2026-01-18 143310" src="https://github.com/user-attachments/assets/149b7bed-4234-4df0-b7b6-03634028b66a" />
<img width="979" height="881" alt="Capture d&#39;écran 2026-01-18 143320" src="https://github.com/user-attachments/assets/e63c4fe6-52b2-4e57-84ab-2cc63a5a005d" />

---

## ✅ Résumé

* GlowGuide AI est une **application intelligente de recommandation cosmétique**
* Elle combine **React**, **Flask** et **Machine Learning**
* Les recommandations sont **personnalisées**, basées sur les choix de l’utilisateur
* Le modèle est entraîné à partir de données réelles (CSV)

---

📌 Cette application peut être utilisée comme projet académique, démonstration IA ou base pour une application réelle.
