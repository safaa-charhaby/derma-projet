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

### 📄 Données d’entraînement

Le modèle est entraîné à partir d’un **fichier CSV** contenant :

* des colonnes de préoccupations cutanées (features)
* une colonne cible indiquant les ingrédients recommandés

Exemple simplifié :

| Acne Fighting | Hydrating | Anti-Aging | Recommended Ingredients |
| ------------- | --------- | ---------- | ----------------------- |
| 1             | 0         | 1          | Niacinamide;Retinol     |

---

### 🏋️ Entraînement du modèle

Le script `train_model.py` :

* charge le CSV
* entraîne un modèle (ex: RandomForest)
* sauvegarde le modèle avec `joblib`

```python
joblib.dump(model_data, "model/final_model.joblib")
```

---

### 📦 Contenu du fichier `final_model.joblib`

```python
{
  "features": ["Acne Fighting", "Hydrating", "Anti-Aging", ...],
  "classifier": trained_model,
  "label_binarizer": mlb
}
```

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
