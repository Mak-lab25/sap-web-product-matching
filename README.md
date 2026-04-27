# 🔍 SAP ↔ Web Product Matching (NLP + FAISS)

> Pipeline NLP permettant de réconcilier automatiquement des produits SAP avec un catalogue web à grande échelle.

---

## 🎯 Problématique

Une partie des produits SAP n’était pas visible sur le site web, rendant difficile :
- l’analyse du chiffre d’affaires
- la gestion du catalogue
- la cohérence des données

Ces produits représentaient **15% du CA**.

---

## 💡 Solution

Développement d’un système de matching basé sur :

- embeddings SBERT (Sentence-BERT)
- similarité cosinus
- indexation FAISS (recherche rapide)

Le système compare chaque produit SAP aux **600k produits du catalogue web**.

---

## 🚀 Demo

![App Screenshot](screenshot.png)

---

## 📊 Résultats

- ~80% de correspondances pertinentes
- Matching de 15 000 produits en quelques secondes
- Réduction significative du travail manuel

---

## 📈 Impact Business

- **+ visibilité produits** sur le site web
- **+ fiabilité des analyses CA**
- **gain de temps opérationnel**

---

## 🧠 Analyse

Les scores de similarité ne reflètent pas toujours la pertinence métier.

➡️ L’intégration des hiérarchies produits améliore fortement la cohérence  
➡️ Une validation humaine reste nécessaire

---

## ⚙️ Technologies

- Python
- Sentence-BERT (all-mpnet-base-v2)
- FAISS
- Pandas / NumPy
- Streamlit

---

## 🛠️ Installation

```bash
git clone https://github.com/Mak-lab25/sap-web-product-matching
cd sap-web-product-matching
pip install -r requirements.txt
streamlit run app.py
