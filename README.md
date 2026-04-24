# sap-web-product-matching
Pipeline basé sur le NLP pour cartographier les catalogues de produits SAP et Web, améliorant la visibilité des produits et la qualité des données grâce à SBERT et FAISS.

# 🔍 SAP ↔ Web Product Matching (NLP + FAISS)

## 🎯 Objectif
Réconcilier automatiquement les produits SAP avec le catalogue Web (566k produits) à l’aide de NLP.

## ⚙️ Stack
- Python
- Sentence-BERT (MPNet)
- FAISS
- Streamlit

## 🚀 Pipeline
1. Construction de phrases contextuelles
2. Encodage SBERT
3. Indexation FAISS
4. Matching SAP → Web

## 📊 Résultats
- ~80% correspondances pertinentes
- Matching en quelques secondes

## 🖥️ Lancer le projet
```bash
pip install -r requirements.txt
streamlit run app_matching.py

## 💡 Limites
- Certaines correspondances restent approximatives
- Intervention humaine nécessaire pour validation finale
