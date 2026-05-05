# 🔍 SAP ↔ Web Product Matching — Réconciliation automatique de catalogues produits par IA

> **Problème courant dans toute organisation avec un ERP (SAP, Oracle...) et un catalogue e-commerce découplés.**  
> Ce projet automatise en quelques secondes ce qui prenait des semaines à faire manuellement.

🔗 **[👉 Tester la démo en ligne](https://huggingface.co/spaces/alkaren38gmailcom/sap_web)**

---

## 🎯 Le problème business

Dans de nombreuses entreprises, les produits vendus en ligne sont gérés dans **deux systèmes distincts** :
- un **ERP** (ex : SAP) qui centralise toutes les références produits
- un **catalogue web** avec sa propre hiérarchie de catégorie (univers) et de rayons

Ces deux systèmes ne parlent pas le même langage. Résultat : **15% des produits SAP n'apparaissaient pas dans le catalogue web**, rendant le reporting incomplet et les décisions commerciales biaisées.

| Avant ce projet | Après ce projet |
|---|---|
| 15% du CA invisible dans les rapports | 100% du CA attribué à un univers web |
| Classification manuelle : plusieurs semaines | Matching automatique : quelques minutes |
| Décisions basées sur un reporting partiel | Reporting fiable et complet |

---

## 💡 La solution

Un pipeline NLP qui compare automatiquement les libellés SAP aux 600 000 produits du catalogue web, et attribue à chaque référence SAP la catégorie web la plus proche — sans intervention humaine.

```
Fichier SAP (libellés produits)
        ↓
   Encodage SBERT  →  vecteurs sémantiques
        ↓
   Recherche FAISS  →  correspondance la plus proche dans le catalogue web
        ↓
   Score de similarité + export Excel
```

**L'avantage clé :** le modèle comprend le *sens* des libellés, pas seulement les mots exacts.  
`"Fauteuil pèse-personne électronique"` → correctement associé à `Médical | Mobilier et soins`

---

## 📊 Résultats

- ✅ **~80% de correspondances pertinentes** (score ≥ 80%)
- ⚡ **15 000 produits matchés en 2-3 minutes**
- 📉 **Réduction du travail manuel de plusieurs semaines à quelques clics**
- 💶 **15% du CA web rendu visible** dans les rapports par univers

---

## 🚀 Interface Streamlit

L'application permet à n'importe quel utilisateur (sans compétences techniques) de :

1. **Importer** son fichier source (export SAP ou autre ERP)
2. **Choisir** les colonnes à utiliser pour la comparaison
3. **Lancer** le matching en un clic
4. **Visualiser** l'impact sur le CA par univerou catégorie produits (avant / après)
5. **Télécharger** le fichier Excel des correspondances

> 💡 Le référentiel catalogue (560 000 produits, pré-encodé) est chargé automatiquement depuis Hugging Face — aucune configuration nécessaire.

---

## 📈 Visualisation CA — Avant / Après

L'onglet "Visualisation CA" montre concrètement l'impact du matching :

- **Avant** : CA par univers web incomplet (refs SAP exclues du reporting)
- **Après** : CA rectifié par univers, avec la part réconciliée mise en évidence
- **Tableau de gains** : % de CA récupéré par univers grâce au matching

---

## 🧠 Limites & perspectives

- Les scores de similarité ne reflètent pas toujours la pertinence métier → **une validation humaine reste recommandée** pour les scores < 70%
- L'intégration des hiérarchies produits dans la phrase contextuelle améliore fortement la cohérence des résultats
- **Prochaine étape envisagée** : intégration automatique des nouvelles références SAP dès leur création

---

## ⚙️ Stack technique

| Composant | Rôle |
|---|---|
| `Sentence-BERT (all-mpnet-base-v2)` | Encodage sémantique des libellés produits |
| `FAISS (IndexFlatIP)` | Recherche vectorielle rapide sur 600K produits |
| `Streamlit` | Interface utilisateur interactive |
| `Hugging Face Hub` | Stockage et chargement du référentiel pré-encodé |
| `Pandas / NumPy` | Manipulation des données |
| `Plotly` | Visualisations CA avant/après |

---

## 🛠️ Installation locale

```bash
git clone https://github.com/Mak-lab25/sap-web-product-matching
cd sap-web-product-matching
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ Les fichiers pré-encodés (embeddings `.npy`, index `.faiss`, catalogue `.parquet`) sont téléchargés automatiquement depuis Hugging Face au premier lancement (~1-2 min).

---

## 📁 Structure du projet

```
sap-web-product-matching/
├── app.py                  # Interface Streamlit (matching + visualisation CA)
├── requirements.txt        # Dépendances Python
├── README.md
└── samples/
    ├── sap_echantillon.xlsx        # Exemple de fichier source SAP
    └── catalogue_web_sample.xlsx   # Exemple de catalogue web
```

---

## 🤝 Cas d'usage similaires

Ce pipeline est adaptable à tout contexte de **réconciliation de deux référentiels produits** :
- Fournisseur → Distributeur (normalisation des libellés)
- Migration ERP (correspondance entre deux nomenclatures)
- Multi-marketplace (classification automatique vers Cdiscount, Amazon, Fnac...)
