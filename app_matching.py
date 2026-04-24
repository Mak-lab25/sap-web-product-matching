# On importe Streamlit : c’est un outil pour créer une page web avec des boutons et des tableaux
import streamlit as st

# On importe Pandas : c’est une boîte à outils pour lire et manipuler des tableaux (comme Excel)
import pandas as pd

# On importe NumPy : c’est une boîte à outils pour manipuler des nombres
import numpy as np

# On importe FAISS : c’est un moteur très rapide pour trouver ce qui se ressemble
import faiss

# On importe Torch : il permet d’utiliser le processeur ou la carte graphique
import torch

# On importe SentenceTransformer : il transforme des phrases en nombres
from sentence_transformers import SentenceTransformer

# On importe io : il sert à créer un fichier Excel à télécharger
import io


# =========================
# CONFIGURATION DE LA PAGE
# =========================

# On dit à Streamlit comment afficher la page
# - page_title : le titre dans l’onglet du navigateur
# - layout="wide" : la page prend toute la largeur de l’écran
st.set_page_config(
    page_title="Matching SAP ↔ UGAP (MPNet)",
    layout="wide"
)


# On indique où se trouvent les fichiers déjà préparés sur l’ordinateur

# Ce fichier contient les phrases Web transformées en nombres
EMBEDDINGS_PATH = "embeddings_ugap_mnet.npy"

# Ce fichier contient le catalogue Web (comme un gros Excel optimisé)
CATALOGUE_PATH = "produits_ugap_mnet.parquet"

# Ce fichier contient le moteur de recherche FAISS
FAISS_INDEX_PATH = "index_ugap_mnet.faiss"


# =========================
# 1️⃣ CHARGER LE MODÈLE
# =========================

# @st.cache_resource veut dire :
# “Ne refais pas ce travail à chaque fois, garde le résultat en mémoire”
@st.cache_resource
def load_model():
    
    # On charge le modèle MPNet, qui comprend le sens des phrases
    model = SentenceTransformer("all-mpnet-base-v2")

    # On vérifie si l’ordinateur a une carte graphique
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # On envoie le modèle sur la carte graphique ou le processeur
    return model.to(device)


# On appelle la fonction pour charger le modèle
model = load_model()


# =========================
# 2️⃣ CHARGER LE CATALOGUE WEB
# =========================

# @st.cache_data veut dire :
# “Garde les données en mémoire pour ne pas les relire tout le temps”
@st.cache_data
def load_web_catalogue():
    
    # On lit le fichier parquet qui contient les produits Web
    df_web = pd.read_parquet(CATALOGUE_PATH)

    # On s’assure que la colonne PhraseContextuelle est bien propre
    df_web["PhraseContextuelle"] = (
        df_web["PhraseContextuelle"]   # on prend la colonne
        .fillna("")                    # si une case est vide, on met ""
        .astype(str)                   # on transforme tout en texte
        .str.strip()                   # on enlève les espaces au début et à la fin
    )

    # On renvoie le tableau
    return df_web


# On charge le catalogue Web
df_web = load_web_catalogue()


# =========================
# 3️⃣ CHARGER FAISS ET LES EMBEDDINGS
# =========================

# Cette fonction charge le moteur FAISS et les nombres associés
@st.cache_resource
def load_faiss_index():
    
    # On charge le moteur de recherche FAISS depuis le fichier
    index = faiss.read_index(FAISS_INDEX_PATH)

    # On charge les phrases Web déjà transformées en nombres
    embeddings = np.load(EMBEDDINGS_PATH)

    # On renvoie les deux
    return index, embeddings


# On récupère le moteur FAISS et les embeddings Web
index, embeddings_web = load_faiss_index()


# =========================
# 4️⃣ TRANSFORMER LES PHRASES SAP EN NOMBRES
# =========================

# Cette fonction transforme les phrases SAP en nombres
@st.cache_data
def encode_sap_phrases(phrases):
    
    # Le modèle transforme chaque phrase en un vecteur de nombres
    return model.encode(
        phrases,                   # les phrases à transformer
        convert_to_numpy=True,     # on veut un tableau de nombres
        normalize_embeddings=True, # on normalise pour mieux comparer
        batch_size=64              # on traite 64 phrases à la fois
    )


# =========================
# 5️⃣ INTERFACE UTILISATEUR
# =========================

# On affiche un titre sur la page
st.title("🚀 Matching SAP ↔ UGAP (modèle MPNet)")


# On affiche un bouton pour importer un fichier Excel
uploaded_file = st.file_uploader(
    "📂 Importez votre fichier SAP (Excel avec 'Reference' et 'PhraseContextuelle')",
    type=["xlsx"]
)


# Si un fichier a été importé
if uploaded_file:
    
    # On lit le fichier Excel et on garde tout en texte
    df_sap = pd.read_excel(uploaded_file, dtype=str)

    # On vérifie que les colonnes importantes existent
    if "PhraseContextuelle" not in df_sap.columns or "Reference" not in df_sap.columns:
        
        # On affiche un message d’erreur
        st.error("❌ Le fichier SAP doit contenir les colonnes : 'Reference' et 'PhraseContextuelle'")
        
        # On arrête le programme
        st.stop()

    # On affiche un message de réussite
    st.success(f"✅ Fichier SAP chargé ({len(df_sap):,} produits)")

    # On montre les 5 premières lignes du fichier
    st.dataframe(df_sap.head())


    # Si l’utilisateur clique sur le bouton
    if st.button("🚀 Lancer le matching"):
        
        # On affiche un message pendant que l’ordinateur travaille
        with st.spinner("🔎 Calcul des correspondances en cours..."):

            # On transforme les phrases SAP en nombres
            sap_embeddings = encode_sap_phrases(
                df_sap["PhraseContextuelle"].astype(str).tolist()
            )

            # FAISS cherche, pour chaque produit SAP,
            # le produit Web le plus ressemblant
            D, I = index.search(sap_embeddings, k=1)

            # D contient les scores de ressemblance
            scores = D.flatten() * 100

            # I contient les positions des produits Web trouvés
            idxs = I.flatten()

            # On récupère les lignes Web correspondantes
            df_web_match = df_web.iloc[idxs].reset_index(drop=True)

            # On met SAP et Web côte à côte dans un seul tableau
            df_result = pd.concat(
                [
                    df_sap.reset_index(drop=True),
                    df_web_match.add_prefix("Web_")
                ],
                axis=1
            )

            # On ajoute la colonne avec le score final
            df_result["Score_similarite_cosinus"] = scores.round(2)

        # Message indiquant que tout est fini
        st.success(f"✅ Matching terminé ({len(df_result):,} correspondances calculées)")

        # On affiche les 10 premières correspondances
        st.dataframe(df_result.head(10), use_container_width=True)


        # =========================
        # EXPORT DU FICHIER EXCEL
        # =========================

        # On crée un fichier Excel en mémoire
        buffer = io.BytesIO()

        # On écrit le tableau dans le fichier Excel
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_result.to_excel(writer, index=False, sheet_name="Correspondances")

        # On affiche un bouton pour télécharger le fichier
        st.download_button(
            label="📥 Télécharger le fichier Excel",
            data=buffer,
            file_name="comparaison_SAP_Web_MPNet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# Si aucun fichier n’a encore été importé
else:
    
    # On affiche un message pour guider l’utilisateur
    st.info("⬆️ Importez un fichier SAP pour commencer la comparaison.")