import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="🍲 Démo Recherche Sémantique", layout="centered")
st.title("🔍 Recherche sémantique de recettes")

# ----- Base de recettes -----
recipes = [
    {"title": "Tarte Tatin",
     "desc": """Ingrédients : 6 pommes, 100g de sucre, 50g de beurre, 1 pâte feuilletée. 
Préparation : Préchauffez le four à 180°C. Dans une poêle allant au four, faites caraméliser le sucre avec le beurre. Ajoutez les pommes épluchées et coupées en quartiers. Disposez la pâte feuilletée par-dessus, en rentrant les bords. Enfournez 25-30 minutes. Retournez la tarte tiède avant de servir."""},

    {"title": "Tarte aux fraises",
     "desc": """Ingrédients : 1 pâte sablée, 500g de fraises, 250ml de crème pâtissière. 
Préparation : Préchauffez le four à 180°C. Étalez la pâte dans un moule et faites-la cuire à blanc 15 min. Laissez refroidir. Garnissez de crème pâtissière et disposez les fraises lavées et coupées. Servez frais."""},

    {"title": "Mousse au chocolat",
     "desc": """Ingrédients : 200g de chocolat noir, 4 œufs, 50g de sucre, 1 pincée de sel. 
Préparation : Faites fondre le chocolat au bain-marie. Séparez les blancs des jaunes. Fouettez les blancs en neige avec une pincée de sel. Mélangez les jaunes avec le chocolat fondu. Incorporez délicatement les blancs en neige. Réfrigérez au moins 3h avant de servir."""},

    {"title": "Crème brûlée",
     "desc": """Ingrédients : 500ml de crème, 5 jaunes d'œufs, 100g de sucre, 1 gousse de vanille. 
Préparation : Préchauffez le four à 150°C. Faites chauffer la crème avec la vanille. Fouettez les jaunes avec le sucre, puis ajoutez la crème chaude. Versez dans des ramequins et faites cuire au bain-marie 40-45 min. Laissez refroidir et caramélisez le dessus au chalumeau."""},

    {"title": "Tiramisu",
     "desc": """Ingrédients : 250g de mascarpone, 3 œufs, 80g de sucre, 200g de biscuits à la cuillère, café fort, cacao en poudre. 
Préparation : Séparez les blancs des jaunes. Fouettez jaunes + sucre + mascarpone. Montez les blancs en neige et incorporez-les. Trempez les biscuits dans le café et disposez-les dans un plat. Étalez une couche de crème, puis une deuxième couche de biscuits et crème. Saupoudrez de cacao. Réfrigérez 4h avant de servir."""},
]

# Query exemple



# ----- Création des embeddings -----
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [r["title"] + " " + r["desc"] for r in recipes]
embeddings = model.encode(texts, normalize_embeddings=True)

# ----- Création de l'index FAISS -----
d = embeddings.shape[1]  # dimension des embeddings
index = faiss.IndexFlatIP(d)  # index pour similarité cosinus
index.add(embeddings)

# ----- Interface Streamlit -----

query = st.text_input("Que veux-tu cuisiner ?", "tarte aux pommes")

if st.button("Rechercher"):
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, k=5)  # top 3 résultats

    st.success("Résultats les plus pertinents :")
    for idx, score in zip(I[0], D[0]):
        r = recipes[idx]
        st.subheader(r["title"])
        st.markdown(r["desc"])
        st.markdown(f"*Score de similarité : {score:.3f}*")
        st.divider()
