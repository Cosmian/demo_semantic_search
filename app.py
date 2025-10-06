import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----- Page setup -----
st.set_page_config(page_title="🍲 Semantic Recipe Search", layout="centered")

# ----- Header -----
st.markdown(
    """
    <h1 style='text-align: center;'>🔍 <b>Semantic Recipe Search</b></h1>
    <p style='text-align: center; color: gray;'>
        Find recipes by meaning — not just keywords! <br>
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ----- Recipe database -----
recipes = [
    {"title": "Tarte Tatin ",
     "desc": """**Ingredients:** 6 apples, 100g sugar, 50g butter, 1 puff pastry.  
**Preparation:** Preheat oven to 180°C (350°F). In an oven-safe skillet, caramelize the sugar with the butter. 
Add the peeled and quartered apples. Place the puff pastry on top, tucking in the edges. 
Bake for 25–30 minutes. Invert the tart while still warm before serving."""},

    {"title": "Strawberry Tart",
     "desc": """**Ingredients:** 1 shortcrust pastry, 500g strawberries, 250ml pastry cream.  
**Preparation:** Preheat oven to 180°C (350°F). Roll out the pastry and bake blind for 15 minutes. 
Let cool. Fill with pastry cream and top with sliced strawberries. Serve chilled."""},

    {"title": "Chocolate Mousse",
     "desc": """**Ingredients:** 200g dark chocolate, 4 eggs, 50g sugar, 1 pinch of salt.  
**Preparation:** Melt the chocolate in a bain-marie. Separate whites and yolks. 
Beat the whites with salt until stiff peaks form. Mix yolks with melted chocolate, 
then gently fold in the whites. Chill for at least 3 hours before serving."""},

    {"title": "Crème Brûlée",
     "desc": """**Ingredients:** 500ml cream, 5 egg yolks, 100g sugar, 1 vanilla bean.  
**Preparation:** Preheat oven to 150°C (300°F). Heat the cream with vanilla. 
Whisk yolks with sugar, then add hot cream. Pour into ramekins and bake in a water bath for 40–45 minutes. 
Cool and caramelize the top with a torch before serving."""},

    {"title": "Tiramisu",
     "desc": """**Ingredients:** 250g mascarpone, 3 eggs, 80g sugar, 200g ladyfingers, strong coffee, cocoa powder.  
**Preparation:** Separate egg whites and yolks. Whisk yolks with sugar and mascarpone. 
Beat whites until stiff and fold in. Dip ladyfingers in coffee, layer with cream, repeat, 
and dust with cocoa. Chill for 4 hours before serving."""},
]

# ----- Create embeddings -----
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [r["title"] + " " + r["desc"] for r in recipes]
embeddings = model.encode(texts, normalize_embeddings=True)

# ----- FAISS index -----
d = embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# ----- User interface -----
st.markdown("### 👩‍🍳 What would you like to cook today?")
query = st.text_input("Type your idea here:", "apple tart")

if st.button("🔎 Search"):
    query_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_vec, k=5)

    st.markdown("---")
    st.markdown("### 🍰 Most relevant recipes:")
    for idx, score in zip(I[0], D[0]):
        r = recipes[idx]
        st.subheader(r["title"])
        st.markdown(r["desc"])
        st.caption(f"🔹 Similarity score: {score:.3f}")
        st.divider()


