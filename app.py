import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

# 🎯 Download large files if not present
if not os.path.exists("bert_similarity.pkl"):
    gdown.download("https://drive.google.com/uc?id=1Ej2pZwn2nBA0mRlThNMvrkgDOF-NOjFN", "bert_similarity.pkl", quiet=False)

if not os.path.exists("bert_embeddings.pkl"):
    gdown.download("https://drive.google.com/uc?id=1RNu_jdtYvt3yDnD0-ocPiPGvGWoBfgwG", "bert_embeddings.pkl", quiet=False)

# ✅ Load and cache heavy files
@st.cache_data
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

movies_dict = load_pickle('movie_dict.pkl')
movies = pd.DataFrame(movies_dict)
bert_similarity = load_pickle('bert_similarity.pkl')
bert_embeddings = load_pickle('bert_embeddings.pkl')
movie_sentiment = load_pickle('movie_sentiment.pkl')
df = load_csv("imdb-movies-dataset.csv")
df = df.dropna(subset=['Title', 'Poster', 'Year', 'Rating', 'Genre', 'Description'])

# 🧠 MMR function (for diversity)
def mmr(doc_embeddings, query_embedding, top_n, diversity=0.5):
    sim = cosine_similarity(doc_embeddings, query_embedding.reshape(1, -1)).flatten()
    selected = []
    remaining = list(range(len(sim)))

    for _ in range(top_n):
        if not selected:
            idx = np.argmax(sim)
            selected.append(idx)
            remaining.remove(idx)
            continue
        mmr_scores = []
        for i in remaining:
            redundancy = max(cosine_similarity(doc_embeddings[i].reshape(1, -1),
                                               doc_embeddings[selected]).flatten())
            score = diversity * sim[i] - (1 - diversity) * redundancy
            mmr_scores.append(score)
        idx = remaining[np.argmax(mmr_scores)]
        selected.append(idx)
        remaining.remove(idx)
    return selected

# 🎬 Recommendation logic
def recommend(movie):
    try:
        movie_index = movies[movies['Title'] == movie].index[0]
        query_embedding = bert_embeddings[movie_index]
        top_indices = mmr(bert_embeddings, query_embedding, top_n=15, diversity=0.6)

        scored = []
        for i in top_indices:
            title = movies.iloc[i].Title
            if title == movie:  # 🔥 Skip selected movie itself
                continue
            sim_score = bert_similarity[movie_index][i]
            sent_score = movie_sentiment.get(title.strip(), 0.5)
            final_score = 0.7 * sim_score + 0.3 * sent_score
            scored.append((title, final_score))

        scored = sorted(scored, key=lambda x: x[1], reverse=True)
        recommendations = [title for title, _ in scored[:5]]

        if not recommendations:
            st.warning("⚠️ No recommendations found.")
        return recommendations

    except IndexError:
        st.error("❌ Movie not found in model data.")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return []

# 🎨 Streamlit UI
st.title("🎬 Long-Tail-Aware Movie Recommender")
valid_titles = movies['Title'].values  # movies is from movie_dict.pkl
selected_movie = st.selectbox("Choose a movie", valid_titles)

if st.button("Recommend"):
    st.subheader("📽️ Recommended Movies:")
    for title in recommend(selected_movie):
        st.markdown(f"### 🎬 {title}")
        details = df[df['Title'] == title]
        if not details.empty:
            row = details.iloc[0]
            st.image(row['Poster'], width=150)
            st.write(f"⭐ IMDb: {row['Rating']} | 📅 Year: {int(row['Year'])}")
            st.write(f"🎭 Genre: {row['Genre']}")
            st.write(f"🧾 {row['Description']}")
        else:
            st.write("ℹ️ No metadata available.")
        st.markdown("---")




