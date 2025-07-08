import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# === 📦 Download large files from Google Drive if missing ===
def download_if_missing(filename, gdrive_id):
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={gdrive_id}"
        gdown.download(url, filename, quiet=False)

# 🔽 Replace with your real Google Drive file IDs
download_if_missing("bert_similarity.pkl", "1Ej2pZwn2nBA0mRlThNMvrkgDOF-NOjFN")
download_if_missing("bert_embeddings.pkl", "1RNu_jdtYvt3yDnD0-ocPiPGvGWoBfgwG")
download_if_missing("movie_sentiment.pkl", "https://drive.google.com/uc?id=19TWvwzqVDR4nlDGSivGWuJm3hnVL2X4B")
download_if_missing("movie_dict.pkl", "https://drive.google.com/uc?id=1PhF6Y38J0l5Urn7SsQErmP_R9JuN_o0O")

# === 📂 Load Required Files ===
with open("movie_dict.pkl", "rb") as f:
    movies_dict = pickle.load(f)
movies = pd.DataFrame(movies_dict)

with open("bert_similarity.pkl", "rb") as f:
    bert_similarity = pickle.load(f)

with open("bert_embeddings.pkl", "rb") as f:
    bert_embeddings = pickle.load(f)

with open("movie_sentiment.pkl", "rb") as f:
    movie_sentiment = pickle.load(f)

# Load full metadata for display
df = pd.read_csv("imdb-movies-dataset.csv")
df = df.dropna(subset=['Title', 'Poster', 'Year', 'Rating', 'Genre', 'Description'])
df = df.drop_duplicates(subset=['Title'])

# === 🧠 MMR for Diverse Recs ===
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
            redundancy = max(cosine_similarity(doc_embeddings[i].reshape(1, -1), doc_embeddings[selected]).flatten())
            score = diversity * sim[i] - (1 - diversity) * redundancy
            mmr_scores.append(score)
        idx = remaining[np.argmax(mmr_scores)]
        selected.append(idx)
        remaining.remove(idx)
    return selected

# === 🎯 Final Recommendation Function ===
def recommend(movie):
    if movie not in movies['Title'].values:
        return []

    movie_index = movies[movies['Title'] == movie].index[0]
    query_embedding = bert_embeddings[movie_index]
    top_indices = mmr(bert_embeddings, query_embedding, top_n=20, diversity=0.6)

    scored = []
    for i in top_indices:
        title = movies.iloc[i].Title
        if title == movie:
            continue  # Skip the selected movie itself
        sim_score = bert_similarity[movie_index][i]
        sent_score = movie_sentiment.get(title, 0.5)
        final_score = 0.7 * sim_score + 0.3 * sent_score
        scored.append((title, final_score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [title for title, _ in scored[:5]]

# === 🖼️ Streamlit Frontend ===
st.title("🎬 Long-Tail-Aware Movie Recommender")

valid_titles = movies['Title'].values
selected_movie = st.selectbox("Choose a movie", valid_titles)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    if not recommendations:
        st.error("❌ Movie not found in model data.")
    else:
        st.subheader("📽️ Recommended Movies:")
        for title in recommendations:
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




