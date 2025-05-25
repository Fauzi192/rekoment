import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------- KONFIGURASI STREAMLIT -------------------
st.set_page_config(page_title="Rekomendasi Anime", layout="centered")

st.title("🎌 Sistem Rekomendasi Anime")
st.write("Masukkan judul anime favoritmu dan dapatkan rekomendasi serupa.")

# ------------------- DATASET (sementara hardcoded) -------------------
# Ganti dengan pd.read_csv('anime.csv') jika kamu punya file asli
anime_data = {
    "name": [
        "Naruto", "Bleach", "One Piece", "Death Note", "Attack on Titan",
        "Fullmetal Alchemist", "Demon Slayer", "Jujutsu Kaisen",
        "My Hero Academia", "Tokyo Ghoul"
    ],
    "genre": [
        "Action, Adventure, Super Power", "Action, Supernatural", "Action, Adventure, Fantasy",
        "Mystery, Supernatural, Psychological", "Action, Drama, Fantasy",
        "Action, Military, Fantasy", "Action, Demons, Historical",
        "Action, Supernatural, School", "Action, Comedy, Super Power",
        "Action, Horror, Supernatural"
    ]
}
anime_df = pd.DataFrame('anime.csv')

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    return joblib.load("knn_recommender_model.pkl")

model = load_model()

# ------------------- TF-IDF VECTORIZE -------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(anime_df['genre'])

# ------------------- REKOMENDASI -------------------
def get_recommendations(title, top_n=5):
    if title not in anime_df['name'].values:
        return ["Judul tidak ditemukan."]
    
    idx = anime_df[anime_df['name'] == title].index[0]
    input_vector = tfidf_matrix[idx]
    
    distances, indices = model.kneighbors(input_vector, n_neighbors=top_n+1)
    recommended_indices = indices.flatten()[1:]  # Hilangkan diri sendiri
    
    return anime_df['name'].iloc[recommended_indices].tolist()

# ------------------- UI INPUT -------------------
anime_input = st.text_input("Masukkan judul anime:")

if st.button("Dapatkan Rekomendasi"):
    if anime_input:
        hasil = get_recommendations(anime_input.strip())
        st.write("### Rekomendasi:")
        for rec in hasil:
            st.write(f"- {rec}")
    else:
        st.warning("Silakan masukkan judul terlebih dahulu.")
