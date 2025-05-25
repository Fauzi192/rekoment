import streamlit as st
import pickle
import joblib
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Rekomendasi Anime",
    page_icon="🎌",
    layout="centered"
)

# Judul halaman
st.title("🎌 Sistem Rekomendasi Anime")
st.write("Masukkan judul anime favoritmu dan dapatkan rekomendasi serupa.")

# Load model KNN
def load_model():
    model = joblib.load("knn_recommender_model.pkl")
    return model

model = load_model()

# Dummy data (ganti dengan data asli jika tersedia)
anime_titles = [
    "Naruto", "Bleach", "One Piece", "Death Note", "Attack on Titan",
    "Fullmetal Alchemist", "Demon Slayer", "Jujutsu Kaisen", "My Hero Academia", "Tokyo Ghoul"
]
anime_features = np.random.rand(len(anime_titles), 10)

# Fungsi rekomendasi
def get_recommendations(title, k=5):
    if title not in anime_titles:
        return ["Judul tidak ditemukan dalam database."]
    
    idx = anime_titles.index(title)
    vector = anime_features[idx].reshape(1, -1)
    distances, indices = model.kneighbors(vector, n_neighbors=k+1)
    
    recommendations = []
    for i in range(1, len(indices[0])):
        rec_idx = indices[0][i]
        recommendations.append(anime_titles[rec_idx])
    return recommendations

# Input judul anime
anime_input = st.text_input("Masukkan judul anime:")

if st.button("Dapatkan Rekomendasi"):
    if anime_input:
        result = get_recommendations(anime_input.strip())
        st.write("### Hasil Rekomendasi:")
        for rec in result:
            st.write(f"- {rec}")
    else:
        st.warning("Silakan masukkan judul anime terlebih dahulu.")
