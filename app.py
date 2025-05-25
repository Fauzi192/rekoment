import streamlit as st
import pandas as pd
import pickle
from sklearn.neighbors import NearestNeighbors

# Load data dan model
@st.cache_data
def load_data():
    df = pd.read_csv('anime.csv')  # pastikan file ini ada
    with open('knn_recommender_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return df, model)

# Load data dan model
anime_df, knn_model = load_data()

# Set halaman
st.set_page_config(page_title="Rekomendasi Anime", layout="centered")
st.title("🎌 Sistem Rekomendasi Anime (KNN)")
st.write("Masukkan judul anime favoritmu untuk mendapatkan rekomendasi.")

# Dropdown anime
anime_list = anime_df['name'].tolist()
selected_anime = st.selectbox("Pilih anime favorit kamu:", anime_list)

# Fungsi rekomendasi
def get_knn_recommendations(selected_title, model, df, n_recommendations=5):
    index = df[df['name'] == selected_title].index[0]
    distances, indices = model.kneighbors([df.iloc[index, 1:]], n_neighbors=n_recommendations + 1)
    recommended_indices = indices[0][1:]  # skip indeks pertama karena itu anime yang sama
    return df['name'].iloc[recommended_indices]

# Tampilkan hasil
if selected_anime:
    st.subheader("Rekomendasi Anime Serupa:")
    recommendations = get_knn_recommendations(selected_anime, knn_model, anime_df)
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
