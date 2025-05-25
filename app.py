import streamlit as st
import pandas as pd
import pickle

# Load dataset dan model KNN
import joblib

@st.cache_data
def load_resources():
    df = pd.read_csv("anime.csv")
    model = joblib.load("knn_recommender_model.pkl")  # ganti dari pickle ke joblib
    return df, model
# Fungsi mencari rekomendasi
def get_recommendations(title, df, model, n_neighbors=6):
    if title not in df['name'].values:
        return ["Anime tidak ditemukan dalam data."]
    
    # TF-IDF genre vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df['genre'].fillna(''))

    # Cari indeks anime yang dipilih
    index = df[df['name'] == title].index[0]

    # Cari tetangga terdekat
    distances, indices = model.kneighbors(tfidf_matrix[index], n_neighbors=n_neighbors)

    # Ambil nama-nama anime rekomendasi
    recommended_indices = indices[0][1:]  # abaikan diri sendiri
    recommendations = df.iloc[recommended_indices]['name'].tolist()
    return recommendations

# UI Streamlit
st.set_page_config(page_title="Rekomendasi Anime", layout="centered")
st.title("🎌 Sistem Rekomendasi Anime")
st.markdown("Masukkan anime favoritmu dan dapatkan rekomendasi serupa berdasarkan genre (Jotjib - KNN).")

# Load data dan model
df_anime, knn_model = load_resources()
anime_titles = df_anime['name'].dropna().unique()

# Input pengguna
selected_anime = st.selectbox("Pilih anime favorit:", sorted(anime_titles))

# Hasil rekomendasi
if selected_anime:
    st.subheader("Rekomendasi untuk kamu:")
    recommended_anime = get_recommendations(selected_anime, df_anime, knn_model)
    for i, title in enumerate(recommended_anime, 1):
        st.write(f"{i}. {title}")
