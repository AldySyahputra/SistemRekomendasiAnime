import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import requests
from typing import List, Tuple
import time
from datetime import datetime, timedelta
from functools import lru_cache
from deep_translator import GoogleTranslator
import json
import os

# Inisialisasi session state jika belum ada
if 'language' not in st.session_state:
    st.session_state.language = 'id' # Default Bahasa Indonesia

# Konfigurasi halaman dengan tema yang lebih menarik
st.set_page_config(
    page_title="Anime Recommendation System",
    page_icon="🎌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan yang lebih menarik
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .anime-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
    }
    .anime-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    h1, h2, h3 {
        color: #1e1e1e;
        font-weight: 700;
    }
    .rating-badge {
        background-color: #ffd700;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        color: black;
        font-weight: bold;
        display: inline-block;
        box-shadow: 0 2px 5px rgba(255,215,0,0.3);
    }
    .genre-tag {
        background-color: #e9ecef;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        color: #495057;
        margin: 0.2rem;
        display: inline-block;
    }
    .anime-title {
        font-size: 1.1em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #2d3436;
        min-height: 3em;
    }
    .anime-info {
        color: #636e72;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    .anime-image {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        height: 300px;
        flex-shrink: 0;
    }
    .anime-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.3s ease;
    }
    .anime-image:hover img {
        transform: scale(1.05);
    }
    .status-badge {
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
    }
    .status-ongoing {
        background-color: #00b894;
        color: white;
    }
    .status-completed {
        background-color: #0984e3;
        color: white;
    }
    .status-unknown {
        background-color: #b2bec3;
        color: white;
    }
    .recommendation-button {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    /* CSS untuk tombol rekomendasi abu-abu */
    .recommendation-button > button {
        background-color: #adb5bd; /* Warna abu-abu */
        color: white; /* Pastikan teks tetap putih */
        width: 100%; /* Jadikan lebar penuh */
    }
    .recommendation-button > button:hover {
        background-color: #ced4da; /* Warna abu-abu lebih terang saat hover */
        color: #495057; /* Ubah warna teks saat hover jika diinginkan */
    }
    </style>
""", unsafe_allow_html=True)

# Judul aplikasi dengan desain yang lebih menarik
st.markdown("<h1 style='text-align: center; color: #ff4b4b; margin-bottom: 2rem;'>🎌 YumeList Recommendation</h1>", unsafe_allow_html=True)

# Tambahkan teks deskripsi di bawah judul utama
st.markdown("<p style='text-align: center; margin-top: -1.5rem; margin-bottom: 2rem; color: black; font-weight: bold; font-size: 1.1em;'>さあ、始めよう！Temukan Anime Favoritmu Menggunakan Sistem Rekomendasi Kami</p>", unsafe_allow_html=True)

REVIEWS_FILE = "reviews.json"

def load_reviews():
    if not os.path.exists(REVIEWS_FILE):
        return {}
    try:
        with open(REVIEWS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_reviews(reviews):
    with open(REVIEWS_FILE, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

# Cache untuk menyimpan hasil API
@st.cache_data(ttl=3600)  # Cache selama 1 jam
def get_anime_data(page: int, limit: int = 25) -> dict:
    """Fungsi helper untuk mengambil data anime dari API dengan penanganan error"""
    try:
        url = f"https://api.jikan.moe/v4/top/anime?page={page}&limit={limit}"
        response = requests.get(url, timeout=30)  # Timeout diperbesar menjadi 30 detik
        response.raise_for_status()  # Raise exception untuk status code selain 200
        return response.json()
    except requests.exceptions.RequestException as e:
        # Jika timeout, beri pesan khusus
        if "timed out" in str(e):
            return {"error": "timeout"}
        return {"error": str(e)}

# Load data anime dari Jikan API
@st.cache_data(ttl=3600)  # Aktifkan kembali cache
#@st.cache_data(ttl=3600)  # Nonaktifkan cache sementara untuk debugging
def load_anime_data():
    try:
        new_anime_list = []
        max_retries = 3
        pages_to_fetch = 50  # Ambil 50 halaman saja
        items_per_page = 25  # Tetap 25 item per halaman
        
        # Container untuk progress bar dan status
        progress_container = st.empty()
        status_container = st.empty()
        
        with st.spinner('Memuat data anime...'):
            for page in range(1, pages_to_fetch + 1):
                if len(new_anime_list) >= 1000:  # Maksimal 1000 anime
                    break
                retries = 0
                while retries < max_retries:
                    status_container.info(f"Mengambil halaman {page} dari {pages_to_fetch}...")
                    # Update progress bar
                    progress = min(len(new_anime_list) / 1000, 1.0)
                    progress_container.progress(progress)
                    result = get_anime_data(page, items_per_page)
                    if "error" in result:
                        if result["error"] == "timeout":
                            status_container.warning("Timeout, mencoba lagi setelah 5 detik...")
                            time.sleep(5)  # Delay lebih lama jika timeout
                            retries += 1
                            continue
                        if "429" in str(result["error"]):  # Rate limit
                            status_container.warning("Terkena rate limit, menunggu...")
                            time.sleep(2)  # Delay lebih lama jika rate limit
                            retries += 1
                            continue
                        else:
                            status_container.error(f"Error: {result['error']}")
                            retries += 1
                            if retries >= max_retries:
                                status_container.error(f"Gagal mengambil data halaman {page} setelah {max_retries} percobaan, lanjut ke halaman berikutnya.")
                                break  # Lanjut ke halaman berikutnya
                            time.sleep(2)
                            continue
                    # Proses data anime
                    for anime in result.get("data", []):
                        try:
                            image_url = anime['images']['jpg']['image_url'] if anime['images']['jpg']['image_url'] else "https://cdn.myanimelist.net/images/anime/4/19644.jpg"
                            new_anime = {
                                'name': anime['title'],
                                'rating': float(anime['score']) if anime['score'] else 0.0,
                                'type': anime['type'] or "Unknown",
                                'episodes': int(anime['episodes']) if anime['episodes'] else 0,
                                'genre': ', '.join([genre['name'] for genre in anime['genres']]) if anime['genres'] else "Unknown",
                                'members': int(anime['members']) if anime['members'] else 0,
                                'popularity': int(anime['popularity']) if anime['popularity'] else 0,
                                'status': anime['status'] or "Unknown",
                                'aired_from': anime['aired']['from'],
                                'synopsis': anime['synopsis'] or "Tidak ada sinopsis tersedia.",
                                'image_url': image_url
                            }
                            new_anime_list.append(new_anime)
                            if len(new_anime_list) >= 1000:  # Maksimal 1000 anime
                                break
                        except Exception as e:
                            continue
                    # Berhasil mendapatkan data, lanjut ke halaman berikutnya
                    time.sleep(1.5)  # Delay antar request agar tidak terlalu cepat
                    break  # Keluar dari loop retry
                # Jika gagal setelah retry, lanjut ke halaman berikutnya
                if retries >= max_retries:
                    continue
        
        # Bersihkan progress dan status
        progress_container.empty()
        status_container.empty()
        
        if not new_anime_list:
            return pd.DataFrame()
        
        # Buat DataFrame
        df = pd.DataFrame(new_anime_list)
        
        # Konversi kolom tanggal
        df['aired_from'] = pd.to_datetime(df['aired_from'], errors='coerce')
        df['year'] = df['aired_from'].dt.year
        
        # Isi nilai NaN dengan nilai default
        df['rating'] = df['rating'].fillna(0.0)
        df['members'] = df['members'].fillna(0)
        df['episodes'] = df['episodes'].fillna(0)
        df['synopsis'] = df['synopsis'].fillna("Tidak ada sinopsis tersedia.")
        df['genre'] = df['genre'].fillna("Unknown")
        df['status'] = df['status'].fillna("Unknown")
        df['type'] = df['type'].fillna("Unknown")
        
        # Urutkan berdasarkan rating tertinggi
        df = df.sort_values(by=['rating', 'popularity'], ascending=[False, True])
        
        return df
        
    except Exception as e:
        st.error(f"Error saat memuat data: {str(e)}")
        return pd.DataFrame()

# Load data
anime_df = load_anime_data()

if anime_df.empty:
    st.error("Tidak dapat memuat data anime. Silakan coba lagi nanti.")
    st.stop()

# Pilih anime populer dengan rating tinggi (1000 anime)
popular_anime = anime_df[
    (anime_df['members'] > 50000) &  # Menurunkan threshold members
    (anime_df['rating'] > 7.0)  # Menurunkan threshold rating
].sort_values('rating', ascending=False)

# Konversi ke format yang kita gunakan
latest_animes = []
for _, row in popular_anime.iterrows():
    anime_dict = {
        "name": row['name'],
        "image_url": row['image_url'],  # Gunakan URL gambar yang sudah diambil
        "year": int(row['year']) if pd.notnull(row['year']) else "Unknown",
        "status": row['status'],
        "rating": float(row['rating']),
        "type": row['type'],
        "episodes": int(row['episodes']) if pd.notnull(row['episodes']) else 0,
        "genres": str(row['genre']).split(', '),
        "synopsis": row['synopsis'] if pd.notnull(row['synopsis']) else "Tidak ada sinopsis tersedia."
    }
    latest_animes.append(anime_dict)

# Fungsi untuk mencari anime dengan tampilan yang lebih baik
@st.cache_data(ttl=3600)
def search_anime(query: str) -> List[dict]:
    query = query.lower()
    return [
        anime for anime in latest_animes
        if query in anime["name"].lower() or
        query in " ".join(anime["genres"]).lower() or
        (anime["synopsis"] and query in anime["synopsis"].lower())
    ]

# Fungsi rekomendasi yang ditingkatkan
@st.cache_data(ttl=3600)
def get_anime_recommendations(selected_anime: str, n_recommendations: int = 5) -> List[dict]:
    selected = next((a for a in latest_animes if a["name"].lower() == selected_anime.lower()), None)
    if not selected:
        return []
    
    recommendations = []
    for anime in latest_animes:
        if anime["name"] != selected["name"]:
            # Menghitung kecocokan berdasarkan genre, rating, dan tipe
            common_genres = set(anime["genres"]) & set(selected["genres"])
            genre_similarity = len(common_genres) / len(set(anime["genres"]) | set(selected["genres"]))
            rating_similarity = 1 - abs(anime["rating"] - selected["rating"]) / 10
            type_similarity = 1 if anime["type"] == selected["type"] else 0
            
            # Skor akhir dengan bobot yang disesuaikan
            similarity = (
                genre_similarity * 0.6 + 
                rating_similarity * 0.25 + 
                type_similarity * 0.15
            )
            recommendations.append((anime, similarity))
    
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n_recommendations]

# Fungsi untuk mendapatkan rekomendasi anime menggunakan KNN
@st.cache_data(ttl=3600)
def get_knn_recommendations(selected_anime: str, n_recommendations: int = 5) -> List[dict]:
    # Mengambil fitur yang relevan untuk KNN
    features = anime_df[['rating', 'members']].copy()  # Pastikan ini adalah DataFrame dengan nama kolom
    knn = NearestNeighbors(n_neighbors=n_recommendations)
    knn.fit(features)

    # Mencari indeks anime yang dipilih
    selected_index = anime_df[anime_df['name'].str.lower() == selected_anime.lower()].index
    if selected_index.empty:
        return []

    # Mencari rekomendasi
    selected_features = features.iloc[selected_index[0]].values.reshape(1, -1)  # Mengubah menjadi 2D array
    distances, indices = knn.kneighbors(selected_features)
    
    recommendations = []
    for idx in indices[0]:
        if idx != selected_index[0]:  # Menghindari anime yang sama
            recommendations.append(anime_df.iloc[idx].to_dict())

    return recommendations

def translate_synopsis(synopsis: str, target_language: str) -> str:
    """Fungsi untuk menerjemahkan sinopsis berdasarkan bahasa target"""
    if target_language == 'en':
        return synopsis # Kembalikan sinopsis asli jika target bahasa Inggris
    try:
        # translated = GoogleTranslator(source='auto', target=target_language).translate(synopsis)
        # Ganti dengan kode terjemahan jika target bukan 'en'
        if target_language in ['id', 'ja', 'zh-CN']: # Tambahkan 'ja' dan 'zh-CN'
             translated = GoogleTranslator(source='auto', target=target_language).translate(synopsis) # Gunakan target_language dari parameter
             return translated
        else:
             # Jika target bahasa lain yang tidak ditangani, kembalikan asli saja
             return synopsis

    except Exception as e:
        st.error(f"Terjadi kesalahan saat menerjemahkan sinopsis: {str(e)}")
        return synopsis  # Kembalikan sinopsis asli jika terjadi kesalahan

# Tampilan utama dengan tabs yang lebih menarik
tabs = st.tabs(["🏠 Beranda", "🔍 Pencarian", "⭐ Top Anime"])

# Tab Beranda
with tabs[0]:
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>📺 Anime Terpopuler</h2>", unsafe_allow_html=True)
    
    # Muat ulasan
    reviews = load_reviews()
    
    # Hitung jumlah ulasan untuk setiap anime
    anime_review_counts = {name: len(reviews.get(name, [])) for name in [a['name'] for a in latest_animes]}
    
    # Urutkan daftar anime berdasarkan jumlah ulasan (terbanyak ke tersedikit)
    # Filter hanya anime dengan ulasan (jumlah ulasan > 0)
    sorted_anime_by_reviews = sorted(
        [anime for anime in latest_animes if anime_review_counts.get(anime['name'], 0) > 0],
        key=lambda x: anime_review_counts.get(x['name'], 0),
        reverse=True
    )
    
    # Tampilkan 6 anime teratas dengan ulasan terbanyak
    # Gunakan daftar yang sudah diurutkan
    cols = st.columns(3)
    # Ubah loop untuk menggunakan sorted_anime_by_reviews
    for idx, anime in enumerate(sorted_anime_by_reviews[:6]):
        with cols[idx % 3]:
            with st.container():
                st.markdown(f"""
                    <div class='anime-card'>
                        <div class='anime-image'>
                            <img src="{anime['image_url']}" alt="{anime['name']}">
                        </div>
                        <div class='anime-content' style='flex-grow: 1;'>
                            <h3 class='anime-title'>{anime['name']}</h3>
                            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                <span class='rating-badge'>⭐ {anime['rating']:.2f}</span>
                                <span class='status-badge status-{str(anime.get('status', 'Unknown')).lower().replace(' ', '-')}'>{anime.get('status', 'Unknown')}</span>
                            </div>
                            <div style='margin-bottom: 0.5rem;'>
                                {''.join([f'<span class="genre-tag">{genre}</span>' for genre in str(anime.get('genre', '')).split(', ')[:3]])}
                            </div>
                            <div class='anime-info'>
                                <p>📅 {anime['year']} • {anime['type']} • {anime['episodes']} episodes<br>👥 Members: {anime.get('members', 0):,}</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📖 Sinopsis"):
                    # Tambahkan pilihan bahasa di sini
                    synopsis_language = st.selectbox(
                        "Pilih Bahasa Sinopsis:",
                        ('Indonesia', 'English', 'Jepang', 'Mandarin'),
                        key=f"home_synopsis_lang_{idx}" # Kunci unik
                    )
                    lang_code = 'id' if synopsis_language == 'Indonesia' else \
                                'en' if synopsis_language == 'English' else \
                                'ja' if synopsis_language == 'Jepang' else \
                                'zh-CN' # Kode untuk Mandarin
                    
                    synopsis_text = anime["synopsis"] if pd.notnull(anime["synopsis"]) else "Tidak ada sinopsis tersedia."
                    if pd.isna(synopsis_text):
                        synopsis_text = "Tidak ada sinopsis tersedia."
                    st.write(translate_synopsis(synopsis_text, lang_code))
                
                # ====== Fitur Ulasan & Komentar ======
                reviews = load_reviews()
                anime_reviews = reviews.get(anime['name'], [])
                review_count = len(anime_reviews)
                review_title = f"💬 Ulasan{f' ({review_count})' if review_count > 0 else ''}"

                with st.expander(review_title):
                    if anime_reviews:
                        for ridx, review in enumerate(anime_reviews):
                            st.markdown(f"**{review['user']}**: {review['text']}")
                            # Komentar pada ulasan
                            for cidx, comment in enumerate(review.get('comments', [])):
                                st.markdown(f"<span style='margin-left:2em; color:gray;'>↳ <b>{comment['user']}</b>: {comment['text']}</span>", unsafe_allow_html=True)
                            # Form komentar
                            with st.form(f"home_comment_form_{idx}_{anime['name']}_{ridx}"):
                                comment_user = st.text_input("Nama Anda (Komentar)", key=f"home_comment_user_{anime['name']}_{ridx}")
                                comment_text = st.text_area("Komentar", key=f"home_comment_text_{anime['name']}_{ridx}")
                                submit_comment = st.form_submit_button("Kirim Komentar")
                                if submit_comment and comment_user and comment_text:
                                    if 'comments' not in review:
                                        review['comments'] = []
                                    review['comments'].append({"user": comment_user, "text": comment_text})
                                    save_reviews(reviews)
                                    st.success("Komentar berhasil ditambahkan!")
                                    st.rerun()
                    else:
                        st.info("Belum ada ulasan untuk anime ini.")

                # Form tambah ulasan
                with st.expander("✍️ Tulis Ulasan Anda"):
                    # Tambahkan idx dari loop utama untuk keunikan
                    with st.form(f"home_review_form_{idx}_{anime['name']}"):
                        user = st.text_input("Nama Anda", key=f"home_user_{idx}_{anime['name']}")
                        review_text = st.text_area("Tulis ulasan Anda", key=f"home_review_text_{idx}_{anime['name']}")
                        # Tambahkan idx dari loop utama untuk keunikan dropdown rating
                        rating = st.selectbox("Rating (1-10)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=f"home_rating_{idx}_{anime['name']}")
                        submit_review = st.form_submit_button("Kirim Ulasan")
                        if submit_review and user and review_text:
                            # Simpan ulasan dengan rating
                            anime_reviews.append({"user": user, "text": review_text, "rating": rating, "comments": []})
                            reviews[anime['name']] = anime_reviews
                            save_reviews(reviews)
                            st.success("Ulasan berhasil ditambahkan!")
                            st.rerun()
                
                # Bungkus tombol 'Lihat Rekomendasi Serupa' dengan div khusus untuk styling
                st.markdown("<div class='recommendation-button'>", unsafe_allow_html=True)
                if st.button(f"🎯 Lihat Rekomendasi Serupa", key=f"home_rec_{idx}"):
                    recommendations = get_anime_recommendations(anime["name"])
                    st.markdown("### 🎯 Rekomendasi Serupa:")
                    for rec_anime, similarity in recommendations:
                        st.markdown(f"""
                            <div class='anime-card' style='display: flex; gap: 1rem;'>
                                <div style='width: 120px;'>
                                    <img src="{rec_anime['image_url']}" style='width: 100%; border-radius: 10px;'>
                                </div>
                                <div style='flex: 1;'>
                                    <h4 class='anime-title'>{rec_anime['name']}</h4>
                                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                        <span class='rating-badge'>⭐ {rec_anime['rating']:.2f}</span>
                                        <span class='rating-badge' style='background-color: #74b9ff;'>Kecocokan: {similarity:.1%}</span>
                                    </div>
                                    <div>
                                        {''.join([f'<span class="genre-tag">{genre}</span>' for genre in rec_anime['genres'][:3]])}
                                    </div>
                                    <div class='anime-info'>
                                        <p>{rec_anime['type']} • {rec_anime['episodes']} episodes</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True) # Tutup div untuk tombol 'Lihat Rekomendasi Serupa'

    # --- Tombol Rekomendasi KNN di Bagian Bawah --- #
    st.markdown("---") # Garis pemisah opsional
    st.markdown("<div class='recommendation-button'>", unsafe_allow_html=True)
    # Dropdown untuk memilih anime sebagai basis rekomendasi
    selected_knn_anime = st.selectbox(
        "Pilih Anime Sebagai Dasar Rekomendasi:",
        options=[anime['name'] for anime in latest_animes],
        key="knn_select_anime"
    )
    if selected_knn_anime:
        st.markdown("### 🔎 Lihat Rekomendasi Lainnya:", unsafe_allow_html=True)
        # Ambil rekomendasi dan jarak dari KNN
        features = anime_df[['rating', 'members']].copy()
        knn = NearestNeighbors(n_neighbors=6)
        knn.fit(features)
        selected_index = anime_df[anime_df['name'].str.lower() == selected_knn_anime.lower()].index
        if not selected_index.empty:
            selected_features = features.iloc[selected_index[0]].values.reshape(1, -1)
            distances, indices = knn.kneighbors(selected_features)
            max_distance = distances[0][1:].max() if len(distances[0]) > 1 else 1.0
            min_distance = distances[0][1:].min() if len(distances[0]) > 1 else 0.0
            cols_knn = st.columns(3)
            shown = 0
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx == selected_index[0]:
                    continue  # Lewati anime yang sama
                similarity = 1 - ((dist - min_distance) / (max_distance - min_distance + 1e-8))  # Normalisasi ke 0-1
                similarity_percent = similarity * 100
                anime = anime_df.iloc[idx].to_dict()
                with cols_knn[shown % 3]:
                    with st.container():
                        st.markdown(f"""
                            <div class='anime-card'>
                                <div class='anime-image'>
                                    <img src=\"{anime['image_url']}\" alt=\"{anime['name']}\">
                                </div>
                                <div class='anime-content' style='flex-grow: 1;'>
                                    <h3 class='anime-title'>{anime['name']}</h3>
                                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                        <span class='rating-badge'>⭐ {anime['rating']:.2f}</span>
                                        <span class='rating-badge' style='background-color: #74b9ff;'>Kemiripan: {similarity_percent:.1f}%</span>
                                        <span class='status-badge status-{str(anime.get('status', 'Unknown')).lower().replace(' ', '-')}'>{anime.get('status', 'Unknown')}</span>
                                    </div>
                                    <div style='margin-bottom: 0.5rem;'>
                                        {' '.join([f'<span class=\"genre-tag\">{genre}</span>' for genre in str(anime.get('genre', '')).split(', ')[:3]])}
                                    </div>
                                    <div class='anime-info'>
                                        <p>📅 {anime['year']} • {anime['type']} • {anime['episodes']} episodes<br>👥 Members: {anime.get('members', 0):,}</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        with st.expander("📖 Sinopsis"):
                            synopsis_text = anime.get('synopsis', "Tidak ada sinopsis tersedia.")
                            if pd.isna(synopsis_text):
                                synopsis_text = "Tidak ada sinopsis tersedia."
                            synopsis_language = st.selectbox(
                                "Pilih Bahasa Sinopsis:",
                                ('Indonesia', 'English', 'Jepang', 'Mandarin'),
                                key=f"knn_synopsis_lang_{shown}_{anime['name']}"
                            )
                            lang_code = 'id' if synopsis_language == 'Indonesia' else \
                                        'en' if synopsis_language == 'English' else \
                                        'ja' if synopsis_language == 'Jepang' else \
                                        'zh-CN'
                            st.write(translate_synopsis(synopsis_text, lang_code))
                        # ====== Fitur Ulasan & Komentar ======
                        reviews = load_reviews()
                        anime_reviews = reviews.get(anime['name'], [])
                        review_count = len(anime_reviews)
                        review_title = f"💬 Ulasan Pengguna{f' ({review_count})' if review_count > 0 else ''}"
                        with st.expander(review_title):
                            if anime_reviews:
                                for ridx, review in enumerate(anime_reviews):
                                    st.markdown(f"**{review['user']}**: {review['text']}")
                                    for cidx, comment in enumerate(review.get('comments', [])):
                                        st.markdown(f"<span style='margin-left:2em; color:gray;'>↳ <b>{comment['user']}</b>: {comment['text']}</span>", unsafe_allow_html=True)
                                    with st.form(f"knn_comment_form_{shown}_{anime['name']}_{ridx}"):
                                        comment_user = st.text_input("Nama Anda (Komentar)", key=f"knn_comment_user_{shown}_{anime['name']}_{ridx}")
                                        comment_text = st.text_area("Komentar", key=f"knn_comment_text_{shown}_{anime['name']}_{ridx}")
                                        submit_comment = st.form_submit_button("Kirim Komentar")
                                        if submit_comment and comment_user and comment_text:
                                            if 'comments' not in review:
                                                review['comments'] = []
                                            review['comments'].append({"user": comment_user, "text": comment_text})
                                            save_reviews(reviews)
                                            st.success("Komentar berhasil ditambahkan!")
                                            st.rerun()
                            else:
                                st.info("Belum ada ulasan untuk anime ini.")
                        with st.expander("✍️ Tulis Ulasan Anda"):
                            with st.form(f"knn_review_form_{shown}_{anime['name']}"):
                                user = st.text_input("Nama Anda", key=f"knn_user_{shown}_{anime['name']}")
                                review_text = st.text_area("Tulis ulasan Anda", key=f"knn_review_text_{shown}_{anime['name']}")
                                rating = st.selectbox("Rating (1-10)", options=[1,2,3,4,5,6,7,8,9,10], key=f"knn_rating_{shown}_{anime['name']}")
                                submit_review = st.form_submit_button("Kirim Ulasan")
                                if submit_review and user and review_text:
                                    anime_reviews.append({"user": user, "text": review_text, "rating": rating, "comments": []})
                                    reviews[anime['name']] = anime_reviews
                                    save_reviews(reviews)
                                    st.success("Ulasan berhasil ditambahkan!")
                                    st.rerun()
                shown += 1
        else:
            st.info("Anime tidak ditemukan dalam data.")
    st.markdown("</div>", unsafe_allow_html=True) # Tutup div untuk tombol 'Lihat Rekomendasi Lainnya'

# Tab Pencarian
with tabs[1]:
    st.markdown("<h2 style='text-align: center;'>🔍 Pencarian Anime</h2>", unsafe_allow_html=True)
    search_query = st.text_input("Cari Anime", placeholder="Masukkan judul, genre, atau kata kunci...", 
                                help="Cari berdasarkan judul, genre, atau kata kunci dalam sinopsis")
    
    # Input untuk rating
    rating_filter = st.number_input("Rating Minimum", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    
    if search_query:
        results = search_anime(search_query)
        
        # Filter hasil berdasarkan rating
        results = [anime for anime in results if anime['rating'] >= rating_filter]
        
        if results:
            st.markdown(f"<p style='text-align: center; font-size: 1.2rem; color: #2d3436;'>Ditemukan {len(results)} hasil untuk '{search_query}' dengan rating >= {rating_filter}</p>", 
                       unsafe_allow_html=True)
            
            # Tampilkan hasil pencarian dalam grid
            cols = st.columns(3)
            for idx, anime in enumerate(results):
                with cols[idx % 3]:
                    with st.container():
                        st.markdown(f"""
                            <div class='anime-card'>
                                <div class='anime-image'>
                                    <img src="{anime['image_url']}" alt="{anime['name']}">
                                </div>
                                <div class='anime-content' style='flex-grow: 1;'>
                                    <h3 class='anime-title'>{anime['name']}</h3>
                                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                        <span class='rating-badge'>⭐ {anime['rating']:.2f}</span>
                                        <span class='status-badge status-{str(anime.get('status', 'Unknown')).lower().replace(' ', '-')}'>{anime.get('status', 'Unknown')}</span>
                                    </div>
                                    <div style='margin-bottom: 0.5rem;'>
                                        {' '.join([f'<span class="genre-tag">{genre}</span>' for genre in str(anime.get('genre', '')).split(', ')[:3]])}
                                    </div>
                                    <div class='anime-info'>
                                        <p>📅 {anime['year']} • {anime['type']} • {anime['episodes']} episodes<br>👥 Members: {anime.get('members', 0):,}</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander("📖 Sinopsis"):
                            # Tambahkan pilihan bahasa di sini
                            synopsis_language = st.selectbox(
                                "Pilih Bahasa Sinopsis:",
                                ('Indonesia', 'English', 'Jepang', 'Mandarin'),
                                key=f"search_synopsis_lang_{idx}" # Kunci unik
                            )
                            lang_code = 'id' if synopsis_language == 'Indonesia' else \
                                        'en' if synopsis_language == 'English' else \
                                        'ja' if synopsis_language == 'Jepang' else \
                                        'zh-CN' # Kode untuk Mandarin
                            
                            synopsis_text = anime["synopsis"] if pd.notnull(anime["synopsis"]) else "Tidak ada sinopsis tersedia."
                            if pd.isna(synopsis_text):
                                synopsis_text = "Tidak ada sinopsis tersedia."
                            st.write(translate_synopsis(synopsis_text, lang_code))
                
                        # ====== Fitur Ulasan & Komentar ======
                        reviews = load_reviews()
                        anime_reviews = reviews.get(anime['name'], [])
                        review_count = len(anime_reviews)
                        review_title = f"💬 Ulasan Pengguna{f' ({review_count})' if review_count > 0 else ''}"

                        with st.expander(review_title):
                            if anime_reviews:
                                for ridx, review in enumerate(anime_reviews):
                                    st.markdown(f"**{review['user']}**: {review['text']}")
                                    # Komentar pada ulasan
                                    for cidx, comment in enumerate(review.get('comments', [])):
                                        st.markdown(f"<span style='margin-left:2em; color:gray;'>↳ <b>{comment['user']}</b>: {comment['text']}</span>", unsafe_allow_html=True)
                                    # Form komentar
                                    with st.form(f"search_comment_form_{idx}_{anime['name']}_{ridx}"):
                                        comment_user = st.text_input("Nama Anda (Komentar)", key=f"search_comment_user_{idx}_{anime['name']}_{ridx}")
                                        comment_text = st.text_area("Komentar", key=f"search_comment_text_{idx}_{anime['name']}_{ridx}")
                                        submit_comment = st.form_submit_button("Kirim Komentar")
                                        if submit_comment and comment_user and comment_text:
                                            if 'comments' not in review:
                                                review['comments'] = []
                                            review['comments'].append({"user": comment_user, "text": comment_text})
                                            save_reviews(reviews)
                                            st.success("Komentar berhasil ditambahkan!")
                                            st.rerun()
                            else:
                                st.info("Belum ada ulasan untuk anime ini.")

                        # Form tambah ulasan
                        with st.expander("✍️ Tulis Ulasan Anda"):
                            with st.form(f"search_review_form_{idx}_{anime['name']}"):
                                user = st.text_input("Nama Anda", key=f"search_user_{idx}_{anime['name']}")
                                review_text = st.text_area("Tulis ulasan Anda", key=f"search_review_text_{idx}_{anime['name']}")
                                # Tambahkan dropdown rating
                                rating = st.selectbox("Rating (1-10)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=f"search_rating_{idx}_{anime['name']}")
                                submit_review = st.form_submit_button("Kirim Ulasan")
                                if submit_review and user and review_text:
                                    anime_reviews.append({"user": user, "text": review_text, "rating": rating, "comments": []})
                                    reviews[anime['name']] = anime_reviews
                                    save_reviews(reviews)
                                    st.success("Ulasan berhasil ditambahkan!")
                                    st.rerun()
                        # ===============================================
                        
                        # Tambahkan tombol Lihat Rekomendasi Serupa di sini
                        st.markdown("<div class='recommendation-button'>", unsafe_allow_html=True)
                        if st.button(f"🎯 Lihat Rekomendasi Serupa", key=f"search_rec_{idx}"): # Gunakan key unik
                            recommendations = get_anime_recommendations(anime["name"])
                            st.markdown("### 🎯 Rekomendasi Serupa:")
                            for rec_anime, similarity in recommendations:
                                st.markdown(f"""
                                    <div class='anime-card' style='display: flex; gap: 1rem;'>
                                        <div style='width: 120px;'>
                                            <img src="{rec_anime['image_url']}" style='width: 100%; border-radius: 10px;'>
                                        </div>
                                        <div style='flex: 1;'>
                                            <h4 class='anime-title'>{rec_anime['name']}</h4>
                                            <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                                <span class='rating-badge'>⭐ {rec_anime['rating']:.2f}</span>
                                                <span class='rating-badge' style='background-color: #74b9ff;'>Kecocokan: {similarity:.1%}</span>
                                            </div>
                                            <div>
                                                {''.join([f'<span class="genre-tag">{genre}</span>' for genre in rec_anime['genres'][:3]])}
                                            </div>
                                            <div class='anime-info'>
                                                <p>{rec_anime['type']} • {rec_anime['episodes']} episodes</p>
                                            </div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True) # Tutup div untuk tombol 'Lihat Rekomendasi Serupa'
        else:
            st.warning("Tidak ditemukan anime yang sesuai dengan pencarian dan rating yang ditentukan.")

# Tab Top Anime
with tabs[2]:
    st.markdown("<h2 style='text-align: center;'>⭐ Top Anime</h2>", unsafe_allow_html=True)
    
    # Tampilkan 20 anime teratas
    for idx, anime in enumerate(latest_animes[:20]):
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(anime["image_url"])
            with col2:
                st.markdown(f"""
                    <div class='anime-card'>
                        <h3>#{idx+1} {anime['name']}</h3>
                        <p><span class='rating-badge'>⭐ {anime['rating']:.2f}</span></p>
                        <p><strong>Genre:</strong> {', '.join(anime['genres'])}</p>
                        <p><strong>Status:</strong> {anime['status']}</p>
                        <p><strong>Type:</strong> {anime['type']} ({anime['episodes']} episodes)</p>
                    </div>
                """, unsafe_allow_html=True)
                with st.expander("Sinopsis"):
                    # Tambahkan pilihan bahasa di sini
                    synopsis_language = st.selectbox(
                        "Pilih Bahasa Sinopsis:",
                        ('Indonesia', 'English', 'Jepang', 'Mandarin'),
                        key=f"top_synopsis_lang_{idx}" # Kunci unik
                    )
                    lang_code = 'id' if synopsis_language == 'Indonesia' else \
                                'en' if synopsis_language == 'English' else \
                                'ja' if synopsis_language == 'Jepang' else \
                                'zh-CN' # Kode untuk Mandarin
                    
                    synopsis_text = anime["synopsis"] if pd.notnull(anime["synopsis"]) else "Tidak ada sinopsis tersedia."
                    if pd.isna(synopsis_text):
                        synopsis_text = "Tidak ada sinopsis tersedia."
                    st.write(translate_synopsis(synopsis_text, lang_code))
                
                # ====== Fitur Ulasan & Komentar ======
                reviews = load_reviews()
                anime_reviews = reviews.get(anime['name'], [])
                review_count = len(anime_reviews)
                review_title = f"💬 Ulasan Pengguna{f' ({review_count})' if review_count > 0 else ''}"

                with st.expander(review_title):
                    if anime_reviews:
                        for ridx, review in enumerate(anime_reviews):
                            st.markdown(f"**{review['user']}**: {review['text']}")
                            # Komentar pada ulasan
                            for cidx, comment in enumerate(review.get('comments', [])):
                                st.markdown(f"<span style='margin-left:2em; color:gray;'>↳ <b>{comment['user']}</b>: {comment['text']}</span>", unsafe_allow_html=True)
                            # Form komentar
                            with st.form(f"top_comment_form_{idx}_{anime['name']}_{ridx}"):
                                comment_user = st.text_input("Nama Anda (Komentar)", key=f"top_comment_user_{idx}_{anime['name']}_{ridx}")
                                comment_text = st.text_area("Komentar", key=f"top_comment_text_{idx}_{anime['name']}_{ridx}")
                                submit_comment = st.form_submit_button("Kirim Komentar")
                                if submit_comment and comment_user and comment_text:
                                    if 'comments' not in review:
                                        review['comments'] = []
                                    review['comments'].append({"user": comment_user, "text": comment_text})
                                    save_reviews(reviews)
                                    st.success("Komentar berhasil ditambahkan!")
                                    st.rerun()
                    else:
                        st.info("Belum ada ulasan untuk anime ini.")

                # Form tambah ulasan
                with st.expander("✍️ Tulis Ulasan Anda"):
                    # Tambahkan idx dari loop utama untuk keunikan
                    with st.form(f"top_review_form_{idx}_{anime['name']}"):
                        user = st.text_input("Nama Anda", key=f"top_user_{idx}_{anime['name']}")
                        review_text = st.text_area("Tulis ulasan Anda", key=f"top_review_text_{idx}_{anime['name']}")
                        # Tambahkan idx dari loop utama untuk keunikan dropdown rating
                        rating = st.selectbox("Rating (1-10)", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], key=f"top_rating_{idx}_{anime['name']}")
                        submit_review = st.form_submit_button("Kirim Ulasan")
                        if submit_review and user and review_text:
                            anime_reviews.append({"user": user, "text": review_text, "rating": rating, "comments": []})
                            reviews[anime['name']] = anime_reviews
                            save_reviews(reviews)
                            st.success("Ulasan berhasil ditambahkan!")
                            st.rerun()
                
                # Tombol Lihat Rekomendasi Serupa
                st.markdown("<div class='recommendation-button'>", unsafe_allow_html=True)
                if st.button(f"🎯 Lihat Rekomendasi Serupa", key=f"top_rec_{idx}"):
                    recommendations = get_anime_recommendations(anime["name"])
                    st.markdown("### 🎯 Rekomendasi Serupa:")
                    for rec_anime, similarity in recommendations:
                        st.markdown(f"""
                            <div class='anime-card' style='display: flex; gap: 1rem;'>
                                <div style='width: 120px;'>
                                    <img src="{rec_anime['image_url']}" style='width: 100%; border-radius: 10px;'>
                                </div>
                                <div style='flex: 1;'>
                                    <h4 class='anime-title'>{rec_anime['name']}</h4>
                                    <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                                        <span class='rating-badge'>⭐ {rec_anime['rating']:.2f}</span>
                                        <span class='rating-badge' style='background-color: #74b9ff;'>Kecocokan: {similarity:.1%}</span>
                                    </div>
                                    <div>
                                        {''.join([f'<span class="genre-tag">{genre}</span>' for genre in rec_anime['genres'][:3]])}
                                    </div>
                                    <div class='anime-info'>
                                        <p>{rec_anime['type']} • {rec_anime['episodes']} episodes</p>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# Sidebar yang lebih informatif
with st.sidebar:
    st.markdown("<h3 style='text-align: center;'>📊 Statistik Anime</h3>", unsafe_allow_html=True)

    # Statistik umum
    st.markdown(f"""
        <div style='background-color: white; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;'>
            <h4>Koleksi Anime</h4>
            <p><strong>{len(anime_df)}</strong> judul anime</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Statistik genre
    st.markdown("<h4>Genre Terpopuler</h4>", unsafe_allow_html=True)
    all_genres = []
    for anime in latest_animes:
        all_genres.extend(anime["genres"])
    genre_counts = pd.Series(all_genres).value_counts()
    
    # Visualisasi genre dengan chart yang lebih sederhana
    if not genre_counts.empty:
        chart_data = pd.DataFrame({
            'Genre': genre_counts.index[:10],
            'Count': genre_counts.values[:10]
        })
        st.bar_chart(chart_data.set_index('Genre'))
    
    # Rating distribution
    st.markdown("<h4>Distribusi Rating</h4>", unsafe_allow_html=True)
    ratings = [anime["rating"] for anime in latest_animes]
    if ratings:
        rating_df = pd.DataFrame({
            'Rating': ratings
        })
        st.bar_chart(rating_df)

# Footer yang lebih menarik
st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p>&copy; 2025 Aldy Syahputra Harianja</p>
        <p style='color: #6c757d; font-style: italic;'><small>Data dari MyAnimeList</small></p>
    </div>
""", unsafe_allow_html=True) 