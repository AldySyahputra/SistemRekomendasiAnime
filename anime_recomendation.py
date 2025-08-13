import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import os
import logging
from typing import List, Tuple
from tabulate import tabulate
import time
import sys
import numpy as np

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ASCII Art untuk tampilan
ANIME_BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                     SISTEM REKOMENDASI ANIME                              ║
║                   -------------------------------                          ║
║          Temukan anime baru berdasarkan anime favorit Anda!              ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

LOADING_MESSAGES = [
    "Menganalisis database anime...",
    "Menghitung kesamaan antar anime...",
    "Mempersiapkan rekomendasi terbaik...",
    "Hampir selesai..."
]

def validate_csv_file(file_path: str) -> bool:
    """
    Memvalidasi format dan isi file CSV.
    """
    try:
        # Cek ukuran file
        if os.path.getsize(file_path) == 0:
            logger.error("File anime.csv kosong")
            return False
        
        # Coba baca beberapa baris pertama
        df = pd.read_csv(file_path, nrows=5)
        
        # Cek kolom yang diperlukan
        required_columns = ['name', 'rating', 'members', 'episodes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_columns)}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error saat memvalidasi file: {str(e)}")
        return False

def show_loading_animation():
    """Menampilkan animasi loading"""
    for message in LOADING_MESSAGES:
        print(f"\r{message}", end="", flush=True)
        time.sleep(0.5)
    print("\n")

def ensure_data_folder():
    """
    Memastikan folder data ada dan memberi tahu user jika file anime.csv tidak ada
    """
    try:
        # Mendapatkan path absolut dari script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data')
        
        # Membuat folder data jika belum ada
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Folder data telah dibuat di: {data_dir}")
        
        anime_file = os.path.join(data_dir, 'anime.csv')
        if not os.path.exists(anime_file):
            print("\n" + "="*70)
            print("File anime.csv tidak ditemukan!")
            print("="*70)
            print("\nUntuk menggunakan program ini:")
            print("1. Download dataset anime dari:")
            print("   https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database")
            print(f"2. Simpan file 'anime.csv' ke dalam folder:")
            print(f"   {data_dir}")
            print("3. Jalankan program ini kembali")
            raise FileNotFoundError(f"File anime.csv tidak ditemukan di: {data_dir}")
        
        # Validasi format file
        if not validate_csv_file(anime_file):
            raise ValueError("Format file anime.csv tidak valid")
            
        return anime_file
    except Exception as e:
        logger.error(f"Error saat memeriksa folder data: {str(e)}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Memuat dataset anime dari file CSV."""
    try:
        logger.info(f"Mencoba membaca file dari: {file_path}")
        data = pd.read_csv(file_path)
        
        # Validasi data
        if data.empty:
            raise ValueError("File CSV kosong")
            
        # Bersihkan data
        data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
        data['members'] = pd.to_numeric(data['members'], errors='coerce')
        data['episodes'] = pd.to_numeric(data['episodes'], errors='coerce')
        
        # Filter data yang valid
        data = data.dropna(subset=['name', 'rating', 'members', 'episodes'])
        
        if len(data) == 0:
            raise ValueError("Tidak ada data valid setelah pembersihan")
            
        logger.info(f"Berhasil membaca {len(data)} data anime")
        return data
    except Exception as e:
        logger.error(f"Error saat memuat data: {str(e)}")
        raise

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Mempersiapkan fitur numerik untuk model rekomendasi, termasuk kolom tambahan.
    """
    try:
        # Daftar kolom numerik potensial
        potential_features = ['rating', 'members', 'episodes', 'score', 'scored_by', 'rank', 'popularity', 'favorites']

        # Filter kolom yang benar-benar ada di DataFrame dan bersifat numerik
        numeric_features = [col for col in potential_features if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not numeric_features:
            raise ValueError("Tidak ada kolom numerik yang cocok untuk fitur ditemukan.")

        # Pilih subset DataFrame dengan fitur yang relevan
        features = df[numeric_features].copy()

        # Menangani nilai yang hilang dengan mean
        features = features.fillna(features.mean())

        # Normalisasi data
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        return pd.DataFrame(features_scaled, columns=features.columns), scaler
    except Exception as e:
        logger.error(f"Error saat mempersiapkan fitur: {str(e)}")
        raise

def create_model(features: pd.DataFrame, n_neighbors: int = 5) -> NearestNeighbors:
    """
    Membuat dan melatih model k-NN.
    
    Args:
        features (pd.DataFrame): Fitur yang telah dinormalisasi
        n_neighbors (int): Jumlah tetangga terdekat
    
    Returns:
        NearestNeighbors: Model k-NN yang telah dilatih
    """
    model = NearestNeighbors(n_neighbors=n_neighbors)
    return model.fit(features)

def find_exact_anime(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Mencari anime yang namanya sesuai dengan query.
    
    Args:
        query (str): Nama anime yang dicari
        df (pd.DataFrame): DataFrame anime
    
    Returns:
        pd.DataFrame: DataFrame berisi anime yang ditemukan
    """
    # Mencari anime yang namanya mengandung query (case insensitive)
    return df[df['name'].str.lower().str.contains(query.lower(), na=False)]

def display_anime_list(matches: pd.DataFrame):
    """
    Menampilkan daftar anime yang ditemukan dengan format yang menarik.
    """
    print("\n" + "="*100)
    print("🎯 HASIL PENCARIAN ANIME".center(100))
    print("="*100)
    
    for idx, anime in matches.iterrows():
        print(f"\n📌 [{matches.index.get_loc(idx) + 1}] {anime['name']}")
        print("─"*100)
        print(f"   📺 Tipe        : {anime.get('type', 'Unknown')}")
        print(f"   ⭐ Rating      : {anime['rating']:.2f}/10")
        print(f"   🎬 Episode     : {int(anime['episodes']) if not pd.isna(anime['episodes']) else 'Unknown'}")
        print(f"   👥 Members     : {int(anime['members']):,}")
        if 'genre' in anime and not pd.isna(anime['genre']):
            print(f"   🏷️  Genre       : {anime['genre']}")
        print("─"*100)

def recommend_anime(anime_name: str, df: pd.DataFrame, features_scaled: pd.DataFrame, n_recommendations: int = 5) -> Tuple[List[dict], pd.Series]:
    """
    Memberikan rekomendasi anime berdasarkan nama anime yang diberikan menggunakan k-NN manual.
    """
    try:
        # Mencari anime yang sesuai dengan nama yang dicari
        matching_animes = find_exact_anime(anime_name, df)

        if matching_animes.empty:
            print(f"\n❌ Anime dengan nama '{anime_name}' tidak ditemukan dalam database.")
            print("💡 Cobalah mencari dengan kata kunci lain atau pastikan ejaan sudah benar.")
            return [], None

        if len(matching_animes) > 1:
            display_anime_list(matching_animes)
            print(f"\n💫 Silakan pilih nomor anime yang Anda inginkan (1-{len(matching_animes)}): ")
            try:
                choice = int(input("➤ "))
                if 1 <= choice <= len(matching_animes):
                    target_anime = matching_animes.iloc[choice-1]
                    target_anime_idx = target_anime.name
                else:
                    print("\n❌ Nomor yang Anda pilih tidak valid.")
                    return [], None
            except ValueError:
                print("\n❌ Mohon masukkan nomor yang valid.")
                return [], None
        else:
            target_anime = matching_animes.iloc[0]
            target_anime_idx = target_anime.name

        # Mendapatkan fitur anime target
        target_features = features_scaled.iloc[target_anime_idx].values.reshape(1, -1)

        # Menghitung jarak Euclidean antara anime target dan semua anime lain
        # Menggunakan broadcasting untuk efisiensi
        distances = np.linalg.norm(features_scaled.values - target_features, axis=1)

        # Membuat DataFrame sementara untuk menyimpan jarak dan index asli
        distances_df = pd.DataFrame({'distance': distances, 'index': df.index})

        # Mengurutkan berdasarkan jarak dan mengambil n_recommendations teratas (kecuali anime target itu sendiri)
        # Pastikan untuk mengecualikan anime target itu sendiri dari hasil
        recommended_indices = distances_df.sort_values(by='distance')['index'].tolist()
        recommended_indices = [idx for idx in recommended_indices if idx != target_anime_idx][:n_recommendations]

        recommendations = []
        # Untuk menghitung similarity score, kita bisa menggunakan 1 - (jarak / jarak_maksimum)
        # Untuk jarak maksimum, kita bisa ambil jarak terjauh dari rekomendasi yang dipilih
        max_distance_in_recs = distances_df[distances_df['index'].isin(recommended_indices)]['distance'].max()

        for idx in recommended_indices:
            anime = df.iloc[idx]
            # Hitung similarity score
            distance_to_rec = distances_df[distances_df['index'] == idx]['distance'].iloc[0]
            # Hindari pembagian dengan nol jika hanya ada satu rekomendasi
            similarity_score = 1 - (distance_to_rec / max_distance_in_recs) if max_distance_in_recs > 0 else 1

            recommendations.append({
                'name': anime['name'],
                'rating': anime['rating'],
                'episodes': anime['episodes'],
                'type': anime.get('type', 'Unknown'),
                'members': anime['members'],
                'genre': anime.get('genre', 'Unknown'),
                'similarity_score': similarity_score
            })

        return recommendations, target_anime
    except Exception as e:
        logger.error(f"Error saat memberikan rekomendasi: {str(e)}")
        return [], None

def display_recommendations(recommendations: List[dict], target_anime):
    """Menampilkan rekomendasi dalam format tabel yang menarik."""
    if not recommendations:
        print("\n❌ Tidak dapat menemukan rekomendasi.")
        print("💡 Pastikan nama anime ditulis dengan benar.")
        return

    print("\n" + "="*100)
    print("📌 INFORMASI ANIME YANG DICARI".center(100))
    print("="*100)
    print(f"Judul     : {target_anime['name']}")
    print(f"Tipe      : {target_anime.get('type', 'Unknown')}")
    print(f"Rating    : {target_anime['rating']:.2f}/10")
    print(f"Episode   : {int(target_anime['episodes']) if not pd.isna(target_anime['episodes']) else 'Unknown'}")
    print(f"Members   : {int(target_anime['members']):,}")
    if 'genre' in target_anime and not pd.isna(target_anime['genre']):
        print(f"Genre     : {target_anime['genre']}")
    print("="*100)
    
    print("\n" + "="*100)
    print("🎯 REKOMENDASI ANIME SERUPA".center(100))
    print("="*100)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print("─"*100)
        print(f"   📺 Tipe        : {rec['type']}")
        print(f"   ⭐ Rating      : {rec['rating']:.2f}/10")
        print(f"   🎬 Episode     : {int(rec['episodes']) if not pd.isna(rec['episodes']) else 'Unknown'}")
        print(f"   👥 Members     : {int(rec['members']):,}")
        if 'genre' in rec and rec['genre'] != 'Unknown':
            print(f"   🏷️  Genre       : {rec['genre']}")
        print(f"   📊 Kemiripan   : {rec['similarity_score']:.1%}")
        print("─"*100)
    
    print("\n" + "="*100)

def main():
    """Fungsi utama program."""
    try:
        print(ANIME_BANNER)
        
        # Memastikan folder dan file yang diperlukan tersedia
        anime_file = ensure_data_folder()
        
        # Memuat dan mempersiapkan data
        print("\n📚 Memuat database anime...")
        anime_data = load_data(anime_file)
        
        # Cache untuk fitur
        features_scaled = None
        
        while True:
            try:
                # Meminta input nama anime dari user
                print("\n🔍 Masukkan nama anime yang ingin Anda cari rekomendasi:")
                print("   (atau tekan Enter untuk menggunakan 'Naruto' sebagai contoh)")
                print("   (ketik 'q' untuk keluar)")
                anime_name = input("➤ ").strip()
                
                if anime_name.lower() == 'q':
                    print("\n👋 Terima kasih telah menggunakan Sistem Rekomendasi Anime!")
                    break
                
                if not anime_name:
                    anime_name = 'Naruto'
                    print(f"\n💡 Menggunakan anime default: {anime_name}")
                
                show_loading_animation()
                
                # Inisialisasi fitur jika belum ada
                if features_scaled is None:
                    logger.info("Mempersiapkan fitur...")
                    features_scaled, _ = prepare_features(anime_data)

                logger.info(f"Mencari rekomendasi untuk: {anime_name}")
                recommendations, target_anime = recommend_anime(anime_name, anime_data, features_scaled)
                
                display_recommendations(recommendations, target_anime)
                
                # Tanya user apakah ingin mencari lagi
                print("\n🔄 Apakah Anda ingin mencari rekomendasi lain? (y/n)")
                if input("➤ ").lower() != 'y':
                    print("\n👋 Terima kasih telah menggunakan Sistem Rekomendasi Anime!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Program dihentikan oleh user.")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Silakan coba lagi dengan nama anime yang berbeda.")
                continue
                
    except Exception as e:
        logger.error(f"Terjadi kesalahan: {str(e)}")
        print("\n❌ Terjadi kesalahan saat menjalankan program.")
        print("Pastikan:")
        print("1. File anime.csv ada di folder data")
        print("2. File anime.csv memiliki format yang benar")
        print("3. Anda memiliki izin untuk membaca file tersebut")
        sys.exit(1)

if __name__ == "__main__":
    main()