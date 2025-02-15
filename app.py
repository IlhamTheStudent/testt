from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset movie
movie = pd.read_csv('D:\skripsi brok\SKRIPSI ILHAM\DEPLOY\movie recommender\movies_5000_v2.csv')
movie['overview'] = movie['overview'].fillna('')

# Inisialisasi TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Transformasi teks overview menjadi vektor numerik
tfidf_matrix = tfidf.fit_transform(movie['overview'])

# Hitung cosine similarity antar film berdasarkan keyword sinopsis
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies(keyword, cosine_sim=cosine_sim, movie_data=movie, top_n=10):
    # Transformasi keyword menjadi vektor TF-IDF
    keyword_tfidf = tfidf.transform([keyword])
    
    # Hitung cosine similarity antara keyword dan semua film
    similarity_scores = cosine_similarity(keyword_tfidf, tfidf_matrix).flatten()
    
    # Urutkan film berdasarkan similarity score tertinggi
    related_movies_idx = similarity_scores.argsort()[-top_n:][::-1]
    
    # Ambil 10 film teratas dan tambahkan similarity_score
    recommendations = movie_data.iloc[related_movies_idx].copy()
    recommendations['similarity_score'] = similarity_scores[related_movies_idx]
    
    # Tambahkan URL IMDb atau TMDb (pastikan dataset memiliki 'imdb_id')
    base_url = "https://www.imdb.com/title/"
    recommendations['url'] = base_url + recommendations['imdb_id'].astype(str) + "/"
    
    # Tambahkan poster (jika tersedia)
    image_base_url = "https://image.tmdb.org/t/p/w500"
    recommendations['poster'] = image_base_url + recommendations['poster_path']

    # Ambil tahun dari release_date jika tersedia
    recommendations['release_year'] = pd.to_datetime(recommendations['release_date'], errors='coerce').dt.year

    return recommendations[['title', 'overview', 'similarity_score', 'url', 'poster', 'release_year']]




# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    keyword = ""

    if request.method == 'POST':
        keyword = request.form['keyword'].strip()

        if not keyword:
            return render_template('index.html', error="Kata kunci tidak boleh kosong.", recommendations=None)

        recommendations = recommend_movies(keyword)

        # Perbaiki pengecekan DataFrame
        if recommendations is not None and not recommendations.empty:
            return render_template('index.html', recommendations=recommendations.to_dict(orient='records'), keyword=keyword)
        else:
            return render_template('index.html', error="Tidak ada hasil ditemukan.", recommendations=None)

    return render_template('index.html', recommendations=None, keyword=keyword)

@app.route('/movie/<title>')
def movie_detail(title):
    # Cari film berdasarkan judul
    movie_data = movie[movie['title'] == title].iloc[0]

    # Pastikan kolom 'poster_path' tersedia dalam dataset
    image_base_url = "https://image.tmdb.org/t/p/w500"
    poster_url = image_base_url + movie_data['poster_path'] if 'poster_path' in movie_data else None

    # Ambil release date jika tersedia
    release_date = movie_data['release_date'] if 'release_date' in movie_data else "Tidak diketahui"

    return render_template('movie_detail.html', 
                           title=movie_data['title'], 
                           overview=movie_data['overview'], 
                           poster=poster_url, 
                           release_date=release_date)



if __name__ == "__main__":
    app.run(debug=True)