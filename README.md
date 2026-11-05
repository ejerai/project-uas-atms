# Bansos Insight - Dashboard Analisis & Klasterisasi Sentimen

Bansos Insight adalah sebuah proyek analisis sentimen, klasterisasi, dan visualisasi data terkait topik Bantuan Sosial (Bansos) di media sosial. Machine Learning serta Natural Language Processing (NLP) untuk menggali *insight* dari opini publik.

<img width="1920" height="911" alt="Screenshot (322)" src="https://github.com/user-attachments/assets/62e23210-37e4-4bb0-9151-953ba4879e7a" />


## Deskripsi Proyek
Proyek ini mengumpulkan data teks dari media sosial (Twitter/X) mengenai topik bantuan sosial (Bansos), kemudian melakukan pemrosesan teks, klasifikasi sentimen menggunakan kamus bahasa Indonesia (InSet Lexicon), pemodelan klasterisasi (K-Means & TF-IDF), dan akhirnya disajikan ke dalam **Dashboard Interaktif berbasis Streamlit** dengan desain UI yang modern.

## Fitur Utama
1. **Analisis Sentimen (Lexicon-based)**: Mengklasifikasikan opini masyarakat menjadi sentimen Positif, Negatif, atau Netral menggunakan InSet Lexicon.
2. **Klasterisasi Topik (K-Means & TF-IDF)**: Mengelompokkan tweet ke dalam berbagai klaster pembicaraan (misalnya: Keluhan Publik, Kebijakan Bansos, dll) menggunakan teknik K-Means Clustering yang direduksi dimensinya menggunakan TruncatedSVD.
3. **Deteksi Trending & Keyword**: Menggali kata kunci utama yang sering dibicarakan dalam topik Bansos.
4. **Dashboard Streamlit Interaktif**: Antarmuka responsif dengan desain dark-theme, visualisasi grafik interaktif (Plotly), indikator KPI, dan filter dinamis.

## Tech Stack
- **Bahasa Pemrograman**: Python 3
- **Data Processing & ML**: Pandas, NumPy, Scikit-Learn (TF-IDF, SVD, KMeans)
- **NLP / Sentiment Analysis**: InSet Lexicon (Indonesian Sentiment Lexicon)
- **Visualisasi & Dashboard**: Streamlit, Plotly (Graph Objects & Express)

## Struktur File
- `app.py`: Source code utama untuk menjalankan Dashboard Streamlit dengan redesign dashboard custom UI.
- `PROJEK_UAS_Bansos_RunAll_AutoStreamlit_CLUSTER3.ipynb`: Jupyter Notebook yang berisi seluruh *pipeline* data science, mulai dari crawling, preprocessing, sentiment, klasterisasi, hingga analisis.
- `bansos.csv`: Dataset mentah/raw hasil crawling dari Twitter/X.
- `InSet_pos.tsv` & `InSet_neg.tsv`: Kamus kata berbobot sentimen positif dan negatif untuk Bahasa Indonesia.

## Instalasi & Menjalankan Dashboard

1. **Clone Repository** (atau unduh semua file ke dalam satu folder):
   ```bash
   git clone <link-repo-anda>
   cd <nama-folder-repo>
   ```

2. **Install Dependensi**:
   Pastikan Anda telah menginstal pustaka Python yang dibutuhkan.
   ```bash
   pip install pandas numpy scikit-learn plotly streamlit
   ```

3. **Jalankan Aplikasi Streamlit**:
   Jalankan perintah berikut di terminal/command prompt:
   ```bash
   streamlit run app.py
   ```

4. **Akses Dashboard**:
   Buka browser Anda dan akses URL lokal yang diberikan oleh Streamlit (biasanya `http://localhost:8501`).
