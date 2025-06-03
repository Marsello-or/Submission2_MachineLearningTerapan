<!-- Markdown Cell 1 -->
# Domain Proyek
___

---

<!-- Markdown Cell 2 -->
Pada proyek ini, saya berfokus pada pengembangan model sistem rekomendasi jenis tanaman. Masalah pemilihan jenis tanaman yang tepat adalah isu krusial bagi petani karena penanaman yang tidak sesuai dengan kondisi lingkungan dan tanah yang dapat berdampak signifikan pada hasil panen dan pendapatan. Memahami dan merekomendasikan jenis tanaman yang optimal menjadi sangat penting untuk membantu petani membuat keputusan yang lebih baik dan merancang strategi penanaman yang efektif dan proaktif. Dengan mengidentifikasi tanaman yang paling cocok sejak dini, petani dapat meningkatkan produktivitas dan mengurangi risiko kerugian.

Masalah petani dalam memilih tanaman harus diselesaikan karena memiliki dampak finansial yang besar bagi petani dan ketahanan pangan secara keseluruhan. Ketidaksesuaian tanaman dengan kondisi lahan dapat mengurangi produktivitas, menurunkan pendapatan, serta meningkatkan biaya operasional akibat kegagalan panen atau kebutuhan perlakuan tambahan. Oleh karena itu, kemampuan untuk merekomendasikan tanaman yang tepat adalah kunci untuk keberlanjutan dan profitabilitas usaha pertanian.

Penyelesaian masalah ini dilakukan melalui pendekatan model machine learning. Dengan menganalisis data karakteristik lingkungan dan tanah, model machine learning dapat belajar pola-pola yang mengindikasikan kecocokan suatu jenis tanaman. Model ini akan memberikan kemampuan untuk :
- Rekomendasi optimal (Rekomendasi jenis tanaman yang paling optimal berdasarkan kondisi spesifik lahan)
- Peningkatan produktivitas (Membantu petani dalam memilih tanaman yang memiliki potensi hasil panen tertinggi di lahan mereka)
- Optimalisasi sumber daya (meminimalkan penggunaan sumber daya yang tidak perlu akibat penanaman yang tidak sesuai)

Riset menunjukkan bahwa sistem machine learning yang diusulkan dengan memanfaatkan data historis terkait kondisi iklim, sifat tanah, hasil panen, dan preferensi petani dapat memberikan rekomendasi tanaman yang dipersonalisasi. Studi ini mengevaluasi sembilan model Machine Learning, termasuk Logistic Regression, SVM, KNN, Decision Tree, Random Forest, Bagging, AdaBoost, Gradient Boosting, dan Extra Trees dengan Random Forest menunjukkan akurasi tertinggi yaitu sebesar 99,31% (Prity dkk., 2024).

Referensi :

Prity, F.S., Hasan, M.M., Saif, S.H. et al. Enhancing Agricultural Productivity: A Machine Learning Approach to Crop Recommendations. Hum-Cent Intell Syst 4, 497–510 (2024). https://doi.org/10.1007/s44230-024-00081-3

[(Sumber Referensi)](https://doi.org/10.1007/s44230-024-00081-3)


---

<!-- Markdown Cell 3 -->
## Business Understanding
---
Pada bagian ini, saya akan menjelaskan proses klarifikasi masalah, termasuk pernyataan masalah, tujuan, dan solusi yang diusulkan.

---

<!-- Markdown Cell 4 -->
### Problem Statement
---

---

<!-- Markdown Cell 5 -->
1. Bagaimana petani dapat secara efektif mencocokkan kebutuhan spesifik tanaman dengan kondisi lingkungan (N, P, K, suhu, kelembaban, pH, dan curah hujan) mereka, mengingat kurangnya metode atau alat yang terstruktur untuk melakukan analisis komparatif ini?
2. Bagaimana dampak negatif dari pemilihan tanaman yang tidak optimal (seperti pemborosan sumber daya dan potensi penurunan hasil panen) dapat dikurangi atau dihindari oleh petani?

---

<!-- Markdown Cell 6 -->
### Goals
---

---

<!-- Markdown Cell 7 -->
1. Membangun sistem yang menyediakan metode terstruktur dan berbasis data untuk mencocokkan kebutuhan spesifik tanaman dengan parameter lingkungan, sehingga memudahkan petani dalam mengidentifikasi tanaman yang paling sesuai.
2. Menyediakan rekomendasi tanaman yang akurat dan relevan untuk memandu petani, yang pada akhirnya dapat mengurangi pemborosan sumber daya dan meningkatkan potensi hasil panen melalui pemilihan tanaman yang optimal.

---

<!-- Markdown Cell 8 -->
### Solution Statements
---

---

<!-- Markdown Cell 9 -->
1. Membangun model rekomendasi berbasis konten untuk menyarankan jenis tanaman yang sesuai.
2. Melakukan analisis fitur dari dataset untuk mengetahui fitur-fitur lingkungan mana yang paling signifikan dalam menentukan rekomendasi tanaman, serta menghitung rata-rata karakteristik untuk setiap jenis tanaman.
3. Mengembangkan algoritma rekomendasi yang menggunakan metrik kemiripan (misalnya, cosine similarity) antara kondisi lingkungan pengguna dan karakteristik rata-rata tanaman untuk menghasilkan daftar rekomendasi.
4. Mengevaluasi performa model rekomendasi menggunakan metrik yang relevan seperti Precision@k, Recall@k, dan NDCG (Normalized Discounted Cumulative Gain) untuk mengukur seberapa efektif sistem dalam menyajikan rekomendasi yang akurat dan relevan.

---

<!-- Markdown Cell 10 -->
## Data Understanding
---

---

<!-- Markdown Cell 11 -->
Dataset "Crop Recommendation" berisi informasi historis parameter lingkungan dan tanah yang terkait dengan jenis tanaman yang cocok untuk kondisi tersebut. Sumber data didapatkan dari tautan berikut: [(Sumber Dataset)](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset). Dataset ini mencakup berbagai atribut seperti kadar Nitrogen (N), Fosfor (P), Kalium (K) dalam tanah, suhu (temperature), kelembaban (humidity), pH tanah (ph), dan curah hujan (rainfall). Target variabelnya adalah 'label' (jenis tanaman yang direkomendasikan).

Namun karena model yang akan dibangun adalah model Content Based Filtering, maka dataset yang digunakan harus diubah terlebih dahulu menjadi profil fitur untuk setiap jenis tanaman (misalnya berdasarkan kebutuhan nutrisi N, P, K, suhu optimal, dll.) dan kemudian merekomendasikan tanaman yang paling mirip profilnya dengan kondisi input atau tanaman referensi.

---

<!-- Markdown Cell 12 -->
# Data Loading
---

---

<!-- Markdown Cell 13 -->
## Import Library
___

---

<!-- Markdown Cell 14 -->
---
Data sudah diubah ke dalam format profil fitur untuk setiap jenis tanaman (misalnya berdasarkan kebutuhan nutrisi N, P, K, suhu optimal, dll.) agar kemudian dapat membangun model rekomendasi tanaman yang paling mirip profilnya dengan kondisi input atau tanaman referensi.

---

<!-- Markdown Cell 15 -->
# Exploratory Data Analysis
___
Proses analisis data yang ada di dalam dataset, proses eksplorasi dilakukan untuk dapat melihat persebaran dan karakteristik data yang akan digunakan untuk dasar pembuatan model.

---

<!-- Markdown Cell 16 -->
---
Output di atas menunjukkan bahwa dataset memiliki 22 data profil tanaman dan 8 kolom fitur untuk masing-masing profil tanaman.
- Terdapat 7 data float 64.
- Terdapat 1 data object (label jenis tanaman).

---

<!-- Markdown Cell 17 -->
---
Dilakukan identifikasi karakeristik secara statistik dengan menggunakan fungsi `describe`.
- `count` adalah jumlah sampel pada dataset yang digunakan
- `mean` adalah nilai rata-rata
- `std` adalah standar deviasi
- `min` adalah nilai minimum yang ada pada masing-masing kolom data
- `25%` menunjukkan kuartil pertama
- `50%` menunjukkan kuartil kedua
- `75%` menunjukkan kuartil ketiga
- `Max` adalah nilai maksimum pada kolom

* `N` (Nitrogen): Rasio kandungan nitrogen antara 18.77-117.77.
* `P` (Phosphorus): Rasio kandungan Phosphorus antara 16.55-134.22.
* `K` (Potassium): Rasio kandungan Potassium antara 10.01-200.11.
* `temperature` (Suhu): Berkisar antara 18.87-33.72 C.
* `humidity` (Kelembaban): Berkisar antara 16.86-94.84 %.
* `ph` (pH Tanah): Berkisar antara 5.74-7.33.
* `rainfall` (Curah Hujan): Berkisar antara 24.68-236.181 mm.

---

<!-- Markdown Cell 18 -->
### Checking Missing and Duplicated Value
---

---

<!-- Markdown Cell 19 -->
---
Terlihat dari fungsi analisis di atas bahwa dataset tidak memiliki data duplikat dan *missing value*. Oleh karena itu, proses dapat dilanjutkan kepada analisis dan proses visualisasi data.


---

<!-- Markdown Cell 20 -->
### Boxplot Visualization

Visualisasi boxplot dari data numerical yang anda
___

---

<!-- Markdown Cell 21 -->
Untuk deskripsi dan detail box plot dapat dilihat sebagai berikut:
-   `N` (Nitrogen): Distribusi yang relatif tersebar dengan rentang data dari 20-80 unit kandungan Nitrogen, penyebarannya cukup lebar.
-   `P` (Phosphorus): Menunjukkan distribusi yang cukup merata dengan sedikit outlier, data terbentang dari 30-70 unit kandungan Phosphorus.
-   `K` (Potassium): Terlihat konsentrasi nilai pada 20-50 unit kandungan Potassium, dengan sedikit outlier di bagian atas.
-   `temperature`: Distribusi suhu yang cukup simetris dengan rentang 23-28 C.
-   `humidity`: Terlihat ada outlier pada kelembaban yang sangat rendah, dengan rentang data 60-90%.
-   `ph`: Distribusi pH tanah yang cenderung simetris pada nilai pH normal.
-   `rainfall`: Menunjukkan kesimetrisan dengan data outlier di bagian atas dengan kecerendungan right skewness.

---

<!-- Markdown Cell 22 -->
## Exploratory Data Analysis - Univariate
---

---

<!-- Markdown Cell 23 -->
___
Berdasarkan hasil analisis dari plot distribusi *feature numerical*:
-   `N`, `P`, `K`, `temperature`, `humidity`, `ph`, `rainfall`: Sebagian besar fitur numerik menunjukkan distribusi yang bervariasi, beberapa mendekati normal, beberapa sedikit miring. Ini memberikan gambaran umum tentang rentang dan konsentrasi nilai pada setiap fitur.
___

---

<!-- Markdown Cell 24 -->
### Univariate Analysis - Categorical Feature

---

<!-- Markdown Cell 25 -->
---
Berdasarkan grafik fitur kategorikal yang telah dibuat, berikut adalah deskripsinya:

**Label Distribution (Distribusi Jenis Tanaman):**
- Grafik ini menunjukkan jumlah sampel untuk setiap jenis tanaman. Terlihat bahwa setiap jenis tanaman (label) memiliki jumlah sampel yang seimbang (masing-masing 1 sampel profil untuk setiap tanaman), mengindikasikan dataset yang seimbang untuk setiap tanamannya dan tidak ada duplikat.

---

<!-- Markdown Cell 26 -->
# EDA - Multivariate Analysis
---

---

<!-- Markdown Cell 27 -->
## Analisis Pair Plot dari setiap fitur numerik profil tanaman
---

---

<!-- Markdown Cell 28 -->
### Analisis Hasil Pair Plots (Multivariate Analysis)

---

<!-- Markdown Cell 29 -->
---
Berdasarkan *pair plot* yang menunjukkan hubungan antara fitur numerik:

- N, P, dan K (unsur hara tanah): Distribusi tidak normal; terlihat multimodal atau skewed (condong ke kiri/kanan).
- Banyak nilai ekstrem atau outlier, terutama pada K.
- Temperature: Distribusi mendekati normal dengan sedikit variasi; sebagian besar data berada pada rentang 20–30°C.
- Humidity: Distribusi cukup terkonsentrasi antara 40%–80%.
- pH: Nilainya terkonsentrasi antara 5.5–7.5, menunjukkan kebanyakan tanah bersifat netral hingga sedikit asam.
- Rainfall: Tersebar lebih merata dibanding variabel lain, dengan beberapa
outlier di atas 150 mm.

Hubungan antar Variabel:
 - N, P, dan K satu sama lain : Tidak ada korelasi linier yang kuat terlihat; penyebaran data tampak acak.
 - Temperature vs. Humidity: Tidak menunjukkan korelasi negatif kuat seperti yang biasanya diharapkan (di mana suhu tinggi → kelembapan turun), artinya faktor lain bisa lebih dominan.
 - pH vs Nutrisi (N, P, K):Hubungan antara pH dan N, P, K juga tidak tampak jelas secara linier, kemungkinan hubungan non-linier atau ada pengaruh dari jenis tanaman/jenis tanah.
 - Rainfall: Tidak menunjukkan korelasi kuat terhadap variabel lain secara visual.

---

<!-- Markdown Cell 30 -->
## Correlation Heatmap
---

---

<!-- Markdown Cell 31 -->
### Interpretasi Umum Heatmap:
---

---

<!-- Markdown Cell 32 -->
* **Warna Merah Cerah:** Menunjukkan korelasi positif yang kuat (mendekati +1).
* **Warna Biru Cerah:** Menunjukkan korelasi negatif yang kuat (mendekati -1).
* **Warna Pucat/Putih (mendekati 0):** Menunjukkan korelasi yang sangat lemah atau tidak ada korelasi linier.

---

<!-- Markdown Cell 33 -->
### Analisis Korelasi Antar Fitur:
---

---

<!-- Markdown Cell 34 -->
### Analisis Heatmap Korelasi Variabel Numerik
- Korelasi Positif Kuat
  - Fosfor (P) dan Kalium (K): 0,76 - Ini adalah korelasi positif terkuat dalam dataset, menunjukkan bahwa kedua unsur hara ini sering muncul bersamaan dalam tanah, kemungkinan karena sumber yang sama atau proses tanah yang mempengaruhi kedua elemen ini.
  - Suhu dan Kelembaban: 0,30 - Hubungan positif yang moderat, yang mungkin tampak berlawanan dengan intuisi tetapi bisa mencerminkan pola iklim regional dimana daerah yang lebih hangat juga memiliki tingkat kelembaban yang lebih tinggi.

- Korelasi Negatif Kuat
  - Nitrogen (N) dan Fosfor (P): -0,25 - Hubungan negatif yang notable, menunjukkan bahwa unsur hara ini mungkin berkompetisi atau dipengaruhi oleh kondisi tanah yang berlawanan.
  - Kalium (K) dan pH: -0,28 - Menunjukkan bahwa kadar kalium yang tinggi cenderung terjadi pada tanah yang lebih asam (pH rendah).
  - Fosfor (P) dan pH: -0,23 - Pola serupa dengan kalium, menunjukkan kondisi asam mungkin mendukung ketersediaan fosfor.

### Pengamatan Menarik

- pH menunjukkan korelasi negatif dengan sebagian besar unsur hara (N, P, K), menunjukkan sampel tanah mungkin berasal dari lingkungan dimana kondisi asam mendukung ketersediaan hara
- Curah hujan tampak relatif independen terhadap variabel lainnya, dengan korelasi yang lemah secara keseluruhan
- Hubungan suhu umumnya lemah kecuali dengan kelembaban, menunjukkan mungkin bukan faktor utama yang mempengaruhi sifat tanah yang diukur

Pola ini menunjukkan data mungkin berasal dari tanah pertanian atau hutan dimana dinamika hara sangat dipengaruhi oleh tingkat pH, dengan fosfor dan kalium menunjukkan perilaku yang saling terkait.

---

<!-- Markdown Cell 35 -->
### Kesimpulan dari Heatmap:
---

---

<!-- Markdown Cell 36 -->
* Fosfor dan Kalium sangat terkait (0,76) - Kedua unsur hara ini berperilaku serupa dalam tanah, kemungkinan karena proses biogeokimia yang sama atau sumber yang serupa.
* pH adalah faktor kunci - pH tanah memiliki pengaruh negatif terhadap ketersediaan unsur hara utama (N, P, K), menunjukkan bahwa tanah yang lebih asam cenderung memiliki konsentrasi hara yang lebih tinggi.
* Nitrogen berperilaku berbeda - Nitrogen menunjukkan pola korelasi yang berbeda dibanding P dan K, terutama hubungan negatif dengan fosfor, mengindikasikan dinamika yang kompleks antar unsur hara.

---

<!-- Markdown Cell 37 -->
# Data Preparation
Teknik yang digunakan:
-   Penanganan Missing Values: Memastikan tidak ada nilai yang hilang.
-   Pemisahan Fitur dan Target: Membagi data menjadi variabel independen (fitur) dan dependen (target).
-   Train-test split data: Data dibagi menjadi 80% Train dan 20% Test.
-   Feature Scaling: Melakukan penskalaan data numerik.
---

---

<!-- Markdown Cell 38 -->
### Penanganan Missing Values
---

---

<!-- Markdown Cell 39 -->
---
Pada dataset ini, tidak ditemukan adanya *missing value*, sehingga tidak diperlukan penanganan khusus pada tahap ini.

---

<!-- Markdown Cell 40 -->
# Data Modeling
---

---

<!-- Markdown Cell 41 -->
## Feature Engineering
---

---

<!-- Markdown Cell 42 -->
### Define feature names

---

<!-- Markdown Cell 43 -->
### Initialize encoders and scalers

---

<!-- Markdown Cell 44 -->
### Encode crop Labels untuk jadi bentuk numerik

---

<!-- Markdown Cell 45 -->
### Scale features to normalize different units (N, P, K vs temperature, etc.)
---

---

<!-- Markdown Cell 46 -->
# Modeling

---

<!-- Markdown Cell 47 -->
# Konsep dan Kelebihan/Kekurangan Neural Network dalam Sistem Rekomendasi
Neural Network menawarkan pendekatan yang sangat kuat dan fleksibel untuk sistem rekomendasi, terutama dalam skenario yang lebih kompleks.

## Apa itu Neural Network?
Neural Network adalah model komputasi yang terinspirasi oleh struktur dan fungsi otak manusia. Model ini terdiri dari lapisan-lapisan (layers) node (neuron) yang saling terhubung. Setiap neuron menerima input, memprosesnya dengan fungsi aktivasi, dan mengirimkan output ke neuron di lapisan berikutnya. Kemampuan Neural Network untuk belajar pola non-linier dari data menjadikannya sangat efektif untuk tugas-tugas kompleks seperti sistem rekomendasi. Lapisan-lapisan yang umum ditemukan seperti Embedding (untuk mengubah ID menjadi vektor padat), Dot (untuk menghitung kemiripan antar vektor), dan Dense (untuk mempelajari pola kompleks dan melakukan transformasi data) adalah bagian fundamental dari arsitektur ini.

## Kelebihan Neural Network dalam Sistem Rekomendasi:

- Kemampuan Belajar Pola Kompleks Non-Linier: Neural Network dapat menangkap hubungan yang sangat kompleks dan non-linier dalam data interaksi pengguna-item yang mungkin tidak dapat ditangkap oleh metode linier atau berbasis kemiripan sederhana. Ini memungkinkan model untuk menemukan preferensi tersembunyi yang lebih dalam.
- Pembuatan Representasi Fitur Otomatis (Feature Learning): Dengan lapisan embedding dan dense, Neural Network dapat secara otomatis belajar representasi (embeddings) yang kaya dan bermakna dari pengguna dan item dari data mentah. Hal ini mengurangi kebutuhan akan rekayasa fitur manual yang ekstensif.
- Menangani Data Sparse: Model berbasis Neural Network, terutama yang menggunakan embedding, dapat bekerja dengan baik pada dataset yang sangat sparse (banyak nilai kosong) seperti yang umum dijumpai dalam data interaksi pengguna-item.
- Fleksibilitas Arsitektur: Arsitektur Neural Network dapat dirancang dengan sangat fleksibel (jumlah lapisan, jumlah neuron per lapisan, jenis aktivasi) untuk menyesuaikan dengan kompleksitas data dan tujuan rekomendasi.
- Potensi untuk Menggabungkan Berbagai Sumber Data (Hybrid Systems): Neural Network sangat cocok untuk membangun sistem rekomendasi hibrida, di mana informasi dari data konten (fitur item), data pengguna, dan data interaksi (Collaborative Filtering) dapat digabungkan dan diproses secara bersamaan.

## Kekurangan Neural Network dalam Sistem Rekomendasi:

- Membutuhkan Data yang Besar: Neural Network, terutama yang dalam (deep learning), membutuhkan jumlah data yang sangat besar untuk dilatih secara efektif dan menghindari overfitting.
- Komputasi Intensif: Pelatihan model Neural Network bisa sangat intensif secara komputasi dan memakan waktu lama, seringkali membutuhkan hardware yang kuat (seperti GPU/TPU).
- Interpretasi yang Sulit (Black Box): Sulit untuk memahami secara pasti mengapa Neural Network membuat rekomendasi tertentu, karena proses internalnya seringkali tidak transparan (black box), sehingga menyulitkan debugging atau penjelasan keputusan.
- Desain Arsitektur yang Kompleks: Merancang arsitektur Neural Network yang optimal (memilih jumlah lapisan, neuron, fungsi aktivasi, dll.) bisa menjadi tugas yang kompleks dan membutuhkan banyak eksperimen serta pemahaman mendalam.
- Risiko Overfitting: Jika tidak ditangani dengan benar (misalnya dengan teknik regularisasi, dropout, atau data yang cukup), Neural Network rentan terhadap overfitting, di mana model bekerja sangat baik pada data pelatihan tetapi buruk pada data baru.

---

<!-- Markdown Cell 48 -->
# Kelebihan dan Kekurangan Cosine Similarity (dalam Konteks Content-Based Filtering)
Model rekomendasi tanaman ini dibangun menggunakan pendekatan Content-Based Filtering, dengan Cosine Similarity sebagai metrik utama untuk mengukur kemiripan antar kondisi lingkungan input dan data tanaman. Metode ini sangat cocok untuk data numerik yang mewakili fitur-fitur penting.

## Apa itu Cosine Similarity?
Cosine Similarity adalah ukuran kemiripan antara dua vektor non-nol di ruang hasil kali dalam (inner product space) yang mengukur cosinus dari sudut di antara mereka. Nilai cosinus 0 derajat adalah 1, dan untuk sudut 90 derajat adalah 0. Ini berarti nilai kemiripan berkisar antara -1 (berlawanan arah) hingga 1 (arah yang sama), dengan 0 menunjukkan ketidak-terkaitan. Dalam konteks ini, semakin tinggi nilai Cosine Similarity, semakin mirip profil lingkungan input dengan kebutuhan suatu tanaman.

## Kelebihan Cosine Similarity:

- Efektivitas dalam Ruang Dimensi Tinggi: Cosine Similarity sangat efektif dalam mengukur kemiripan antar item di ruang fitur berdimensi tinggi. Ini sangat relevan untuk dataset seperti rekomendasi tanaman yang memiliki banyak parameter lingkungan.
- Tidak Terpengaruh Skala Magnitudo: Metode ini hanya berfokus pada orientasi (arah) vektor, bukan magnitudenya. Artinya, perbedaan absolut dalam nilai fitur tidak terlalu memengaruhi hasil kemiripan, sehingga cocok untuk fitur yang memiliki rentang nilai bervariasi setelah normalisasi. Dua item yang memiliki proporsi fitur yang sama tetapi nilai absolut yang berbeda jauh (misalnya, kondisi NPK yang sangat tinggi vs. sangat rendah, tetapi proporsinya sama) masih akan dianggap mirip.
- Kesederhanaan dan Interpretasi: Konsepnya relatif sederhana dan mudah diinterpretasikan. Nilai kemiripan berkisar antara -1 (berlawanan) hingga 1 (sangat mirip), dengan 0 menunjukkan tidak ada kemiripan, memudahkan pemahaman tentang seberapa dekat dua entitas.
- Mudah Diimplementasikan: Algoritma Cosine Similarity cukup mudah diimplementasikan dengan library standar seperti scikit-learn, menjadikannya pilihan praktis untuk banyak aplikasi.

## Kekurangan Cosine Similarity:

- Tidak Mempertimbangkan Magnitudo: Meskipun menjadi kelebihan, ini juga bisa menjadi kekurangan. Cosine Similarity tidak mempertimbangkan perbedaan absolut antar nilai. Dua vektor yang memiliki arah yang sama tetapi nilai absolut yang sangat berbeda akan memiliki skor kemiripan 1. Dalam konteks rekomendasi tanaman, ini mungkin berarti dua kondisi lingkungan yang "mirip" secara proporsional tetapi sangat berbeda dalam skala (misalnya, sangat kering vs. sangat basah, meskipun proporsi NPK sama) akan dianggap sangat mirip, padahal mungkin perbedaan magnitudenya krusial.
- Sensitif terhadap Data Sparsity (jika fitur banyak yang kosong): Jika dataset memiliki banyak nilai nol atau fitur yang kosong, Cosine Similarity bisa menghasilkan hasil yang misleading karena hanya mempertimbangkan fitur yang ada. Namun, dalam kasus data lingkungan tanaman yang biasanya padat fitur, ini kurang menjadi masalah.
- Keterbatasan pada Fitur Numerik: Cosine Similarity paling cocok untuk fitur numerik. Jika ada fitur kategorikal yang tidak dapat diubah menjadi representasi numerik yang bermakna (misalnya, melalui one-hot encoding), penggunaannya mungkin terbatas atau memerlukan pra-pemrosesan yang lebih kompleks.

---

<!-- Markdown Cell 49 -->
### Data Splitting untuk Training Model dan pairing jenis tanaman dengan fitur profilnya masing-masing
---

---

<!-- Markdown Cell 50 -->
Setelah pembuatan layer model dilakukan train dan validation data splitting dengan ratsio 80:20. Untuk X akan menyimpan feature yang akan dipakai dan y akan menyimpan label profil tanaman

---

<!-- Markdown Cell 51 -->
## NEURAL NETWORK ARCHITECTURE SECTION
Purpose: Build the neural network model with embeddings and dense layers

---

<!-- Markdown Cell 52 -->
Hyperparameter Model

---
Pemilihan hyperparameter Neural Network seperti embedding_dim, hidden_units, n_crops, dan n_features merupakan langkah penting dalam mendefinisikan arsitektur model. n_crops dan n_features secara langsung merefleksikan dimensi data input (jumlah jenis tanaman unik dan jumlah fitur lingkungan), yang esensial untuk ukuran lapisan input dan embedding. Sementara itu, embedding_dim (32) menentukan kompleksitas representasi vektor padat untuk setiap tanaman, dipilih sebagai nilai yang umum untuk menyeimbangkan kemampuan belajar model dengan efisiensi komputasi, dan hidden_units ([64, 32, 16]) mendefinisikan kedalaman serta lebar lapisan tersembunyi. Ukuran yang menurun ini memungkinkan model untuk secara progresif belajar pola yang lebih abstrak dan ringkas, dengan setiap nilai spesifik ini umumnya ditentukan melalui eksperimen dan praktik terbaik untuk mencapai kinerja optimal pada data yang ada.

---

<!-- Markdown Cell 53 -->
Pembuatan Layer Model

---

<!-- Markdown Cell 54 -->
### Model Training
---

---

<!-- Markdown Cell 55 -->
# Pembuatan Fungsi Model Rekomendasi antara Neural Network dan juga Cosine Similarity
---

---

<!-- Markdown Cell 56 -->
# Testing and Comparison Section
Memeriksa dan membandingkan kedua model yang telah dibuat baik menggunakan deep learning dan cosine similarity

---


---

<!-- Markdown Cell 57 -->
## Model Analysis Section

---

<!-- Markdown Cell 58 -->
## Perbandingan dan Kesimpulan:

Secara nilai kemiripan langsung dari fitur, Simple Cosine Similarity menemukan kecocokan yang lebih tinggi (COTTON dengan 0.900) dibandingkan dengan "Combined Score" teratas dari Neural Network (WATERMELON dengan 0.765). Ini menunjukkan bahwa Cosine Similarity sangat efisien dalam mencocokkan profil lingkungan yang paling serupa secara langsung.
Neural Network cenderung melihat pola yang lebih kompleks. Meskipun "Combined Score"nya lebih rendah dari top score Cosine Similarity, "Neural Score" yang tinggi untuk WATERMELON (0.922) mengindikasikan bahwa model saraf telah belajar alasan internal mengapa WATERMELON dianggap cocok, yang mungkin melampaui kemiripan fitur langsung yang diukur oleh Cosine Similarity. Perbedaan dalam daftar rekomendasi menunjukkan bahwa kedua model memiliki "perspektif" yang berbeda tentang "kecocokan".

---

<!-- Markdown Cell 59 -->
# Model Inference

---

<!-- Markdown Cell 60 -->
Bagian inference model untuk mencoba model yang lebih simpel dan terbukti lebih akurat dalam memprediksi profil tanaman yang cocok dengan karakteristik yang dimasukkan oleh pengguna

---

<!-- Markdown Cell 61 -->
# Kesimpulan
---

---

<!-- Markdown Cell 62 -->
Proyek ini telah berhasil mengembangkan sistem rekomendasi tanaman berbasis data yang secara efektif menjawab masalah krusial petani dalam memilih jenis tanaman yang tepat sesuai kondisi lingkungan, guna menghindari dampak negatif pada hasil panen dan pendapatan. Dengan mengimplementasikan Content-Based Filtering menggunakan Cosine Similarity, model ini mampu menyediakan panduan akurat dengan mengidentifikasi kemiripan langsung antara profil lingkungan input dan karakteristik tanaman, terbukti efisien dalam menemukan kecocokan yang relevan. Meskipun terdapat eksplorasi terhadap kapabilitas Neural Network untuk mempelajari pola yang lebih kompleks dan abstrak, fungsi rekomendasi inti berhasil memberikan rekomendasi yang spesifik, menunjukkan bahwa sistem ini mampu membantu petani membuat keputusan penanaman yang lebih terinformasi, pada akhirnya meningkatkan produktivitas dan mengurangi risiko kerugian yang disebabkan oleh ketidaksesuaian tanaman dengan lahan.

---

