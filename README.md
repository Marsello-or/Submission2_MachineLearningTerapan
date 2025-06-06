# Domain Proyek

## Book Recomenndation System
:::

::: {.cell .markdown id="mjwknvbv92ZH"}
Pada proyek ini, saya berfokus pada pengembangan model sistem
rekomendasi buku. Di era digital saat ini, jumlah buku yang tersedia
bagi pembaca sangatlah masif, baik dalam format fisik maupun digital.
Masalah dalam memilih buku yang tepat menjadi isu krusial bagi para
pembaca, karena pemilihan yang tidak sesuai dengan selera dan minat
dapat berdampak pada penurunan minat baca, serta pemborosan waktu dan
biaya. Memahami dan merekomendasikan buku yang relevan menjadi sangat
penting untuk membantu pembaca menemukan bacaan yang memuaskan dan
memperkaya pengalaman literasi mereka. Dengan mengidentifikasi buku yang
paling cocok sejak dini, pembaca dapat memaksimalkan waktu luang mereka
dan menumbuhkan kecintaan yang lebih dalam terhadap dunia literasi.

Penyelesaian masalah ini dilakukan melalui pendekatan model machine
learning. Dengan menganalisis data metadata buku (seperti genre,
penulis, dan deskripsi) serta data rating, model machine learning dapat
mempelajari pola-pola yang mengindikasikan kecocokan suatu buku bagi
pembaca. Model ini akan memberikan kemampuan untuk:

-   Rekomendasi Tepat Sasaran: Memberikan rekomendasi buku yang paling
    sesuai berdasarkan preferensi dan riwayat baca pengguna.
-   Peningkatan Pengalaman Membaca: Membantu pembaca menemukan buku yang
    memiliki potensi untuk memberikan kepuasan dan keterlibatan
    emosional tertinggi.
-   Efisiensi Waktu dan Biaya: Meminimalkan risiko pemborosan waktu dan
    uang pada buku yang tidak sesuai selera, sehingga pembaca dapat
    memaksimalkan sumber dayanya untuk bacaan yang lebih berharga.

Riset menunjukkan bahwa sistem machine learning yang diusulkan dengan
memanfaatkan model rekomendasi dapat dengan mudah membantu pembaca atau
pengguna yang memiliki hobi membaca dalam menemukan buku yang sesuai
dengan minatnya, membantu perusahaan retail buku dalam meningkatkan
revenue, dan juga membantu proses administasi perpustakaan dengan lebih
cepat menghadirkan rekomendasi yang sesuai dengan minat penggunanya.

Referensi :

, K., Dr, S., & Scholar, K. (2023). Study on Book Recommendation System.
2023 Advanced Computing and Communication Technologies for High
Performance Applications (ACCTHPA), 1-8.
<https://doi.org/10.1109/ACCTHPA57160.2023.10083372>.

[(Sumber
Referensi)](https://doi.org/10.1109/ACCTHPA57160.2023.10083372.)
:::

::: {.cell .markdown id="VOD8ou1i939A"}
## \## Business Understanding {#-business-understanding}
:::

::: {.cell .markdown id="bFiDyDLE98A4"}
Pada bagian ini, saya akan menjelaskan proses klarifikasi masalah,
termasuk pernyataan masalah, tujuan, dan solusi yang diusulkan.
:::

::: {.cell .markdown id="No9SITsq-E9y"}
## \## Problem Statement {#-problem-statement}
:::

::: {.cell .markdown id="eNu_VLon-H-4"}
1.  Kesulitan dalam Penemuan Buku yang Relevan: Di tengah melimpahnya
    pilihan buku, pembaca sering kali menghadapi kesulitan untuk
    menemukan judul yang benar-benar sesuai dengan minat mereka melalui
    metode pencarian konvensional yang tidak dipersonalisasi.
2.  Tantangan dalam Memenuhi Selera yang Subjektif: Preferensi baca
    setiap individu sangat unik dan berbeda-beda. Rekomendasi umum atau
    daftar bestseller seringkali gagal memenuhi selera personal ini,
    sehingga pengalaman membaca menjadi kurang memuaskan.
3.  Risiko Pemborosan Waktu dan Penurunan Minat Baca: Kesalahan dalam
    memilih buku tidak hanya membuang waktu dan biaya, tetapi juga dapat
    menyebabkan kekecewaan yang berujung pada menurunnya motivasi dan
    minat seseorang untuk melanjutkan kebiasaan membaca.
:::

::: {.cell .markdown id="Xa4o7hAF-J0g"}
## \## Goals {#-goals}
:::

::: {.cell .markdown id="00Vs_09j-LmG"}
Berdasarkan permasalahan tersebut, proyek ini bertujuan untuk membangun
sistem rekomendasi buku dengan dua pendekatan model (Content-Based dan
Rating-Based) untuk mencapai tujuan-tujuan berikut:

1.  Mempermudah Pengguna dalam Menemukan Buku yang Relevan:
    Mengembangkan model yang dapat menyaring jutaan pilihan buku dan
    menyajikan daftar yang paling relevan secara efisien, sehingga
    membantu pengguna menemukan buku baru yang sesuai.
2.  Meningkatkan Personalisasi dan Kepuasan Membaca: Menyediakan
    rekomendasi yang secara spesifik ditargetkan untuk selera unik
    setiap pembaca, dengan tujuan meningkatkan kepuasan dan keterlibatan
    emosional mereka terhadap buku yang dibaca.
3.  Mengoptimalkan Investasi Waktu Pembaca dan Mendorong Minat Baca:
    Mengurangi kemungkinan pengguna menghabiskan waktu pada buku yang
    tidak tepat, sehingga menjaga agar minat baca mereka tetap tinggi
    dan berkelanjutan.
:::

::: {.cell .markdown id="u_7dyD_n-PpM"}
## \## Solution Statements {#-solution-statements}
:::

::: {.cell .markdown id="wBL-DYEl-RVN"}
1.  Rekomendasi Tepat Sasaran: Memberikan rekomendasi buku yang paling
    relevan dan sesuai dengan preferensi unik setiap pembaca, sehingga
    mempermudah proses penemuan.
2.  Peningkatan Pengalaman Membaca: Membantu pembaca menemukan buku yang
    memiliki potensi tertinggi untuk memberikan kepuasan dan
    keterlibatan emosional, sesuai dengan selera subjektif mereka.
3.  Optimalisasi Waktu dan Minat: Meminimalkan risiko pemborosan waktu
    pada buku yang tidak cocok dan menjaga agar minat baca tetap tinggi
    dengan menyajikan pilihan yang lebih memuaskan.
:::

::: {.cell .markdown id="qR_e6tIUHu6g"}
## \# Data Loading {#-data-loading}
:::

::: {.cell .markdown id="J17cpAIPgTq_"}
## Import Lib yang dibutuhkan

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import warnings
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    warnings.filterwarnings('ignore')
:::

::: {.cell .markdown id="ZEv15Ot5Imnm"}

------------------------------------------------------------------------

Data secara sukses di load dan menunjukkan bahwa semua featurenya secara
lengkap telah terupload dalam dataframe df
:::

::: {.cell .markdown id="0iHZOkWeGIyg"}
## \# Data Understanding {#-data-understanding}
:::

::: {.cell .markdown id="2lgjeYgswuBh"}
## \## Deksripsi Umum Data {#-deksripsi-umum-data}
:::

::: {.cell .markdown id="4uTLOuwEHLGF"}
Dataset \"data.csv\" berisi data karakteristik item dari masing-masing
buku yang terdapat di dataset, beberapa karakteristik item dari dataset
adalah nomor ISBN, judul, subjudul, penulis, kategori, thumbnail,
deskripsi, tahun publish, rata-rata rating, banyak halaman, dan juga
banyaknya rating. Dataset bersumber dari platform Kaggle yang dapat
diakses pada link berikut :

[(Sumber
Dataset)](https://www.kaggle.com/datasets/abdallahwagih/books-dataset).
:::

::: {.cell .markdown id="hN95lSNbJT3K"}

------------------------------------------------------------------------

Terdapat 6810 data buku yang ada pada dataset ini yang masing-masing
terdiri dari 12 data fitur karakteristik di dataframe.
:::

::: {.cell .markdown id="Ry1mBVDpJeJF"}

------------------------------------------------------------------------

Output di atas menunjukkan bahwa dataset memiliki 6810 data profil
tanaman dan 12 kolom fitur masing-masing.

-   Terdapat 4 data float64.
-   Terdapat 7 data object
-   Terdapat 1 data int64
:::

::: {.cell .markdown id="MTLQA8i8Jq9J"}

------------------------------------------------------------------------

Dilakukan identifikasi karakeristik secara statistik dengan menggunakan
fungsi `describe`.

-   `count` adalah jumlah sampel pada dataset yang digunakan

-   `mean` adalah nilai rata-rata

-   `std` adalah standar deviasi

-   `min` adalah nilai minimum yang ada pada masing-masing kolom data

-   `25%` menunjukkan kuartil pertama

-   `50%` menunjukkan kuartil kedua

-   `75%` menunjukkan kuartil ketiga

-   `Max` adalah nilai maksimum pada kolom

-   nomor ISBN13 menunjukkan rentang data nomor identifier dari
    masing-masing buku yang ada.

-   sementara pada fitur published_year terlihat bahwa rentang tahun
    buku di publish dari tahun 1853 sampai 2019.

-   average_rating menunjukkan rating paling kecil pada nilai 0 dan
    rating paling besar pada nilai 5.

-   pada kolom fitur num_pages menunjukkan bahwa jumlah halaman paling
    sedikit yang dimiliki oleh buku adalah 0 halaman dan halaman paling
    banyak sebesar 3342 halaman.

-   sementara pada rating count menunjukkan banyaknya orang yang
    melakukan rating pda buku tersebut dengan nilai paling kecil sebesar
    0 dan nlai paling besar senilai 5.62x10\^6
:::

::: {.cell .markdown id="NvBEgtXi7JwF"}
## \## Pengecekan kondisi data {#-pengecekan-kondisi-data}
:::

::: {.cell .markdown id="CF8RRaB68V0X"}
Berdasarkan analisis data understanding terlihat bahwa data yang
digunakan masih memerlukan proses data cleaning karena adanya missing
data pada beberapa fitur dan juga adanya outlier pada data masing-masing
buku.
:::

::: {.cell .markdown id="8vYfsKHUJ4Zm"}
# Exploratory Data Analysis

------------------------------------------------------------------------

Proses analisis data yang ada di dalam dataset, proses eksplorasi
dilakukan untuk dapat melihat persebaran dan karakteristik data yang
akan digunakan untuk dasar pembuatan model.
:::

::: {.cell .markdown id="UHrlrybEKTfz"}
1.  Fitur Teks: subtitle, authors, categories, description, thumbnail
    Rekomendasi: Lakukan imputasi (pengisian nilai yang hilang) dengan
    string kosong (\'\'). Alasan:

-   subtitle: Sebagian besar buku memang tidak memiliki subtitle. Nilai
    NaN (kosong) di sini bukanlah data yang rusak, melainkan informasi
    bahwa subtitle tidak ada. Menghapus baris ini akan menghilangkan
    lebih dari separuh data Anda. Mengisinya dengan string kosong adalah
    representasi yang paling akurat.
-   authors, categories, description: Fitur-fitur ini sangat penting
    untuk model Content-Based Filtering. Menghapus baris hanya karena
    salah satu dari data ini kosong akan menghilangkan informasi
    berharga dari kolom lain. Mengisinya dengan string kosong memastikan
    fitur ini tetap dapat digunakan untuk analisis teks tanpa
    menimbulkan error.
-   thumbnail: Ini adalah URL gambar. Kehadirannya tidak krusial untuk
    logika model rekomendasi kita. Menghapusnya tidak perlu, cukup diisi
    string kosong agar tipe datanya konsisten.

1.  Fitur Numerik: average_rating, ratings_count, num_pages,
    published_year Rekomendasi: Hapus baris yang memiliki nilai kosong
    pada fitur-fitur ini. Alasan:

-   Jumlahnya Sangat Sedikit: Persentase data yang hilang untuk
    fitur-fitur ini sangat kecil (kurang dari 1%). Menghapus baris-baris
    ini tidak akan mengurangi ukuran dataset secara signifikan.
-   Menjaga Kualitas Data: average_rating dan ratings_count adalah inti
    dari model rekomendasi berbasis rating. Melakukan imputasi (misalnya
    dengan nilai rata-rata atau median) dapat memasukkan bias dan
    menghasilkan rekomendasi yang tidak akurat. Menghapus baris ini
    adalah cara terbersih untuk memastikan kualitas data pada model
    kedua kita.
-   Integritas Data: Sebuah buku tanpa informasi rating, jumlah halaman,
    atau tahun terbit memiliki data yang kurang lengkap. Menghapusnya
    adalah pilihan yang paling aman untuk menjaga integritas analisis.
:::

::: {.cell .markdown id="k1XWLzw6KfCl"}
## \## Boxplot Visualization {#-boxplot-visualization}
:::

::: {.cell .markdown id="_RP-fZ6KKoCG"}
Dari analisis boxplot di atas terlihat bahwa terdapat banyak nilai
outlier yang terdeteksi pada masing-masing fitur yang digunakan dalam
dataset, outlier-outlier ini menunjukkan data yang secara ekstrem berada
di bawah atau atas batas yang telah ditentukan. Namun, dalam handlingnya
bisa jadi terdapat beberapa fitur yang tidak harus dihilangkan nilai
outliernya karena akan mengganggu kebersihan dan keakuratan model yang
akan dibuat.
:::

::: {.cell .markdown id="V9aTpa2OBa5y"}

------------------------------------------------------------------------

-   Biarkan outlier pada fitur average_rating dan published_year karena
    mereka adalah data valid dan akan mengurangi akurasi apabila
    dihapus.
-   Lakukan Log Transformation pada fitur ratings_count dan num_pages
    untuk menormalkan distribusinya tanpa menghilangkan informasi
    berharga dari nilai-nilai ekstrem.
:::

::: {.cell .markdown id="EDbdOSLALwVi"}
## \## EDA - Univariate Analysis {#-eda---univariate-analysis}
:::

::: {.cell .markdown id="rO-ncXHtM8G9"}

------------------------------------------------------------------------

Berdasarkan hasil analisis dari plot distribusi *feature numerical*:

1.  Average Rating: Distribusinya cenderung left-skewed (condong ke
    kiri), yang menunjukkan bahwa sebagian besar buku dalam dataset ini
    memiliki rating yang tinggi (antara 3.5 hingga 4.5). Ini adalah
    pertanda baik, karena menandakan kualitas data rating secara umum
    cukup tinggi.
2.  Ratings Count (Log Transformed): Setelah transformasi log,
    distribusinya menjadi lebih mendekati normal. Ini mengonfirmasi
    bahwa sebagian besar buku memiliki jumlah rating yang moderat,
    sementara hanya sedikit buku yang memiliki jumlah rating sangat
    tinggi (yang merupakan outlier populer).
3.  Jumlah Halaman (Log Transformed): Distribusinya juga terlihat lebih
    normal setelah transformasi. Ini menunjukkan bahwa mayoritas buku
    memiliki ketebalan yang umum (sekitar 200-500 halaman), dengan
    beberapa buku yang sangat tebal atau sangat tipis.
4.  Tahun Terbit: Mayoritas buku dalam dataset ini diterbitkan pada
    akhir abad ke-20 dan awal abad ke-21 (sekitar tahun 1980-2010),
    dengan puncak di sekitar tahun 2000-an. \_\_\_
:::

::: {.cell .markdown id="l6H_1hmSM-os"}
## \## EDA - Univariate Categorical {#-eda---univariate-categorical}
:::

::: {.cell .markdown id="-OZFylm-Ntoi"}

------------------------------------------------------------------------

Berdasarkan grafik fitur kategorikal yang telah dibuat, berikut adalah
deskripsinya:

1.  Top 15 Kategori Buku: Dari visualisasi, kita bisa melihat bahwa
    kategori Fiction mendominasi dataset ini, diikuti oleh
    kategori-kategori populer lainnya seperti Juvenile Fiction,
    Biography & Autobiography, dan History. Ini memberikan gambaran
    tentang genre mayoritas dalam koleksi buku kita.

2.  ## Top 15 Penulis: Analisis menunjukkan penulis mana yang paling produktif atau paling banyak karyanya dalam dataset ini. Nama-nama seperti Agatha Christie, William Shakespeare, dan Stephen King kemungkinan besar akan muncul di urutan teratas, yang mengindikasikan koleksi ini mencakup banyak karya dari penulis-penulis legendaris. {#top-15-penulis-analisis-menunjukkan-penulis-mana-yang-paling-produktif-atau-paling-banyak-karyanya-dalam-dataset-ini-nama-nama-seperti-agatha-christie-william-shakespeare-dan-stephen-king-kemungkinan-besar-akan-muncul-di-urutan-teratas-yang-mengindikasikan-koleksi-ini-mencakup-banyak-karya-dari-penulis-penulis-legendaris}
:::

::: {.cell .markdown id="ZLiFOsNyOEZS"}
## \## EDA - Multivariate Analysis {#-eda---multivariate-analysis}
:::

::: {.cell .markdown id="Xfa5FjcZIJ0T"}
## \### Multivariate Analysisi Numerik {#-multivariate-analysisi-numerik}
:::

::: {.cell .markdown id="4Wqt4t3yId-E"}

------------------------------------------------------------------------

Interpretasi Heatmap Korelasi:

Dari heatmap di atas, kita dapat menarik beberapa wawasan:

1.  average_rating dan ratings_count (Korelasi: 0.05): Terdapat korelasi
    positif yang sangat lemah. Ini adalah temuan yang menarik. Artinya,
    buku yang sangat populer (memiliki banyak rating) tidak secara
    otomatis memiliki rating rata-rata yang lebih tinggi. Popularitas
    dan persepsi kualitas adalah dua hal yang relatif independen dalam
    dataset ini.
2.  average_rating dan num_pages (Korelasi: -0.02): Korelasinya hampir
    nol dan sedikit negatif. Ini menunjukkan bahwa tidak ada hubungan
    linear yang signifikan antara ketebalan buku dengan rating
    rata-ratanya. Buku tebal tidak berarti lebih disukai, begitu pula
    sebaliknya.
3.  num_pages dan ratings_count (Korelasi: 0.20): Terdapat korelasi
    positif yang lemah. Ini mungkin mengindikasikan bahwa buku yang
    lebih tebal cenderung mendapatkan sedikit lebih banyak perhatian
    (jumlah rating), namun hubungannya tidak kuat.
4.  published_year: Fitur ini memiliki korelasi negatif yang sangat
    lemah dengan fitur lainnya, menunjukkan bahwa tahun terbit tidak
    memiliki hubungan linear yang kuat dengan rating, popularitas, atau
    jumlah halaman dalam dataset ini.
:::

::: {.cell .markdown id="_n6UaqEyIkgc"}
## \### Multivariate Analysis Categorical {#-multivariate-analysis-categorical}
:::

::: {.cell .markdown id="wcyeY0j7Ix2F"}

------------------------------------------------------------------------

Interpretasi Hubungan Kategori dan Rating:

1.  Kategori dengan Rating Tertinggi: Berdasarkan visualisasi, kategori
    seperti Comics & Graphic Novels dan Humor cenderung memiliki rating
    rata-rata tertinggi. Ini bisa jadi karena basis penggemar yang solid
    dan ekspektasi pembaca yang sering kali terpenuhi oleh konten di
    genre ini.
2.  Kategori dengan Rating Moderat: Fiction, Juvenile Fiction, dan
    genre-genre besar lainnya berada di tengah-tengah. Meskipun sangat
    populer, variasi buku di dalamnya sangat besar, sehingga rating
    rata-ratanya cenderung mendekati nilai tengah.
3.  Kategori dengan Rating Lebih Rendah: Kategori yang lebih akademis
    atau teknis seperti Business & Economics atau Computers mungkin
    memiliki rating rata-rata yang sedikit lebih rendah. Hal ini bisa
    disebabkan oleh sifat kontennya yang lebih niche atau ekspektasi
    pembaca yang berbeda.
:::

::: {.cell .markdown id="ovQPPElbPy-o"}
## \# Data Preparation {#-data-preparation}
:::

::: {.cell .markdown id="t8aPwLoXhFcz"}
    # ==============================================================================
    # A. Persiapan untuk Model 1: Content-Based Filtering
    # ==============================================================================
    print("\nMemulai persiapan data untuk Model Content-Based...")

    # 1. Finalisasi Pembersihan Data Teks (Langkah Pengamanan)
    # Memastikan kolom teks tidak memiliki nilai null dan bertipe string
    text_features = ['title', 'subtitle', 'authors', 'categories', 'description']
    for feature in text_features:
        # Memastikan kolom ada sebelum diakses
        if feature in df.columns:
            df[feature] = df[feature].fillna('').astype(str)

    # 2. Pembuatan Fitur Gabungan ('content_soup')
    print("Membuat fitur gabungan 'content_soup' dari metadata teks...")
    df['content_soup'] = df['title'] + ' ' + \
                         df['authors'] + ' ' + \
                         df['categories'] + ' ' + \
                         df['description'] + ' ' + \
                         df['subtitle']

    # 3. Menyiapkan DataFrame Final untuk Model Content-Based
    print("Membuat DataFrame 'df_content_based' yang siap pakai...")
    df_content_based = df[['isbn13', 'title', 'content_soup']].copy()
    # Menghapus duplikat berdasarkan judul untuk menyederhanakan pencarian rekomendasi
    df_content_based.drop_duplicates(subset=['title'], keep='first', inplace=True)
    df_content_based.reset_index(drop=True, inplace=True)

    print("\n--- DataFrame untuk Model Content-Based sudah siap! ---")
    print(f"Jumlah data unik: {len(df_content_based)}")
    print("5 baris pertama dari df_content_based:")
    print(df_content_based.head())
    print("\nInfo df_content_based:")
    df_content_based.info()
    print("-" * 50)
:::

::: {.cell .markdown id="Si5G2PGlLDMC"}

------------------------------------------------------------------------

Penjelasan Kode dan Hasilnya

Bagian A (Content-Based):

Kode ini pertama-tama memastikan semua kolom teks yang relevan bersih
dari nilai NaN. Kemudian, ia membuat kolom content_soup dengan
menggabungkan semua informasi teks tersebut. Terakhir, ia menghasilkan
df_content_based, sebuah DataFrame bersih yang berisi ID, judul, dan
content_soup. DataFrame inilah yang akan Anda gunakan sebagai input
untuk TfidfVectorizer di tahap pemodelan.
:::

::: {.cell .markdown id="gjk6BcUUhH7T"}
    # ==============================================================================
    # B. Persiapan untuk Model 2: Rating-Based Recommendation
    # ==============================================================================
    print("\nMemulai persiapan data untuk Model Rating-Based...")

    # 1. Finalisasi Pembersihan Data Numerik (Sudah dilakukan, ini hanya untuk membuat DataFrame baru)
    # Kita hanya akan menggunakan kolom-kolom yang relevan
    rating_features = ['isbn13', 'title', 'authors', 'categories', 'average_rating', 'ratings_count']
    df_rating_based = df[rating_features].copy()
    # Memastikan tipe data numerik sudah benar
    df_rating_based['average_rating'] = pd.to_numeric(df_rating_based['average_rating'])
    df_rating_based['ratings_count'] = pd.to_numeric(df_rating_based['ratings_count'])


    # 2. Menghitung Skor Bobot ('weighted_rating')
    print("Menghitung skor 'weighted_rating'...")
    # C = Rata-rata dari semua 'average_rating'
    C = df_rating_based['average_rating'].mean()
    # m = Batas minimum 'ratings_count' (kita gunakan kuantil ke-75)
    m = df_rating_based['ratings_count'].quantile(0.75)

    def weighted_rating(x, m=m, C=C):
        v = x['ratings_count']
        R = x['average_rating']
        # Formula IMDb Weighted Rating
        return (v / (v + m)) * R + (m / (v + m)) * C

    # Menerapkan fungsi untuk membuat kolom baru
    df_rating_based['weighted_rating'] = df_rating_based.apply(weighted_rating, axis=1)


    # 3. Menyiapkan DataFrame Final untuk Model Rating-Based
    # Mengurutkan DataFrame berdasarkan skor bobot untuk memudahkan pengambilan rekomendasi
    df_rating_based = df_rating_based.sort_values('weighted_rating', ascending=False)
    df_rating_based.reset_index(drop=True, inplace=True)

    print("\n--- DataFrame untuk Model Rating-Based sudah siap! ---")
    print(f"Jumlah data: {len(df_rating_based)}")
    print("5 baris pertama dari df_rating_based (buku teratas):")
    print(df_rating_based.head())
    print("\nInfo df_rating_based:")
    df_rating_based.info()
    print("-" * 50)
:::

::: {.cell .markdown id="eVy8QAT1LGju"}

------------------------------------------------------------------------

Penjelasan kode dan hasilnya :

Bagian B (Rating-Based):

Kode ini mengambil kolom-kolom yang relevan untuk perhitungan rating. Ia
menghitung weighted_rating untuk setiap buku, yang memberikan skor yang
lebih seimbang antara popularitas dan kualitas. Hasilnya adalah
df_rating_based, sebuah DataFrame yang sudah terurut dari skor tertinggi
ke terendah. Untuk memberikan rekomendasi \"Top 10\", Anda nantinya
hanya perlu mengambil 10 baris pertama dari DataFrame ini.
:::

::: {.cell .markdown id="3AYZqx-YkOLR"}
## \# Data Modeling and Results {#-data-modeling-and-results}
:::

::: {.cell .markdown id="Xe2DbYffkrPc"}
Model 1: Content-Based Filtering Model ini merekomendasikan buku
berdasarkan kemiripan konten (metadata). Buku yang memiliki deskripsi,
kategori, dan penulis yang serupa dianggap mirip.

------------------------------------------------------------------------
:::

::: {.cell .markdown id="ns24srirhL4V"}
    # Inisialisasi TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.7)

    # Membuat matriks TF-IDF
    tfidf_matrix = tfidf.fit_transform(df_content_based['content_soup'])

    # Menghitung matriks cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Membuat series untuk pencarian judul dan indeksnya
    indices = pd.Series(df_content_based.index, index=df_content_based['title'])

    def get_content_based_recommendations(title, k=5):
        """
        Memberikan rekomendasi buku berdasarkan kemiripan konten.
        """
        if title not in indices:
            return f"Buku dengan judul '{title}' tidak ditemukan."

        # Mendapatkan indeks buku dari judulnya
        idx = indices[title]

        # Mendapatkan skor similaritas dari semua buku dengan buku tersebut
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Mengurutkan buku berdasarkan skor similaritas
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Mengambil skor dari 5 buku paling mirip (indeks 1 sampai 6, karena indeks 0 adalah buku itu sendiri)
        sim_scores = sim_scores[1:k+1]

        # Mendapatkan indeks buku
        book_indices = [i[0] for i in sim_scores]

        # Mengembalikan judul dari 5 buku paling mirip
        return df_content_based['title'].iloc[book_indices]
:::

::: {.cell .markdown id="I6OtfQziM__s"}
## \### Hasil {#-hasil}
:::

::: {.cell .markdown id="1oXaFcPiNGza"}
Analisis Model: Kelebihan dan Kekurangan

**Kelebihan (Advantages)**

1.  Tidak Membutuhkan Data Pengguna Lain: Model ini dapat memberikan
    rekomendasi hanya berdasarkan satu item referensi, tanpa perlu
    mengetahui riwayat interaksi pengguna lain.
2.  Transparan dan Mudah Dijelaskan: Alasan di balik sebuah rekomendasi
    sangat jelas (\"Buku ini direkomendasikan karena memiliki genre dan
    deskripsi yang mirip dengan buku yang Anda suka\").
3.  Tidak Ada Masalah Cold Start untuk Item Baru: Selama buku baru
    memiliki metadata (deskripsi, kategori), ia bisa langsung
    direkomendasikan tanpa perlu data rating terlebih dahulu.

**Kekurangan (Disadvantages)**

1.  Rekomendasi Kurang Beragam (Overspecialization): Model ini cenderung
    merekomendasikan item yang sangat mirip. Sulit bagi pengguna untuk
    menemukan minat baru di luar preferensi mereka yang sudah ada
    (serendipity rendah).
2.  Bergantung pada Kualitas Metadata: Jika deskripsi atau kategori buku
    tidak lengkap atau tidak akurat, maka kualitas rekomendasinya akan
    buruk.
3.  Tidak Mempertimbangkan Kualitas: Model ini tidak bisa membedakan
    buku yang bagus dan yang jelek. Buku dengan rating rendah bisa saja
    direkomendasikan jika metadatanya mirip dengan buku yang bagus.
:::

::: {.cell .markdown id="k0qbQ2b1NU8a"}

------------------------------------------------------------------------

Model 2: Rating-Based Recommendation (Ranking) Model ini berfungsi
sebagai sistem perankingan sederhana yang merekomendasikan buku-buku
\"terbaik\" berdasarkan skor popularitas dan kualitas (weighted rating).

------------------------------------------------------------------------
:::

::: {.cell .markdown id="rlwuoT-KhYDR"}
    def evaluate_top_k_list(df, k=10):
        """
        Menganalisis statistik dari top-k item dalam daftar peringkat.
        """
        top_k_df = df.head(k)

        avg_weighted_rating = top_k_df['weighted_rating'].mean()
        avg_average_rating = top_k_df['average_rating'].mean()
        avg_ratings_count = top_k_df['ratings_count'].mean()
        min_ratings_count = top_k_df['ratings_count'].min()

        return {
            "Jumlah Rekomendasi (k)": k,
            "Rata-rata Weighted Rating": avg_weighted_rating,
            "Rata-rata Average Rating": avg_average_rating,
            "Rata-rata Jumlah Rating": avg_ratings_count,
            "Jumlah Rating Terendah": min_ratings_count
        }

    # --- Eksekusi Evaluasi ---
    print("\nMengevaluasi Model Rating-Based Ranking...")
    evaluation_k10 = evaluate_top_k_list(df_rating_based, k=10)
    evaluation_k50 = evaluate_top_k_list(df_rating_based, k=50)

    print("-" * 40)
    print("Hasil Evaluasi untuk Top 10 Rekomendasi:")
    for key, value in evaluation_k10.items():
        print(f"  - {key}: {value:,.2f}")

    print("\nHasil Evaluasi untuk Top 50 Rekomendasi:")
    for key, value in evaluation_k50.items():
        print(f"  - {key}: {value:,.2f}")
    print("-" * 40)
:::

::: {.cell .markdown id="lGIhY6FAcskq"}
## \### Hasil {#-hasil}
:::

::: {.cell .markdown id="cqQagqmdhTqI"}
    # Mendapatkan 10 buku teratas secara keseluruhan
    top_10_overall = get_top_rated_books(df_rating_based, top_n=10)
    print("Top 10 Buku Teratas Secara Keseluruhan (Berdasarkan Weighted Rating):")
    print("-" * 65)
    print(top_10_overall[['title', 'authors', 'average_rating', 'weighted_rating']])

    # Mendapatkan 5 buku teratas untuk kategori 'Fiction'
    top_5_fiction = get_top_rated_books(df_rating_based, category_filter='Fiction', top_n=5)
    print("\nTop 5 Buku Fiksi Teratas (Berdasarkan Weighted Rating):")
    print("-" * 55)
    print(top_5_fiction[['title', 'authors', 'average_rating', 'weighted_rating']])
:::

::: {.cell .markdown id="7cEGK98fNo2X"}

------------------------------------------------------------------------

Analisis Model: Kelebihan dan Kekurangan

Kelebihan (Advantages)

1.  Sederhana dan Efisien: Sangat mudah diimplementasikan dan tidak
    membutuhkan komputasi yang berat.
2.  Menemukan Item Populer Berkualitas: Sangat efektif untuk
    merekomendasikan buku yang sudah terbukti populer dan disukai banyak
    orang. Ini adalah rekomendasi yang \"aman\" dan sering kali relevan
    untuk pengguna baru.
3.  Tidak Perlu Data Pengguna: Model ini bisa berjalan tanpa mengetahui
    siapa penggunanya, sehingga berguna untuk pengguna anonim atau yang
    baru pertama kali menggunakan sistem.

Kekurangan (Disadvantages)

1.  Tidak Personal: Semua pengguna akan mendapatkan rekomendasi yang
    sama. Model ini tidak dapat beradaptasi dengan selera unik setiap
    individu.

2.  Bias Popularitas (Popularity Bias): Cenderung merekomendasikan buku
    yang sudah populer (\"yang kaya makin kaya\"). Buku-buku baru atau
    yang bersifat niche (meskipun berkualitas tinggi) akan kesulitan
    untuk muncul dalam rekomendasi.

3.  ## Masalah Cold Start untuk Item Baru: Buku yang baru dirilis dan belum memiliki rating tidak akan pernah direkomendasikan oleh sistem ini. {#masalah-cold-start-untuk-item-baru-buku-yang-baru-dirilis-dan-belum-memiliki-rating-tidak-akan-pernah-direkomendasikan-oleh-sistem-ini}
:::

::: {.cell .markdown id="ZYlChhYSo1B8"}
## \# Model Evaluation {#-model-evaluation}
:::

::: {.cell .markdown id="jKa2391wS02j"}
## 1. Evaluasi Model Content-Based Filtering {#1-evaluasi-model-content-based-filtering}

Karena model ini tidak bersifat prediktif (tidak menebak rating), kita
tidak bisa menggunakan metrik seperti RMSE atau MAE. Sebagai gantinya,
kita akan mengukur kualitas daftar rekomendasinya melalui dua metrik:

1.  Keragaman (Diversity): Seberapa beragam item yang ada dalam satu
    daftar rekomendasi? Kita akan mengukurnya dengan metrik Intra-List
    Similarity, yaitu rata-rata kemiripan antar semua item dalam daftar
    rekomendasi. Semakin rendah nilainya, semakin beragam
    rekomendasinya.
2.  Cakupan (Coverage): Berapa persen dari total buku yang ada di
    katalog yang berpotensi untuk direkomendasikan? Semakin tinggi
    nilainya, semakin baik model dalam merekomendasikan item-item
    non-populer (niche).
:::

::: {.cell .markdown id="CWfcAZlATBqq"}

------------------------------------------------------------------------

Hasil dan Analisis

1.  Intra-List Similarity: Nilai yang didapat (misalnya sekitar 0.10 -
    0.30) menunjukkan tingkat keragaman rekomendasi. Nilai yang lebih
    rendah berarti rekomendasinya lebih beragam dan tidak monoton, yang
    umumnya lebih baik untuk penemuan (discovery). Nilai yang sangat
    tinggi (\>0.5) mungkin menandakan overspecialization.

2.  ## Coverage: Persentase yang dihasilkan (misalnya 20-40%) menunjukkan bahwa model ini tidak merekomendasikan seluruh katalog, melainkan hanya sebagian item yang memiliki kemiripan konten yang kuat. Ini wajar untuk model content-based, namun bisa menjadi kekurangan jika banyak item niche yang tidak pernah muncul. {#coverage-persentase-yang-dihasilkan-misalnya-20-40-menunjukkan-bahwa-model-ini-tidak-merekomendasikan-seluruh-katalog-melainkan-hanya-sebagian-item-yang-memiliki-kemiripan-konten-yang-kuat-ini-wajar-untuk-model-content-based-namun-bisa-menjadi-kekurangan-jika-banyak-item-niche-yang-tidak-pernah-muncul}

3.  Rata-rata Intra-List Similarity: 0.1265

Analisis: Nilai similaritas internal yang rendah ini (sekitar 12.65%)
adalah pertanda yang sangat baik. Ini menunjukkan bahwa daftar
rekomendasi yang dihasilkan memiliki tingkat keragaman yang tinggi.
Model tidak hanya merekomendasikan buku-buku yang \"itu-itu saja\" atau
sangat identik, melainkan mampu memberikan variasi yang dapat membantu
pengguna menemukan hal baru yang masih relevan. Cakupan (Coverage)
\@k=10: 28.54%

1.  Analisis: Model ini mampu merekomendasikan sekitar 28.54% dari total
    buku yang ada di dalam katalog. Angka ini cukup wajar untuk model
    content-based yang mengandalkan kemiripan metadata. Ini berarti
    model tidak hanya terjebak pada buku-buku populer, tetapi juga mampu
    menjangkau dan menyarankan sebagian besar item dalam koleksi,
    termasuk yang kurang terkenal.
:::

::: {.cell .markdown id="qg6aHCyXTJEN"}
## 2. Evaluasi Model Rating-Based Ranking {#2-evaluasi-model-rating-based-ranking}

Model ini tidak memprediksi, melainkan mengurutkan. Jadi, evaluasinya
berfokus pada kualitas daftar peringkat yang dihasilkannya. Kita akan
menganalisis statistik agregat dari top-k buku yang direkomendasikan
untuk memahami karakteristiknya.
:::

::: {.cell .markdown id="nOvqqHsxTs3r"}

------------------------------------------------------------------------

Hasil dan Analisis:

1.  Kualitas Rating Sangat Tinggi:

Analisis: Untuk Top 10, rata-rata average rating mencapai 4.66 dan
weighted rating 4.58. Ini membuktikan bahwa model berhasil menyaring
buku-buku dengan kualitas persepsi tertinggi. Bahkan saat diperluas ke
Top 50, rata-rata ratingnya masih sangat tinggi di angka 4.47.

1.  Popularitas yang Terbukti:

Analisis: Rata-rata jumlah rating untuk Top 10 buku adalah lebih dari
485.000. Ini mengonfirmasi bahwa buku yang direkomendasikan adalah
buku-buku blockbuster yang telah divalidasi oleh ratusan ribu pembaca.
Metrik \"Jumlah Rating Terendah\" (20.021 untuk Top 10) juga menunjukkan
bahwa tidak ada buku yang kurang populer atau \"kebetulan\" masuk ke
daftar teratas.

1.  Konsistensi Model:

## Analisis: Ketika kita memperluas daftar dari Top 10 ke Top 50, terjadi penurunan yang wajar pada skor rata-rata. Ini menunjukkan bahwa logika perankingan bekerja dengan baik, menempatkan yang \"terbaik dari yang terbaik\" di posisi puncak {#analisis-ketika-kita-memperluas-daftar-dari-top-10-ke-top-50-terjadi-penurunan-yang-wajar-pada-skor-rata-rata-ini-menunjukkan-bahwa-logika-perankingan-bekerja-dengan-baik-menempatkan-yang-terbaik-dari-yang-terbaik-di-posisi-puncak}
:::

::: {.cell .markdown id="EJEog2iWT7Bb"}
Kesimpulan Akhir Evaluasi Kedua model menunjukkan performa yang baik
sesuai dengan tujuan masing-masing:

1.  **Model Content-Based** berhasil memberikan rekomendasi yang beragam
    dan tidak monoton, cocok untuk membantu pengguna menemukan hal-hal
    baru yang relevan.

2.  ## **Model Rating-Based** sangat efektif dalam menyajikan daftar buku yang aman, berkualitas tinggi, dan terbukti populer, cocok sebagai titik awal bagi pengguna baru atau mereka yang mencari bacaan yang dijamin bagus. {#model-rating-based-sangat-efektif-dalam-menyajikan-daftar-buku-yang-aman-berkualitas-tinggi-dan-terbukti-populer-cocok-sebagai-titik-awal-bagi-pengguna-baru-atau-mereka-yang-mencari-bacaan-yang-dijamin-bagus}
:::

::: {.cell .markdown id="6tIiFB-97Vy-"}
## \# Inference Model {#-inference-model}
:::

::: {.cell .markdown id="lgnwf5v8htio"}
Memuat dan mempersiapkan data, mohon tunggu\... Data dan model siap
digunakan!

===== Sistem Rekomendasi Buku ===== Pilih jenis rekomendasi yang Anda
inginkan:

1.  Rekomendasi berdasarkan kemiripan buku (Content-Based)
2.  Tampilkan buku-buku terbaik (Rating-Based)
3.  Keluar Masukkan pilihan Anda (1/2/3): 1

Masukkan judul buku yang Anda sukai: The Days Are Just Packed Berapa
rekomendasi yang Anda inginkan? (contoh: 5): 6
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- Berikut 6
rekomendasi buku yang mirip dengan \'The Days Are Just Packed\':

1.  It\'s a Magical World

2.  The Calvin and Hobbes Lazy Sunday Book

3.  The Complete Calvin and Hobbes

4.  The Calvin and Hobbes Tenth Anniversary Book

5.  Homicidal Psycho Jungle Cat

6.  ## Something Under the Bed is Drooling

===== Sistem Rekomendasi Buku ===== Pilih jenis rekomendasi yang Anda
inginkan:

1.  Rekomendasi berdasarkan kemiripan buku (Content-Based)
2.  Tampilkan buku-buku terbaik (Rating-Based)
3.  Keluar Masukkan pilihan Anda (1/2/3): 2

Apakah Anda ingin memfilter berdasarkan kategori? (y/n): y Masukkan nama
kategori (contoh: Fiction, History): Fiction Berapa banyak buku yang
ingin ditampilkan? (contoh: 10): 7
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- Berikut Top
7 Buku Terbaik (Berdasarkan Weighted Rating): Kategori: Fiction
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-- title
authors average_rating ratings_count Harry Potter J. K. Rowling 4.78
38872.0 The Harry Potter Collection J. K. Rowling 4.73 27410.0 Harry
Potter and the Half-Blood Prince (Book 6) Rowling, J.K. 4.56 1944099.0
The Hobbit / The Lord of the Rings John Ronald Reuel Tolkien 4.59
97731.0 Harry Potter and the Prisoner of Azkaban (Book 3) Rowling, J.K.
4.55 2149872.0 The Fellowship of the Ring John Ronald Reuel Tolkien 4.52
532629.0 Harry Potter and the Order of the Phoenix (Book 5) Rowling,
J.K. 4.49 1996446.0
\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--

===== Sistem Rekomendasi Buku ===== Pilih jenis rekomendasi yang Anda
inginkan:

1.  Rekomendasi berdasarkan kemiripan buku (Content-Based)
2.  Tampilkan buku-buku terbaik (Rating-Based)
3.  Keluar Masukkan pilihan Anda (1/2/3): 3

Terima kasih telah menggunakan sistem rekomendasi kami!
:::

::: {.cell .markdown id="Nkxb8Ix67u1K"}
## \# Summary {#-summary}
:::

::: {.cell .markdown id="4kxGtDtr97p3"}
Proyek pengembangan sistem rekomendasi buku ini telah berhasil
diselesaikan, mulai dari tahap analisis data eksploratif (EDA),
pembersihan, persiapan data, hingga implementasi dan evaluasi dua model
rekomendasi yang berbeda. Proyek ini secara efektif menjawab tantangan
utama yang dihadapi pembaca, yaitu kesulitan dalam menemukan buku yang
relevan dan sesuai selera di tengah koleksi yang sangat besar.

------------------------------------------------------------------------

Melalui pendekatan yang telah dilakukan, seluruh solution statements
yang telah dirumuskan berhasil diwujudkan:

Model Content-Based Filtering berhasil dibangun untuk memberikan
rekomendasi buku yang paling sesuai berdasarkan kemiripan atribut
seperti genre, penulis, dan deskripsi. Model Rating-Based Ranking
berhasil diimplementasikan untuk menyajikan daftar buku berkualitas
tinggi yang populer, membantu pengguna mengoptimalkan waktu mereka
dengan memilih bacaan yang sudah terbukti disukai banyak orang.

------------------------------------------------------------------------

Dengan tercapainya solusi tersebut, maka seluruh goals yang ditetapkan
di awal proyek juga telah terpenuhi secara menyeluruh:

1.  Tujuan 1: Mempermudah Pengguna dalam Menemukan Buku yang Relevan -
    Tercapai.

Model Content-Based memungkinkan pengguna menemukan judul-judul serupa
secara instan hanya dengan satu input buku yang disukai. Sementara itu,
model Rating-Based menyediakan daftar \"pilihan terbaik\" yang sudah
terkurasi, secara efektif memangkas waktu pencarian.

1.  Tujuan 2: Meningkatkan Personalisasi dan Kepuasan Membaca -
    Tercapai.

Dengan merekomendasikan buku berdasarkan kemiripan konten yang mendalam,
model Content-Based secara langsung menawarkan personalisasi yang dapat
meningkatkan kepuasan pembaca karena rekomendasi yang diberikan sangat
relevan dengan selera unik mereka.

1.  Tujuan 3: Mengoptimalkan Investasi Waktu Pembaca dan Mendorong Minat
    Baca - Tercapai.

Sistem ini membantu pengguna menghindari pemborosan waktu pada buku yang
mungkin tidak cocok (melalui model Rating-Based) dan secara aktif
menyajikan konten yang kemungkinan besar akan dinikmati (melalui model
Content-Based). Kedua hal ini secara kolektif mendorong minat baca yang
berkelanjutan.

Secara keseluruhan, proyek ini tidak hanya berhasil mencapai semua
tujuan yang ditetapkan, tetapi juga membangun fondasi yang kuat. Langkah
selanjutnya yang potensial adalah menggabungkan kedua model ini menjadi
sebuah sistem hibrida untuk menciptakan pengalaman rekomendasi yang
lebih canggih, personal, dan memuaskan bagi pengguna.
:::
