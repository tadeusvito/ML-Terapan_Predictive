# Laporan Proyek Machine Learning-Terapan - Tadeus Vito Gavra Sitanggang
## Domain Proyek
Industri perhotelan telah berkembang pesat dalam beberapa dekade terakhir, seiring dengan meningkatnya kebutuhan masyarakat akan akomodasi yang terjangkau dan berkualitas saat bepergian. Namun, penentuan harga hotel seringkali menjadi tantangan tersendiri bagi pihak manajemen hotel. Pihak hotel sering menetapkan harga berdasarkan pengalaman subjektif atau perbandingan sederhana dengan hotel serupa, sementara calon tamu mengandalkan negosiasi tanpa memahami nilai sebenarnya dari hotel yang mereka pilih. Akibatnya, sering terjadi ketidakseimbangan antara harga pasar dengan nilai yang diterima tamu, yang dapat merugikan baik pihak hotel maupun tamu.

Tujuan proyek ini adalah untuk mengembangkan model prediktif yang dapat membantu pihak hotel dalam menentukan harga yang lebih akurat dan sesuai dengan nilai yang diterima tamu. Dengan menganalisis fitur-fitur kunci seperti rating hotel, jumlah malam menginap, jumlah tamu, jarak ke atraksi, dan waktu pemesanan, diharapkan model ini dapat memberikan wawasan baru dan meningkatkan kemampuan memprediksi harga hotel secara lebih baik.

Hasil akhir proyek ini diharapkan dapat membantu pihak hotel dalam menetapkan harga yang kompetitif dan sesuai dengan ekspektasi tamu, serta membantu calon tamu dalam memahami nilai yang mereka dapatkan saat memesan hotel.

## Business Understanding
### Problem Statements
- Bagaimana model prediktif dapat membantu pihak hotel dalam menentukan harga yang lebih akurat dan sesuai dengan nilai yang diterima tamu, serta membantu calon tamu dalam memahami nilai yang mereka dapatkan saat memesan hotel?
- Bagaimana faktor-faktor seperti rating hotel, jumlah malam menginap, jumlah tamu, dan waktu pemesanan memengaruhi harga hotel?

### Goals
- Mengidentifikasi faktor-faktor kunci seperti rating hotel, jumlah malam menginap, jumlah tamu, dan waktu pemesanan yang memengaruhi harga hotel.
- Mengembangkan model prediktif yang dapat membantu pihak hotel dalam menentukan harga hotel yang lebih akurat dan sesuai dengan nilai yang diterima tamu.

### Solution statements
- Menggunakan algoritma machine learning dasar seperti Regresi Linier, K-Nearest Neighbors (KNN), dan Random Forest sebagai model awal untuk memprediksi harga hotel.
- Menggunakan algoritma machine learning lanjutan seperti AdaBoost untuk meningkatkan akurasi prediksi harga hotel.
- Melakukan tuning hyperparameter pada model-model terpilih menggunakan teknik seperti Grid Search dan Random Search untuk menemukan kombinasi parameter yang optimal.
- Mengevaluasi performa model setelah tuning hyperparameter dan membandingkannya dengan model awal untuk memastikan peningkatan akurasi prediksi.

## Data Understanding
Dataset ini merupakan dataset sintetis yang berisi informasi mengenai 600 sampel booking hotel yang dirancang untuk mendukung penelitian dalam bidang machine learning. Dataset ini mencakup berbagai fitur penting, seperti rating hotel, jumlah malam menginap, jumlah tamu, dan waktu pemesanan yang memengaruhi harga hotel yang diambil dari referensi yang ada di Kaggle.  Dengan fokus pada kemudahan penggunaan dan efisiensi, dataset ini dirancang untuk memenuhi standar tinggi yang diperlukan dalam analisis dan pengembangan model prediktif. Keberadaan dataset ini diharapkan dapat memberikan wawasan yang berharga dalam memahami faktor-faktor yang memengaruhi harga hotel dan membantu dalam pengembangan solusi berbasis machine learning.

Sumber Referensi: [Hotel Booking](https://www.kaggle.com/datasets/mojtaba142/hotel-booking)

### Deskripsi Variable
|      Nama Kolom       |                         Deskripsi                           |  
|-----------------------|------------------------------------------------------------|  
| Hotel Rating          | Rating hotel (numerik) berdasarkan penilaian pengguna      |  
| Number of Nights      | Jumlah malam menginap (numerik)                            |  
| Number of Guests      | Jumlah tamu yang menginap (numerik)                       |  
| Room Type             | Tipe kamar yang dipesan (kategorikal)                      |  
| Hotel Location        | Lokasi hotel (kategorikal)                                 |  
| Distance to Attractions| Jarak hotel ke atraksi terdekat (numerik, dalam kilometer)|  
| Season                | Musim saat pemesanan (kategorikal)                        |  
| Hotel Facilities      | Fasilitas yang tersedia di hotel (kategorikal)            |  
| Booking Time          | Waktu pemesanan dalam jam (numerik)                       |  
| Price per Night       | Harga per malam untuk menginap (numerik)                  |  

- **Jumlah Baris (Entries)**: 600 baris data, terdapat 600 entri atau record yang tercatat dalam dataset ini.  
- **Jumlah Kolom (Columns)**: 10 kolom data, yang mencakup berbagai atribut terkait booking hotel yang dicatat dalam dataset. 

### Missing Value
![image](https://github.com/user-attachments/assets/5adb9808-684b-4f43-b69c-5714447091e5)

Hasil output diatas menunjukkan bahwa tidak ada nilai yang hilang (missing value) di setiap kolom dalam dataset. Keberadaan data yang lengkap ini sangat penting untuk analisis lebih lanjut, karena memastikan bahwa semua informasi terkait hotel, seperti rating, jumlah tamu, jenis kamar, lokasi, jarak ke atraksi, musim, fasilitas, waktu pemesanan, dan harga per malam, tersedia secara utuh. Dengan demikian, analisis dan pengambilan keputusan dapat dilakukan dengan lebih akurat dan efektif.


### Data Duplicate 
![image](https://github.com/user-attachments/assets/5b246706-7a68-4127-879f-0252e888123b)

Pada hasil output di atas menunjukkan bahwa tidak ada data yang memiliki isi duplikat sama dengan yang lainnya. Hal ini menandakan bahwa dataset yang digunakan telah bersih dari entri yang berulang, yang dapat mempengaruhi keakuratan model.

### Outliers
![image](https://github.com/user-attachments/assets/fc42dc80-333d-440a-91fb-682cb8196cd7)
![image](https://github.com/user-attachments/assets/8dbca107-edd4-4f71-bc37-33ba037324d8)
Dari analisis box plot pada dataset ini, dapat disimpulkan bahwa sebagian besar data terdistribusi dengan baik tanpa adanya outlier yang signifikan. Beberapa nilai ekstrem terdeteksi pada sebagian kecil data, namun tidak cukup mencolok untuk mempengaruhi validitas analisis atau model prediksi secara keseluruhan. Secara umum, distribusi data terlihat stabil, dengan rentang nilai yang terpusat dan terjaga. Meskipun demikian, beberapa nilai ekstrem perlu diperhatikan lebih lanjut untuk memastikan konsistensi dan akurasi dalam analisis lebih mendalam.



## Exploratory Data Analysis  

### Univariate Analysis  

#### Categorical Features  

**Room Type:**  

![Room Type Distribution](https://github.com/user-attachments/assets/7adf043a-4bfd-4370-a01d-f2f13af40405)


**Insight:**  
Dari grafik yang ditampilkan, dapat dilihat distribusi tipe kamar pada dataset ini. Tipe kamar Suite memiliki jumlah sampel terbesar, yaitu 205 dengan persentase 34,2%. Diikuti oleh Tipe kamar Single dengan 202 sampel (33,7%) dan Tipe kamar Double dengan 193 sampel (32,2%). Ini menunjukkan bahwa pembagian pemesanan kamar hampir merata antara ketiga jenis kamar, dengan sedikit lebih banyak pemesanan untuk tipe Suite dan Single dibandingkan dengan tipe Double.  

---  

**Hotel Location:**  

![Hotel Location Distribution](https://github.com/user-attachments/assets/1abae8e3-44d0-4759-94bf-3cc4af35b28f)


**Insight:**  
Grafik di atas menunjukkan distribusi data terkait lokasi hotel berdasarkan jumlah sampel dan persentase. Ketiga lokasi—Suburbs, City Center, dan Countryside—memiliki distribusi yang relatif seimbang, dengan sedikit perbedaan antara masing-masing. Suburbs memiliki jumlah sampel tertinggi (207) dan persentase terbesar (34.5%), diikuti oleh City Center (197 sampel, 32.8%) dan Countryside (196 sampel, 32.7%). Ini menunjukkan bahwa ada kecenderungan hampir seimbang dalam pemilihan lokasi hotel, meskipun Suburbs sedikit lebih dominan dalam hal jumlah sampel.  

---  

**Season:**  

![Season Distribution](https://github.com/user-attachments/assets/45a7197d-37a3-4092-a241-7ed144687a67)


**Insight:**  
Grafik di atas menunjukkan distribusi data berdasarkan musim, dengan dua kategori yaitu High Season dan Low Season. High Season memiliki jumlah sampel tertinggi (312) dan persentase terbesar (52%), sementara Low Season memiliki 288 sampel dan persentase 48%. Meskipun ada sedikit perbedaan antara keduanya, High Season memiliki sedikit lebih banyak sampel, yang mencerminkan sedikit lebih banyak aktivitas atau pengunjung selama musim tersebut.  

---  

**Hotel Facilities:**  

![Hotel Facilities Distribution](https://github.com/user-attachments/assets/83126b59-f51e-4281-9bac-94248843a525)


**Insight:**  
Grafik di atas menunjukkan distribusi fasilitas hotel yang tersedia dalam dataset. Dari total 600 sampel, fasilitas yang paling banyak tersedia adalah Spa, dengan 129 entri atau 21,5% dari keseluruhan. Diikuti oleh Restaurant yang mencatat 124 entri (20,7%), dan Pool dengan 121 entri (20,2%). Fasilitas Free Wi-Fi juga cukup umum, dengan 118 entri (19,7%), sementara Gym menjadi fasilitas yang paling sedikit tersedia, dengan 108 entri (18,0%). Data ini menunjukkan bahwa Spa dan Restaurant adalah fasilitas yang paling diminati oleh hotel, yang dapat menjadi pertimbangan penting bagi pengelola hotel dalam meningkatkan daya tarik dan kepuasan tamu.  

### Numerical Features  

![Numerical Features](https://github.com/user-attachments/assets/fffab6da-b024-4d1c-8d57-a57b12c22f37)
![Numerical Features](https://github.com/user-attachments/assets/98b781e3-739a-4590-8a2f-07695b062f82)


**Insight:**  
- **Hotel Rating:** Sebagian besar hotel memiliki rating antara 3 hingga 5, dengan rating 4 menjadi yang paling umum. Ini menunjukkan bahwa banyak tamu cenderung memilih hotel dengan kualitas yang baik.  
- **Booking Time:** Waktu pemesanan bervariasi, dengan banyak pemesanan dilakukan dalam rentang 0 hingga 30 hari sebelum kedatangan. Ini menunjukkan bahwa tamu cenderung melakukan pemesanan dalam waktu dekat, mungkin karena fleksibilitas dalam perjalanan.  
- **Number of Nights:** Tamu cenderung menginap antara 1 hingga 14 malam, dengan frekuensi yang cukup merata. Ini menunjukkan bahwa ada variasi dalam durasi menginap, mungkin tergantung pada tujuan perjalanan.  
- **Number of Guests:** Jumlah tamu bervariasi, tetapi sebagian besar hotel tampaknya memiliki kapasitas untuk menampung lebih dari 2 tamu. Ini menunjukkan bahwa hotel mungkin lebih sering digunakan untuk kelompok atau keluarga.  
- **Price per Night:** Harga per malam menunjukkan distribusi yang bervariasi, dengan banyak hotel berada di kisaran harga 200 hingga 400. Ini menunjukkan adanya pilihan harga yang beragam, memungkinkan tamu untuk memilih sesuai anggaran mereka.  
- **Distance to Attractions:** Jarak ke atraksi bervariasi, dengan beberapa tamu memilih hotel yang lebih dekat ke pusat atraksi. Ini menunjukkan pentingnya lokasi dalam pemilihan hotel.


## Multivariate Analysis  
### Categorical Features  

![Rata-rata Harga](https://github.com/user-attachments/assets/4c151a81-a93d-40cc-9f01-40687a98c86d)


**Insight:**  

- **Room Type:** Rata-rata harga untuk berbagai jenis kamar (Single, Suite, Double) tampak cukup seragam. Ini menunjukkan bahwa tidak ada perbedaan harga yang signifikan antara jenis kamar yang berbeda. Hal ini bisa berarti bahwa hotel tersebut memiliki strategi harga yang konsisten untuk semua jenis kamar.  
  
- **Hotel Location:** Rata-rata harga untuk lokasi hotel (Countryside, City Center, Suburbs) juga menunjukkan pola yang serupa. Ini menunjukkan bahwa lokasi tidak terlalu mempengaruhi harga, atau hotel mungkin memiliki kebijakan harga yang seragam di semua lokasi.  
  
- **Season:** Terdapat perbedaan yang jelas antara harga di musim tinggi dan musim rendah. Harga di musim tinggi lebih tinggi dibandingkan dengan musim rendah, yang sesuai dengan ekspektasi umum bahwa permintaan akan akomodasi meningkat selama musim puncak.  
  
- **Hotel Facilities:** Rata-rata harga untuk fasilitas hotel (Pool, Restaurant, Spa, Gym, Free Wi-Fi) menunjukkan variasi yang lebih besar. Fasilitas seperti Spa dan Gym mungkin menarik lebih banyak tamu, sehingga harga bisa lebih tinggi. Namun, semua fasilitas tampak memiliki harga yang relatif seimbang, menunjukkan bahwa hotel mungkin menawarkan nilai yang baik untuk berbagai fasilitas.  

### Numerical Features  

![Fitur Numerik](https://github.com/user-attachments/assets/3e24f3b7-5a46-4a57-ab37-c898d64fe2f6)
![Fitur Numerik](https://github.com/user-attachments/assets/23d1a450-f668-4130-bc3c-731590e00482)



**Insight:**  

1. **Kualitas vs. Harga:**  
   Tamu cenderung memilih hotel dengan rating lebih tinggi meskipun harganya lebih mahal.  

2. **Durasi Menginap:**  
   Banyak tamu memilih untuk menginap dalam waktu singkat, tetapi ada juga yang merencanakan perjalanan jauh-jauh hari.  

3. **Variasi Harga:**  
   Harga per malam bervariasi secara signifikan, dan ini mungkin dipengaruhi oleh rating hotel dan waktu pemesanan.  

---  

### Correlation Matrix  
![Matriks Korelasi](https://github.com/user-attachments/assets/90474cc0-538b-45ff-b670-52d68b9438fb)
  
**Insight:**  

1. **Korelasi Positif yang Kuat:**  
   - **Number of Nights dan Price per Night** memiliki korelasi yang sangat tinggi (0.86), menunjukkan bahwa semakin banyak malam yang dipesan, semakin tinggi harga per malam yang dikenakan. Ini mungkin mencerminkan kebijakan harga yang lebih tinggi untuk pemesanan jangka panjang.  
   - **Number of Guests** juga menunjukkan korelasi positif yang signifikan dengan **Number of Nights (0.73)** dan **Price per Night (0.85)**, menunjukkan bahwa pemesanan untuk lebih banyak tamu cenderung berlangsung lebih lama dan dengan harga yang lebih tinggi.  

2. **Korelasi Negatif:**  
   - **Booking Time dan Distance Booking Ratio** memiliki korelasi negatif yang cukup kuat (-0.52), menunjukkan bahwa semakin jauh waktu pemesanan dari tanggal kedatangan, semakin kecil kemungkinan tamu memilih hotel yang jauh dari atraksi. Ini menunjukkan bahwa tamu lebih cenderung memesan hotel yang dekat dengan atraksi saat mereka melakukan pemesanan mendekati waktu kedatangan.  

3. **Korelasi Lemah:**  
   - Sebagian besar fitur lainnya menunjukkan korelasi yang lemah, dengan nilai di sekitar 0.1 hingga 0.3. Misalnya, **Hotel Rating dan Distance to Attractions** memiliki korelasi positif yang sangat rendah (0.05), menunjukkan bahwa tidak ada hubungan yang signifikan antara rating hotel dan jarak ke atraksi.  

4. **Interaksi Tamu dan Malam:**  
   - **Nights_Guests_Interaction** menunjukkan korelasi positif dengan **Number of Nights (0.63)** dan **Number of Guests (0.63)**, yang menunjukkan bahwa interaksi antara jumlah malam dan jumlah tamu berkontribusi pada pemesanan yang lebih lama dan lebih banyak tamu.

## Data Preparation
Sebelum memasukkan data ke model latih, empat langkah berikut akan dilakukan:  

1. **Encoding Fitur Kategorik:**  
   Encoding fitur kategorik dilaksanakan di beberapa fitur yang bertipe object. Hal ini dilakukan karena model machine learning hanya dapat menerima data dalam bentuk numerik. Untuk encoding fitur, akan digunakan `LabelEncoder`.  

2. **PCA (Principal Component Analysis):**  
   Reduksi dimensi akan dilakukan dengan menggunakan PCA pada dua fitur, yaitu **Number of Nights** dan **Nights_Guests_Interaction**. Teknik PCA digunakan untuk mengurangi jumlah fitur dengan cara mentransformasikan data ke dalam ruang vektor yang lebih rendah namun tetap mempertahankan informasi yang paling penting. Hal ini dapat membantu mengurangi kompleksitas model dan meningkatkan kecepatan pemrosesan.

3. **Train-Test-Split:**  
   Dataset akan dibagi menjadi data latih dan data uji dengan perbandingan 80:20, yaitu 80 persen data akan menjadi data latih dan 20 persen data akan menjadi data uji. Hal ini dilakukan supaya kita dapat melakukan validasi dengan benar tanpa bias dari model.  

4. **Standarisasi:**  
   Standarisasi menggunakan teknik `StandardScaler` dari library Scikit-learn. `StandardScaler` melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. `StandardScaler` menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Scaling ini dilaksanakan untuk membantu model machine learning yang akan dipakai lebih mudah diolah.

## Modelling
Pada tahap ini, model machine learning yang akan dipakai ada tiga algoritma. Lalu performa masing-masing algoritma akan dievaluasi untuk menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan digunakan, antara lain:  

1. **K-Nearest Neighbors (KNN):**  
   KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Untuk parameter yang akan digunakan yaitu `n_neighbors` dengan nilai sebesar 13.  
   - **Kelebihan:** Algoritma KNN mudah dipahami dan digunakan serta relatif sederhana dibandingkan dengan algoritma lain.  
   - **Kekurangan:** Jika dihadapkan pada jumlah fitur atau dimensi yang besar, performanya dapat menurun.  

2. **Random Forest:**  
   Algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak. Untuk parameter yang akan digunakan yaitu `n_estimators=100`, `max_depth=10`, `random_state=55`, `n_jobs=-1`.  
   - **Kelebihan:** Kuat terhadap data outlier (pencilan data), berjalan secara efisien pada kumpulan data yang besar, dan bekerja dengan baik dengan data non-linear.  
   - **Kekurangan:** Pembelajaran bisa berjalan lambat, tergantung pada parameter yang digunakan, dan tidak bisa memperbaiki model yang dihasilkan secara berulang.  

3. **Boosting Algorithm:**  
   Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. Untuk parameter yang akan digunakan yaitu `learning_rate=0.01`, `random_state=50`.  
   - **Kelebihan:** Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest.  
   - **Kekurangan:** Learning secara progresif dan sangat sensitif terhadap data noise dan outlier.
  
**Error dari masing-masing model:**

![image](https://github.com/user-attachments/assets/bd3657cc-c260-4e47-9219-52ab4b0fae27)

Pada Gambar diatas, dapat dilihat bahwa model Random Forest memiliki nilai error yang lebih kecil dibanding model lain, menandakan bahwa model Random Forest merupakan model terbaik yang dapat digunakan untuk memprediksi harga hotel.

## Evaluation

Metrik yang akan kita gunakan pada prediksi ini adalah **MSE** atau **Mean Squared Error**, yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut:  

MSE = (1/n) * Σ (y - ŷ)²  

Dimana:  

- \( y \) = Nilai Aktual permintaan  
- \( ŷ \) = Nilai hasil prediksi 
- \( n \) = Banyaknya data

**Hasil Evaluasi Model** 

Berikut adalah hasil evaluasi ketiga model menggunakan MSE pada tabel di bawah ini:  

| Model     | Train      | Test       |  
|-----------|------------|------------|  
| KNN       | 2.382581   | 2.712814   |  
| RF        | 0.054777   | 0.393011   |  
| Boosting  | 0.996689   | 1.39057    |

Pada tabel di atas, dapat dilihat bahwa model Random Forest memiliki nilai error yang lebih kecil dibanding model lain pada data latih dan data uji.  

### Hasil Prediksi Ketiga Model dari 10 Data  

| y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |  
|--------|---------------|-------------|--------------------|  
| 472    | 457.8        | 486.1      | 478.4              |  
| 296    | 251.5        | 315.0      | 320.6              |  
| 248    | 277.2        | 235.6      | 224.6              |  
| 257    | 286.9        | 282.4      | 321.0              |  
| 313    | 319.7        | 334.5      | 335.0              |  
| 158    | 260.2        | 161.8      | 174.5              |  
| 243    | 291.7        | 246.8      | 225.5              |  
| 212    | 252.1        | 200.5      | 178.0              |  
| 159    | 228.2        | 134.9      | 174.1              |  
| 426    | 356.3        | 391.8      | 391.8              |

### Analisis Hasil Prediksi  

Pada tabel di atas, dapat kita lihat bahwa prediksi dari ketiga algoritma yang paling mendekati `y_true` adalah prediksi Random Forest. Hal ini menandakan bahwa Random Forest merupakan algoritma terbaik dibandingkan dengan algoritma yang lain.  

## Kesimpulan  

Berdasarkan analisis yang dilakukan, dapat disimpulkan bahwa:  

1. **Model Prediktif untuk Penentuan Harga Hotel:**  
   - Model prediktif yang dikembangkan, terutama menggunakan algoritma Random Forest, terbukti efektif dalam membantu pihak hotel menentukan harga yang lebih akurat. Dengan memanfaatkan data historis dan fitur-fitur kunci seperti rating hotel, jumlah malam menginap, jumlah tamu, dan waktu pemesanan, model ini dapat memberikan estimasi harga yang lebih sesuai dengan nilai yang diterima tamu. Hal ini memungkinkan pihak hotel untuk menetapkan harga yang kompetitif dan adil, serta meningkatkan kepuasan tamu.  

2. **Pengaruh Faktor-Faktor terhadap Harga Hotel:**  
   - Analisis menunjukkan bahwa faktor-faktor seperti rating hotel, jumlah malam menginap, jumlah tamu, dan waktu pemesanan memiliki pengaruh signifikan terhadap harga hotel.   
     - **Rating Hotel:** Tamu cenderung memilih hotel dengan rating lebih tinggi, yang sering kali berhubungan dengan harga yang lebih tinggi.  
     - **Jumlah Malam Menginap:** Semakin banyak malam yang dipesan, semakin tinggi harga per malam yang dikenakan, mencerminkan kebijakan harga untuk pemesanan jangka panjang.  
     - **Jumlah Tamu:** Pemesanan untuk lebih banyak tamu cenderung berlangsung lebih lama dan dengan harga yang lebih tinggi, menunjukkan bahwa hotel sering digunakan untuk kelompok atau keluarga.  
     - **Waktu Pemesanan:** Tamu yang melakukan pemesanan mendekati waktu kedatangan cenderung memilih hotel yang lebih dekat dengan atraksi, yang juga dapat mempengaruhi harga.  

Dengan demikian, penerapan model prediktif ini tidak hanya memberikan keuntungan bagi pihak hotel dalam penetapan harga, tetapi juga membantu calon tamu dalam memahami nilai yang mereka dapatkan saat memesan hotel. Hal ini berpotensi meningkatkan pengalaman tamu dan mendorong loyalitas terhadap hotel.

## Referensi  
1. [Kaggle Hotel Booking Dataset](https://www.kaggle.com/datasets/mojtaba142/hotel-booking)  
2. [SoftQubes Blog](https://www.softqubes.com/blog/hotel-price-prediction-with-data-analysis-machine-learning/)




