# Nama : Ahmad Raihan
# Modul : Belajar Penerapan Data Science
# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Struktur Direktori
Adapun struktur direktori yang digunakan pada submission ini adalah:
  1. app – Berisi file model yang telah dilatih untuk memprediksi status mahasiswa graduate, enrolled, atau dropout.
  2. notebook – Berisi notebook (Jupyter Notebook) yang memuat seluruh proses analisis dan pemodelan.
  3. prediction – Berisi skrip yang digunakan untuk melakukan prediksi terhadap data baru.
  4. README – File markdown yang menjelaskan deskripsi proyek, alur pengerjaan, hingga kesimpulan akhir.
  5. raihan-dashboard – Berisi dokumentasi berupa kumpulan screenshot dashboard yang telah dibuat.
  6. raihan-video – Berisi video penjelasan terkait dashboard serta rekomendasi bisnis yang diberikan.
  7. metabase.db.mv.db – File database Metabase yang menyimpan konfigurasi dashboard interaktif yang telah dikembangkan.
  8. requirements.txt – Berisi daftar pustaka dan dependensi yang dibutuhkan untuk menjalankan proyek.

## 1. Business Understanding
![istockphoto-1335901917-612x612](https://github.com/user-attachments/assets/c940d59a-7763-4f6b-99ed-d210b335b0cc)
### 1.1 Latar Belakang
Jaya Jaya Institut merupakan sebuah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000. Selama lebih dari dua dekade, institusi ini telah menghasilkan banyak lulusan berkualitas dengan reputasi yang sangat baik. Namun demikian, Jaya Jaya Institut juga menghadapi tantangan serius terkait tingginya angka siswa yang tidak menyelesaikan pendidikannya (dropout).
Tingginya angka dropout tentu menjadi perhatian utama bagi sebuah institusi pendidikan, karena dapat berdampak pada reputasi dan kualitas pendidikan yang diberikan. Untuk itu, pihak institusi ingin mendeteksi secara dini siswa yang berpotensi mengalami dropout, sehingga dapat diberikan bimbingan atau intervensi yang tepat guna meningkatkan peluang mereka untuk menyelesaikan studi.

### 1.1 Permasalahan Bisnis
Berdasarkan latar belakang yang telah dijelaskan, berikut adalah tiga permasalahan utama yang ingin dijawab dalam proyek ini:
1. Apa saja faktor yang memengaruhi siswa untuk berhenti atau tidak menyelesaikan pendidikannya (dropout)?
2. Bagaimana membangun model prediktif yang mampu mengidentifikasi siswa yang berisiko tinggi mengalami dropout?
3. Bagaimana menyajikan hasil analisis dan prediksi tersebut dalam bentuk dashboard yang informatif dan mudah dipahami?

### 1.2 Cakupan Proyek
Proyek ini memiliki cakupan proyek sebagai berikut:
1. Mengidentifikasi faktor-faktor utama yang memengaruhi siswa untuk berhenti atau tidak menyelesaikan pendidikannya (dropout).
2. Membangun model prediktif berbasis machine learning yang mampu mengidentifikasi siswa yang berisiko tinggi mengalami dropout.
3. Menyediakan business dashboard yang dapat membantu tim institusi pendidikan dalam memantau status mahasiswa.

### 1.3 Solution Statements
- Berikut merupakan tahapan-tahapan yang dilakukan dalam menyelesaikan proyek ini:

  1. **Exploratory Data Analysis (EDA)**: Melakukan eksplorasi dan pemahaman mendalam terhadap data melalui visualisasi dan analisis statistik untuk mengidentifikasi pola serta fitur-fitur yang mungkin berkaitan erat dengan status mahasiswa.

  2. **Data Preprocessing**: Melakukan serangkaian proses pembersihan dan transformasi data, termasuk perhitungan korelasi antar fitur, encoding variabel kategorikal, penyeimbangan data menggunakan teknik SMOTE, serta seleksi fitur untuk meningkatkan performa model.

  3. **Modeling dan Evaluasi**: Menerapkan berbagai algoritma machine learning seperti Random Forest, Gradient Boosting, Xgboost untuk membangun model prediktif. Evaluasi dilakukan menggunakan metrik seperti Accuracy, Precision, Recall, dan F1-Score guna menentukan model dengan kinerja terbaik.

  4. **Prediksi**: Menggunakan model terbaik yang telah terpilih untuk melakukan prediksi terhadap status mahasiswa yang graduate, enrolled, atau dropout, dengan tujuan mendukung pengambilan keputusan yang proaktif oleh tim institusi pendidikan.

  5. **Pembuatan Dashboard**: Membangun dashboard yang informatif dan mudah dipahami sebagai media visualisasi hasil analisis dan prediksi, sehingga dapat digunakan sebagai alat bantu oleh tim institusi pendidikan.

## 2. Persiapan
### 2.1 **Dataset**
Dataset yang digunakan dalam proyek ini adalah **[Dataset Karyawan Jaya Jaya Maju](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)**, yang disediakan sesuai dengan instruksi pada submission proyek ini.

## 3. Tahapan Pengerjaan
### 3.1 Membuka notebook.ipynb
- Pastikan seluruh **dependensi** telah terinstal sesuai dengan daftar pada **`requirements.txt`**.
- Jalankan seluruh isi **notebook.ipynb** di **Google Colab** atau IDE sejenis untuk melihat hasil dari **analisis data**, temuan, dan **insight** yang diperoleh.

### 2. Menjalankan app.py
- Pastikan seluruh **dependensi** telah terinstal sesuai dengan daftar pada **`requirements.txt`**.
- File `prediction.py` dapat dijalankan secara langsung menggunakan **VSCode** atau IDE lokal lain yang mendukung Python.

- Script ini memuat:
  - **Fungsi prediction** (`predict_status`) untuk melakukan proses encoding, normalisasi, selektsi fitur sebelum prediksi.
  - **Model yang telah dilatih** (`model.joblib`) yang dimuat menggunakan `joblib`.
  - **Feature Encoding** (`encoding.joblib`) yang dimuat menggunakan  `joblib`.
  - **Feature Scaling** (`scaler.joblib`) yang dimuat menggunakan  `joblib`.
  - **Feature Selection** (`selector.joblib`) yang dimuat menggunakan `joblib`.
  - **Antarmuka input** menggunakan Streamlit, di mana pengguna dapat memasukkan data melalui form interaktif.
  - **Prediksi** terhadap data tersebut untuk menentukan apakah status mahasiswa (`graduate, enrolled, atau dropout`).

- Untuk menjalankan prediksi:
  1. Pastikan file `model.joblib`, `encoding.joblib`, `scaler.joblib`, `selector.joblib` berada dalam direktori yang sama.
  2. Buka terminal dan arahkan ke folder tempat file disimpan.
  3. Jalankan perintah berikut:
     `streamlit run app.py`
  5. Aplikasi akan terbuka di browser secara otomatis. Masukkan data pada form yang tersedia.
  6. Hasil prediksi akan ditampilkan di halaman aplikasi.

#### **3. Menjalankan Dashboard**
Untuk mengakses **dashboard** secara lokal, Anda dapat menjalankan **Metabase** menggunakan **Docker**. Pastikan aplikasi **Docker** sudah terinstal di perangkat Anda.

**Langkah-langkah untuk menjalankan Metabase menggunakan Docker**:
1. **pull image Metabase dari Docker Hub** dengan perintah:
   ```bash
   docker pull metabase/metabase:latest
   ```
   
2. **Jalankan container Metabase** dengan perintah:
   ```bash
   docker run -p 3000:3000 --name metabase metabase/metabase
   ```

3. **Login ke Metabase dengan url : http://localhost:3000/setup** menggunakan kredensial berikut:
   - **Username**: `root@mail.com`
   - **Password**: `root123`

## 4. Business Dashboard
- Berdasarkan hasil **analisis dari Dasboard** dan modeling menggunakan **Random Forest**, berikut adalah 15 fitur yang paling berpengaruh terhadap keputusan karyawan untuk keluar dari perusahaan:
1. **Curricular_units_2nd_sem_approved**  
   Mahasiswa yang menyelesaikan lebih banyak mata kuliah di semester kedua memiliki kecenderungan besar untuk lulus.  
   EDA menunjukkan bahwa mahasiswa graduated secara konsisten memiliki jumlah mata kuliah approved yang tinggi di semester dua.  
   *Skor importance: 0.167*

2. **Curricular_units_2nd_sem_grade**  
   Kualitas nilai di semester kedua sangat memengaruhi outcome akademik.  
   EDA menunjukkan bahwa mahasiswa graduated memiliki rata-rata nilai semester dua yang lebih tinggi.  
   *Skor importance: 0.110*

3. **Curricular_units_1st_sem_approved**  
   Performa sejak awal kuliah sangat berpengaruh; mahasiswa yang gagal menyelesaikan cukup mata kuliah semester pertama lebih rentan untuk dropout.  
   EDA memperlihatkan pola bahwa jumlah approve rendah di semester pertama dominan pada mahasiswa dropout.  
   *Skor importance: 0.105*

4. **Curricular_units_1st_sem_grade**  
   Nilai semester pertama menjadi sinyal awal keberhasilan atau risiko kegagalan.  
   EDA memperlihatkan mahasiswa dengan nilai rendah di awal cenderung tidak menyelesaikan studi.  
   *Skor importance: 0.083*

5. **Admission_grade**  
   Nilai masuk berkorelasi dengan kemampuan akademik dan kesiapan mahasiswa.  
   EDA menunjukkan mahasiswa dengan nilai masuk tinggi lebih cenderung menyelesaikan studi hingga graduated.  
   *Skor importance: 0.079*

6. **Curricular_units_2nd_sem_evaluations**  
   Mahasiswa yang aktif dalam evaluasi cenderung lebih terlibat dan memiliki outcome lebih baik.  
   EDA memperlihatkan bahwa mahasiswa dropout cenderung memiliki jumlah evaluasi yang sangat rendah.  
   *Skor importance: 0.078*

7. **Age_at_enrollment**  
   Usia saat mendaftar memengaruhi kesiapan dan adaptasi akademik.  
   EDA menunjukkan bahwa mahasiswa yang lebih tua cenderung memiliki status enrolled lebih lama atau bahkan dropout.  
   *Skor importance: 0.066*

8. **Curricular_units_1st_sem_evaluations**  
   Partisipasi dalam evaluasi semester pertama menunjukkan keterlibatan awal dalam proses akademik.  
   EDA mengindikasikan mahasiswa dengan jumlah evaluasi tinggi di semester pertama lebih cenderung bertahan dan graduated.  
   *Skor importance: 0.062*

9. **Tuition_fees_up_to_date**  
   Mahasiswa yang rajin membayar uang kuliah tepat waktu lebih cenderung untuk lulus.  
   EDA menunjukkan bahwa tunggakan pembayaran sangat umum ditemukan pada mahasiswa yang dropout.  
   *Skor importance: 0.053*

10. **Application_mode**  
    Jalur masuk mempengaruhi performa dan adaptasi mahasiswa terhadap lingkungan akademik.  
    EDA memperlihatkan jalur masuk tertentu seperti "Over 23 years old" memiliki pola kelulusan yang berbeda.  
    *Skor importance: 0.043*

11. **Curricular_units_2nd_sem_enrolled**  
    Jumlah mata kuliah yang diambil di semester kedua menunjukkan tingkat komitmen dan keberlanjutan studi.  
    EDA memperlihatkan bahwa mahasiswa yang mengambil sedikit mata kuliah di semester dua cenderung dropout.  
    *Skor importance: 0.041*

12. **Curricular_units_1st_sem_enrolled**  
    Jumlah mata kuliah yang diambil di awal studi menunjukkan rencana akademik dan beban studi.  
    EDA menunjukkan pola bahwa mahasiswa graduated cenderung mengambil jumlah mata kuliah optimal di semester pertama.  
    *Skor importance: 0.037*

13. **Scholarship_holder**  
    Penerima beasiswa cenderung lebih termotivasi dan memiliki dukungan finansial untuk menyelesaikan studi.  
    EDA menunjukkan proporsi penerima beasiswa lebih tinggi pada mahasiswa graduated.  
    *Skor importance: 0.031*

14. **Gender**  
    Jenis kelamin memiliki pengaruh kecil, namun terlihat perbedaan pola akademik.  
    EDA mengindikasikan mahasiswa laki-laki sedikit lebih rentan dropout dibanding perempuan.  
    *Skor importance: 0.024*

15. **Debtor**  
    Mahasiswa yang memiliki utang pendidikan berisiko lebih tinggi untuk dropout.  
    EDA menunjukkan bahwa banyak mahasiswa dengan status dropout adalah mereka yang belum menyelesaikan kewajiban pembayaran.  
    *Skor importance: 0.021*

![image](https://github.com/user-attachments/assets/4b0320df-a0bb-4190-a83b-60f8b9a306f6)
![image](https://github.com/user-attachments/assets/e588ce82-8585-4389-b20a-40e0c7326a0d)
![image](https://github.com/user-attachments/assets/d55399cd-467b-4b04-9636-04b148d7fa26)

## 5. Hasil Model Prediktif
- Berdasarkan hasil evaluasi menggunakan metrik accuracy, precision, recall, f1-score, serta analisis melalui confusion matrix, dapat disimpulkan bahwa **model Random Forest memiliki performa terbaik di antara model Decision Tree dan Gradient Boosting untuk dataset yang tidak seimbang**.

- Hal ini ditunjukkan oleh nilai evaluasi berikut:
  - Accuracy: 76.16%
  - Precision: 75.73%
  - Recall: 76.16%
  - F1-Score: 75.67%

| Model               | Accuracy | Precision | Recall  | F1 Score |
|---------------------|----------|-----------|---------|----------|
| XGBoost            | 0.7390   | 0.7615    | 0.7390  | 0.7426   |
| Random Forest  | 0.7616   | 0.7573    | 0.7616  | 0.7567   |
| Gradient Boosting  | 0.7503 | 0.7531    | 0.7503  | 0.7491   |

### 6. Conclusion
- Dengan memahami faktor-faktor utama yang memengaruhi status mahasiswa—apakah lulus (graduate), masih aktif (enrolled), atau berhenti studi (dropout) serta menerapkan model prediktif yang efektif, institusi pendidikan dapat mengambil langkah proaktif untuk meningkatkan jumlah kelulusan mahasiswa dan mengurangi mahasiswa yang dropout.

### 7. Rekomendasi Action (Optional)
- Adapun beberapa rekomendasi yang dapat diberikan agar dapat mengurangi potensi attrition pada karyawan adalah sebagai berikut:
1. **Memperhatikan performa akademik mahasiswa**, terutama dari jumlah mata kuliah yang disetujui dan nilai di semester 1 dan 2. Mahasiswa dengan nilai rendah perlu dipantau lebih awal dan diberikan pendampingan seperti bimbingan belajar dan remedial.

2. **Meningkatkan keterlibatan mahasiswa dalam aktivitas akademik**, seperti evaluasi, kuis, dan tugas. Partisipasi rendah menunjukkan risiko dropout, sehingga perlu dipantau secara rutin.

3. **Memberikan dukungan khusus untuk mahasiswa baru**, terutama di semester pertama. Pendampingan adaptasi akademik sangat penting agar mahasiswa tidak langsung tertinggal sejak awal.

4. **Mempermudah akses ke dukungan finansial**, termasuk beasiswa dan skema pembayaran yang fleksibel. Mahasiswa yang tidak membayar uang kuliah tepat waktu atau memiliki utang memiliki risiko dropout yang lebih tinggi.

5. **Menawarkan Pembelajaran Khusus untuk Mahasiswa Tertentu**, seperti mahasiswa yang berusia lebih tua agar tetap bisa menyelesaikan studi dengan komitmen yang sesuai.
   
Rekomendasi ini ditargetkan pada faktor-faktor dengan pengaruh terbesar dalam model prediksi dan diperkuat oleh hasil EDA.
