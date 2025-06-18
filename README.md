# Laporan Proyek Machine Learning

## Domain Proyek: Credit Risk Prediction untuk Lembaga Keuangan

Dalam industri keuangan dan perbankan, pengelolaan risiko kredit merupakan aspek krusial dalam menjaga stabilitas operasional dan profitabilitas. Risiko kredit adalah potensi kerugian yang timbul ketika peminjam gagal memenuhi kewajiban pembayaran pinjamannya. Data dari Bank Dunia menunjukkan bahwa rata-rata tingkat kredit bermasalah secara global berkisar antara 3-5% dari total portofolio pinjaman, namun pada masa krisis ekonomi, angka ini bisa melonjak hingga 15-20%.

Sistem penilaian kredit tradisional seringkali bergantung pada rasio keuangan, sejarah kredit, dan penilaian manual yang memiliki keterbatasan dalam menangkap kompleksitas dan dinamika risiko peminjam modern. Hal ini mengakibatkan dua masalah utama: pertama, penolakan terhadap peminjam potensial yang sebenarnya layak (false negative) yang mengurangi peluang bisnis; dan kedua, persetujuan pinjaman kepada peminjam yang berisiko tinggi (false positive) yang meningkatkan kerugian kredit.

Oleh karena itu, diperlukan pendekatan yang lebih akurat dan efisien dalam menilai risiko kredit dengan memanfaatkan teknologi machine learning dan data analytics. Project ini akan membangun model prediktif untuk mengidentifikasi risiko kredit pada aplikasi pinjaman, sehingga dapat meningkatkan keputusan pemberian kredit yang lebih presisi dan optimal.

**Referensi:**
- World Bank. (2022). Global Financial Development Report: Bankers Without Borders.
- Federal Reserve. (2023). Consumer Credit - G.19 Report.
- Otoritas Jasa Keuangan. (2022). Statistik Perbankan Indonesia.

## Business Understanding

### Problem Statements

1. Bagaimana cara mengidentifikasi aplikasi pinjaman yang berisiko tinggi berdasarkan karakteristik peminjam dan struktur pinjaman?
2. Faktor-faktor apa saja yang paling signifikan memengaruhi status pembayaran kredit peminjam?
3. Bagaimana mengoptimalkan keputusan pemberian kredit untuk meminimalkan kerugian finansial dan memaksimalkan portofolio kredit yang sehat?

### Goals

1. Mengembangkan model prediktif yang dapat mengklasifikasikan aplikasi pinjaman menjadi kategori "good" (risiko rendah) atau "bad" (risiko tinggi) dengan akurasi yang tinggi
2. Mengidentifikasi faktor-faktor atau fitur yang paling berpengaruh dalam menentukan risiko kredit
3. Menyediakan solusi analitik yang dapat membantu pengambilan keputusan pemberian kredit yang lebih objektif dan konsisten

### Solution Statements

1. Melakukan eksplorasi dan analisis mendalam terhadap dataset pinjaman historis untuk memahami pola dan tren yang memengaruhi status kredit
2. Membangun dan membandingkan beberapa algoritma klasifikasi seperti `Logistic Regression`, `Decision Tree`, dan `Ada Boost` untuk memprediksi risiko kredit
3. Menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk mengatasi masalah ketidakseimbangan kelas pada data
4. Melakukan tuning hyperparameter untuk model terbaik guna meningkatkan performa prediksi
5. Menganalisis feature importance untuk mengidentifikasi faktor-faktor yang paling signifikan dalam prediksi risiko kredit

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah data historis pinjaman dari periode 2007-2014, yang mencakup informasi tentang peminjam, jenis pinjaman, dan status pembayaran kredit. Dataset ini bersumber dari institusi keuangan yang menyediakan layanan pinjaman peer-to-peer.

- Jumlah data: 466.285 baris
- Jumlah fitur awal: 74 kolom
- Jumlah fitur setelah praproses: 38 kolom (termasuk target)
- Tipe data: Gabungan antara data numerik dan kategorikal

### Variabel-variabel pada Dataset

Berikut adalah beberapa variabel penting dalam dataset:

| No. | Nama Kolom | Deskripsi |
|-----|------------|-----------|
| 1 | `loan_amnt` | Jumlah pinjaman yang diajukan peminjam |
| 2 | `term` | Jangka waktu pinjaman (36 atau 60 bulan) |
| 3 | `int_rate` | Tingkat bunga pinjaman |
| 4 | `installment` | Angsuran bulanan yang harus dibayar peminjam |
| 5 | `grade` | Tingkat (grade) pinjaman yang ditentukan oleh institusi |
| 6 | `emp_length` | Lama peminjam bekerja pada pekerjaan saat ini |
| 7 | `home_ownership` | Status kepemilikan rumah peminjam |
| 8 | `annual_inc` | Pendapatan tahunan peminjam |
| 9 | `verification_status` | Status verifikasi pendapatan peminjam |
| 10 | `purpose` | Tujuan pengajuan pinjaman |
| 11 | `dti` | Debt-to-Income ratio peminjam |
| 12 | `delinq_2yrs` | Jumlah keterlambatan 30+ hari dalam 2 tahun terakhir |
| 13 | `revol_util` | Persentase pemanfaatan kredit bergulir (revolving credit) |
| 14 | `total_acc` | Total jumlah rekening kredit peminjam |
| 15 | `loan_status` | Status pinjaman (target): 'good' atau 'bad' |

Dari hasil analisis awal, ditemukan bahwa beberapa kolom memiliki persentase nilai null yang tinggi (>60%) seperti `desc`, `mths_since_last_record`, `annual_inc_joint`, dan beberapa kolom lainnya yang akan dihapus pada tahap preprocessing data.

### Distribusi Target Variable (loan_status)

Analisis distribusi variabel target menunjukkan ketidakseimbangan kelas:
- Kelas 'good' (pinjaman lancar atau lunas): 88%
- Kelas 'bad' (pinjaman bermasalah): 12%

Ketidakseimbangan ini akan diatasi dengan teknik SMOTE pada tahap pra-pemodelan.

### Analisis Univariat

![Univariate Analysis.](/image/univariate.png "Univariate Analysis.")

1. **Term (Jangka Waktu)**:  Mayoritas pinjaman berdurasi 36 bulan,
 sementara 60 bulan lebih sedikit.

2. ** Employment Length (Lama Bekerja)**: >10 tahun pengalaman
 mendominasi, kategori lain lebih merata.

3. **Home Ownership (Kepemilikan Rumah)**: Sebagian besar peminjam
 RENT atau MORTGAGE, sedikit yang OWN.

4. **Verification Status**: Banyak peminjam tidak diverifikasi, sebagian terverifikasi.

5. **Loan Status (Status Pinjaman)**: Ketidakseimbangan 
data, mayoritas good, sedikit bad.

6. **Purpose (Tujuan)**: Didominasi debt_consolidation, tujuan
 lain jauh lebih sedikit.

7. ** Initial List Status**: Mayoritas f, hanya sedikit w.

### Analisis Multivariat

**Trem**
![Trem.](/image/term.png "Term.")
 Semakin lama tenor pinjaman, semakin tinggi proporsi peminjam dengan statu "bad", menunjukkan risiko gagal bayar lebih besar pada pinjaman jangka panjang.

**Purpose**
![Purpose.](/image/purpose.png "Purpose.")
 Pinjaman pendidikan dan usaha kecil berisiko gagal bayar tertinggi, sementara kendaraan dan kartu kredit lebih rendah.

**Grade**
![Grade.](/image/grade.png "Grade.")
 Grade rendah (F dan G) berisiko tinggi gagal bayar (>40%), sementara grade A dan B lebih aman dengan risiko minimal.

**Verification Status**
![VerStatus.](/image/VerStatus.png "VerStatus.")
 Klien dengan status verifikasi "Not Verified" memiliki risiko gagal bayar tertinggi (20,19%), sedangkan "Source Verified" paling rendah (16%), menunjukkan bahwa verifikasi meningkatkan kualitas kredit

**EMP Length**
![EMPLength.](/image/EMPLenght.png "EMPLength.")
 Semakin lama masa kerja, semakin rendah risiko gagal bayar. Klien dengan masa kerja <1 tahun memiliki tingkat gagal bayar tertinggi (21,42%), menunjukkan bahwa stabilitas berkontribusi pada kesehatan kredit.

**Pendapatan Tahunan**
![DistPendapatan.](/image/DistPendapatan.png "DistPendapatan.")
![SegPendapatan.](/image/SegPendapatan.png "SegPendapatan.")
 Klien dengan pendapatan lebih tinggi cenderung memiliki kelayakan kredit yang lebih baik, ditunjukkan oleh rata-rata pendapatan $71.051 pada kredit baik dibandingkan $65.562 pada kredit buruk. Risiko gagal bayar menurun seiring meningkatnya pendapatan, dengan kredit macet terendah pada klien berpenghasilan tinggi (15,24%) dan tertinggi pada yang berpenghasilan rendah (21,72%).
 
 **Debt To Income Ratio**
![DistDebt.](/image/DistDebt.png "DistDebt.")
![SegtDebt.](/image/SegDebt.png "SegDebt.")
 Rasio Debt-to-Income (DTI) yang lebih tinggi berkorelasi dengan meningkatnya risiko gagal bayar. Pinjaman bermasalah memiliki rata-rata DTI sebesar 14.38, lebih tinggi dibandingkan pinjaman lancar yang hanya 13.75. Selain itu, nasabah dengan DTI rendah memiliki tingkat pinjaman bermasalah lebih rendah (18.04%) dibandingkan nasabah dengan DTI menengah (21.03%), menguatkan bahwa semakin tinggi DTI, semakin besar risiko gagal bayar.

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk mempersiapkan data agar siap digunakan untuk pemodelan. Berikut adalah langkah-langkah yang dilakukan:

### 1. Impute Data
 Mengisi null value dengan nilai median untuk feature numerikal dan mode untuk feature kategorikal

### 2. Feature Engineering
 Menciptakan Feature baru dari feature yang ada sebelumnya untuk menemukan pola baru pada data

### 3. Encoding Values
 Ubah data kategorikal menjadi numerik: gunakan Label Encoding untuk data ordinal dan One-Hot Encoding untuk data tanpa urutan.

### 4. Transformation Values
 Mengubah skala nilai pada setiap feature agar memiliki ukuran yang serempak

### 5. Spliting Data & Data Balancing 
 Membagi dataset menjadi data train dan test, setelah itu menyeimbangkan data antara data minorirtas dan mayoritas agar model dapat dilatih secara optimal

## Modeling

Pada proyek ini, dilakukan perbandingan beberapa algoritma klasifikasi untuk memprediksi risiko kredit. Berikut adalah model-model yang digunakan:

### 1. Logistic Regression

Logistic Regression adalah algoritma klasifikasi yang memodelkan probabilitas kelas target menggunakan fungsi logistik sigmoid. Model ini cocok untuk masalah klasifikasi biner dan memberikan output probabilitas yang dapat diinterpretasikan.

**Kelebihan:**
- Interpretabilitas tinggi, koefisien model menunjukkan pentingnya setiap fitur
- Pemodelan yang efisien dengan kompleksitas komputasi rendah
- Memberikan probabilitas yang terkalibrasi dengan baik

**Kekurangan:**
- Asumsi hubungan linear antara fitur dan log-odds
- Performa mungkin tidak sebaik model kompleks pada data non-linear
- Sensitif terhadap multikolinearitas

### 2. Decision Tree

Decision Tree adalah algoritma pembelajaran mesin non-parametrik yang membagi data menjadi subset berdasarkan nilai fitur, menciptakan struktur pohon keputusan. Model ini dapat menangkap hubungan non-linear dalam data.

**Kelebihan:**
- Mampu menangkap pola non-linear
- Interpretabilitas visual melalui struktur pohon
- Tidak memerlukan scaling fitur

**Kekurangan:**
- Risiko overfitting tinggi, terutama pada pohon yang dalam
- Kurang stabil, perubahan kecil pada data dapat menghasilkan pohon yang sangat berbeda
- Bisa bias terhadap fitur dengan banyak nilai unik

### 3. Ada Boost

AdaBoost adalah teknik ensemble yang menggabungkan beberapa weak learners (biasanya Decision Tree sederhana) menjadi model yang kuat. AdaBoost bekerja dengan memberikan bobot lebih pada data yang salah diklasifikasikan pada iterasi sebelumnya.

**Kelebihan:**
- Mampu mengurangi bias dan variance
- Cenderung tidak overfitting pada dataset besar
- Performa baik pada berbagai jenis data

**Kekurangan:**
- Sensitif terhadap noise dan outlier
- Komputasi lebih intensif dibandingkan model tunggal
- Parameter boosting perlu dituning dengan hati-hati


## Evaluation

Untuk mengevaluasi kinerja model dalam memprediksi risiko kredit, digunakan berbagai metrik evaluasi yang relevan dengan masalah klasifikasi yang memiliki ketidakseimbangan kelas. Berikut adalah metrik-metrik yang digunakan:

### Metrik Evaluasi

1. Accuracy: Persentase prediksi benar dari total prediksi; kurang efektif untuk data tidak seimbang.
2. Precision: Proporsi prediksi positif yang benar; penting untuk menghindari false positive.
3. Recall: Proporsi kasus positif yang berhasil dikenali; penting untuk menangkap semua risiko.
4. F1-Score: Rata-rata harmonik precision dan recall; cocok saat keduanya sama penting.
5. ROC AUC: Mengukur kemampuan model membedakan kelas; semakin tinggi, semakin baik performa

 **Fokus utama evaluasi dalam proyek ini berada pada F1-Score**, karena data yang tidak seimbang membutuhkan keseimbangan antara precision dan recall untuk memastikan model tidak hanya akurat dalam mengenali pinjaman berisiko tinggi, tetapi juga tidak melewatkan terlalu banyak kasus penting.

### Hasil Evaluasi Model
#### Perbandingan Model dengan SMOTE
![WithSMOTE.](/image/WithSMOTE.png "WithSMOTE.")

#### Perbandingan Model tanpa SMOTE
![WithoutSMOTE.](/image/WithoutSMOTE.png "WithoutSMOTE.")

### Kesimpulan
 Dari beberapa model yang diuji, Logistic Regression dengan SMOTE menunjukkan performa terbaik dengan F1-score 96%, mengungguli Decision Tree yang mengalami overfitting dan AdaBoost yang memiliki skor lebih rendah. Oleh karena itu, Logistic Regression dengan SMOTE di implementasikan sebagai model terbaik untuk analisis kelayakan pinjaman.
 
## Hyperparameter Tuning
Setelah menentukan algoritma terbaik yaitu **Logistic Regression**, dilakukan **Hyperparameter Tuning** untuk memaksimalkan performa model. Parameter yang diuji meliputi:

* **C**: Mengatur kekuatan regularisasi (semakin kecil, semakin kuat).
* **Penalty**: Jenis regularisasi (`l1`, `l2`, `elasticnet`, `None`) untuk mencegah overfitting.
* **Solver**: Algoritma optimasi (`liblinear`, `lbfgs`, `saga`).
* **Max Iter**: Jumlah iterasi maksimum (100–1000) untuk memastikan konvergensi model.

| Metric        | Train | Test |
|---------------|-------|------|
| Accuracy      | 0.91  | 0.93 |
| Precision     | 0.88  | 0.98 |
| Recall        | 0.95  | 0.95 |
| F1-Score      | 0.91  | 0.96 |
| ROC AUC       | 0.91  | 0.90 |


## Kesimpulan

Model yang dibangun memiliki F1-score sebesar 96% dan Recall sebesar 95%, yang menunjukkan kemampuan kuat dalam membedakan antara Good Loan dan Bad Loan. Dengan keseimbangan yang baik antara precision dan recall, model ini mampu mengidentifikasi risiko peminjam secara lebih akurat dan efisien.

Tanpa menggunakan machine learning, bank hanya mampu memperoleh 88% pinjaman berstatus Good. Namun, dengan penerapan machine learning, bank dapat meningkatkan identifikasi pinjaman berstatus Good hingga 95%, atau mengalami peningkatan sebesar 7% dalam akurasi kelayakan pinjaman.
