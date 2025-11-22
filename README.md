# Analisis Rantai Markov untuk Perpindahan Merek Laptop Mahasiswa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

Proyek ini menyajikan analisis mendalam Rantai Markov berdasarkan data survei mahasiswa tentang perpindahan merek laptop. Menggunakan metodologi standar dalam literatur akademik untuk memodelkan perilaku konsumen dan memprediksi pangsa pasar jangka panjang.

## 📋 Daftar Isi

- [Fitur](#-fitur)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Metodologi](#-metodologi)
- [Hasil Analisis](#-hasil-analisis)
- [Kontribusi](#-kontribusi)
- [Lisensi](#-lisensi)
- [Referensi](#-referensi)

## ✨ Fitur

- Identifikasi ruang keadaan dari data survei
- Konstruksi matriks transisi empiris
- Visualisasi diagram transisi dengan NetworkX
- Analisis probabilitas langkah ke-n
- Perhitungan distribusi stasioner
- Klasifikasi keadaan (absorbing, recurrent, transient)
- Simulasi jalur Markov
- Ekspor hasil ke CSV dan PNG
- Notebook Jupyter interaktif
- Dokumentasi akademik lengkap

## 🚀 Instalasi

### Prasyarat

- Python 3.8 atau lebih baru dan/atau R 4.0 atau lebih baru
- pip (untuk Python) dan CRAN (untuk R)
- Jupyter Notebook (untuk Python) atau RStudio (untuk R Markdown)

### Langkah Instalasi

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/EgiStr/analisis-rantai-markov-perpindahan-merek-laptop.git
   cd analisis-rantai-markov-perpindahan-merek-laptop
   ```

2. **Untuk Python:**
   - Buat virtual environment (opsional tapi direkomendasikan):
     ```bash
     python -m venv venv
     source venv/bin/activate  # di Linux/Mac
     # atau
     venv\Scripts\activate     # di Windows
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

3. **Untuk R:**
   - Install packages R:
     ```r
     install.packages(c("tidyverse", "markovchain", "igraph", "expm", "ggplot2", "rmarkdown"), repos = "https://cran.rstudio.com/")
     ```

4. **Jalankan Jupyter Notebook atau RStudio:**
   ```bash
   jupyter notebook  # untuk Python
   # atau buka RStudio untuk R Markdown
   ```

## 📖 Penggunaan

### Menggunakan Notebook Jupyter (Python)

1. Buka `Python/markov_analysis_notebook.ipynb`
2. Jalankan sel-sel secara berurutan
3. Hasil analisis akan tersimpan di folder `output/`

### Menggunakan Script Python

```bash
python Python/markov_analysis.py
```

### Menggunakan R Markdown (R)

1. Buka `R/markov_analysis.Rmd` di RStudio
2. Klik "Knit" untuk menghasilkan HTML atau PDF
3. Atau jalankan dari command line:
   ```bash
   Rscript -e "rmarkdown::render('R/markov_analysis.Rmd', output_format = 'html_document')"
   ```

### Menggunakan Script R

```bash
Rscript R/markov_analysis.R
```

### File Output

- `output/transition_matrix.csv` & `output/transition_matrix_R.csv` - Matriks transisi
- `output/stationary_distribution.csv` & `output/stationary_distribution_R.csv` - Distribusi stasioner
- `output/state_classification.csv` & `output/state_classification_R.csv` - Klasifikasi keadaan
- `output/*.png` - Visualisasi grafik (Python dan R)

## 📁 Struktur Proyek

```
analisis-rantai-markov-perpindahan-merek-laptop/
├── data/
│   └── datapemstok.csv          # Data survei mahasiswa
├── docs/                        # Dokumentasi tambahan
├── output/                      # Hasil analisis (dihasilkan)
│   ├── transition_matrix.csv
│   ├── stationary_distribution.csv
│   ├── state_classification.csv
│   └── *.png                    # Grafik visualisasi
├── Python/
│   ├── markov_analysis.py       # Script analisis Python
│   └── markov_analysis_notebook.ipynb  # Notebook Jupyter Python
├── R/
│   ├── markov_analysis.R        # Script analisis R
│   └── markov_analysis.Rmd      # R Markdown untuk laporan
├── .gitignore                   # File yang diabaikan Git
├── CODE_OF_CONDUCT.md           # Kode etik
├── CONTRIBUTING.md              # Panduan kontribusi
├── LICENSE                      # Lisensi MIT
├── README.md                    # Dokumentasi ini
└── requirements.txt             # Dependencies Python
```

## 🔬 Metodologi

### 1. Identifikasi Ruang Keadaan
Ruang keadaan terdiri dari merek laptop unik yang muncul dalam data survei.

### 2. Matriks Transisi
Matriks \( P \) berukuran \( n \times n \) di mana \( p_{ij} \) adalah probabilitas transisi dari keadaan \( i \) ke \( j \).

### 3. Diagram Transisi
Graf berarah yang menunjukkan aliran perpindahan antar merek.

### 4. Probabilitas Langkah ke-n
Perhitungan \( P^{(n)} = P^n \) untuk memprediksi distribusi setelah n langkah.

### 5. Distribusi Stasioner
Solusi persamaan \( \pi P = \pi \) menggunakan eigenvector dominan.

### 6. Klasifikasi Keadaan
- **Absorbing**: Probabilitas diri sendiri = 1
- **Recurrent**: Probabilitas stasioner > 0
- **Transient**: Keadaan lainnya

## 📊 Hasil Analisis

### Ruang Keadaan
9 merek laptop: Acer, Apple, Asus, Axioo, Dell, Fujitsu, Hp, Lenovo, Toshiba

### Distribusi Stasioner
- Axioo: 1.0000 (merek dominan jangka panjang)
- Keadaan lain: 0.0000

### Klasifikasi Keadaan
- **Absorbing**: Apple, Axioo, Dell
- **Transient**: Acer, Asus, Fujitsu, Hp, Lenovo, Toshiba

### Temuan Utama
1. Axioo adalah merek dengan loyalitas tertinggi dalam jangka panjang
2. Rantai Markov bersifat absorbing dengan 3 komponen terpisah
3. Mayoritas merek menunjukkan perilaku transient

## 🤝 Kontribusi

Kami menyambut kontribusi! Silakan ikuti langkah-langkah berikut:

1. Fork repository ini
2. Buat branch fitur baru (`git checkout -b fitur/NamaFitur`)
3. Commit perubahan (`git commit -m 'Tambahkan fitur baru'`)
4. Push ke branch (`git push origin fitur/NamaFitur`)
5. Buat Pull Request

Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk panduan detail.

## 📄 Lisensi

Proyek ini dilisensikan di bawah [Lisensi MIT](LICENSE) - lihat file LICENSE untuk detail.

## 📚 Referensi

1. Howard, R. A. (1971). *Dynamic Probabilistic Systems: Markov Models*. Wiley.
2. Norris, J. R. (1997). *Markov Chains*. Cambridge University Press.
3. Ross, S. M. (2014). *Introduction to Probability Models*. Academic Press.
4. Dokumentasi SciPy: https://docs.scipy.org/
5. Dokumentasi NetworkX: https://networkx.org/


## 🙏 Ucapan Terima Kasih

Terima kasih kepada semua kontributor dan komunitas open source yang membuat proyek ini mungkin.