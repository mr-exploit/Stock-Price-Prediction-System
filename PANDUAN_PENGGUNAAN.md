# Panduan Penggunaan Sistem Prediksi Harga Saham

Dokumen ini berisi panduan langkah demi langkah untuk menjalankan sistem prediksi harga saham.

## Prasyarat

Sebelum menjalankan sistem, pastikan komputer Anda sudah terinstal:

- **Python 3.8** atau lebih tinggi
- **pip** (package manager Python)

## Langkah 1: Persiapan Environment

### 1.1 Clone Repository (jika belum)

```bash
git clone https://github.com/mr-exploit/Stock-Price-Prediction-System.git
cd Stock-Price-Prediction-System
```

### 1.2 Buat Virtual Environment (Disarankan)

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Di Linux/Mac:
source venv/bin/activate

# Di Windows:
venv\Scripts\activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

> **Catatan**: Proses instalasi mungkin memakan waktu beberapa menit karena ada beberapa library besar seperti TensorFlow.

## Langkah 2: Training Model (Melatih Model)

Sebelum melakukan prediksi, Anda perlu melatih model terlebih dahulu menggunakan data historis.

### 2.1 Training dengan Model LSTM (Rekomendasi)

```bash
python main.py --mode train --ticker AAPL --model lstm --epochs 100
```

Parameter:
- `--ticker`: Kode saham yang ingin diprediksi (contoh: AAPL untuk Apple, TSLA untuk Tesla)
- `--model`: Jenis model (`lstm`, `rf`, atau `arima`)
- `--epochs`: Jumlah iterasi training (semakin banyak biasanya semakin akurat, tapi lebih lama)

### 2.2 Contoh Training untuk Berbagai Saham

```bash
# Saham Apple
python main.py --mode train --ticker AAPL --model lstm --epochs 100

# Saham Tesla
python main.py --mode train --ticker TSLA --model lstm --epochs 100

# Saham Microsoft
python main.py --mode train --ticker MSFT --model lstm --epochs 100

# Saham Google
python main.py --mode train --ticker GOOGL --model lstm --epochs 100

# Saham Amazon
python main.py --mode train --ticker AMZN --model lstm --epochs 100
```

> **Tips**: Training pertama kali mungkin memakan waktu 5-30 menit tergantung jumlah epochs dan spesifikasi komputer.

## Langkah 3: Prediksi Harga Saham

Setelah model terlatih, Anda dapat melakukan prediksi harga untuk beberapa hari ke depan.

### 3.1 Prediksi 30 Hari ke Depan

```bash
python main.py --mode predict --ticker AAPL --days 30
```

Parameter:
- `--ticker`: Kode saham yang ingin diprediksi
- `--days`: Jumlah hari prediksi ke depan

### 3.2 Contoh Prediksi

```bash
# Prediksi harga Apple untuk 7 hari ke depan
python main.py --mode predict --ticker AAPL --days 7

# Prediksi harga Tesla untuk 30 hari ke depan
python main.py --mode predict --ticker TSLA --days 30

# Prediksi harga Microsoft untuk 14 hari ke depan
python main.py --mode predict --ticker MSFT --days 14
```

### 3.3 Output Prediksi

Hasil prediksi akan ditampilkan di terminal dan juga disimpan dalam file CSV di folder `output/plots/`:

```
==================================================
Future Price Predictions for AAPL
==================================================
        Date  Predicted_Price
  2024-01-02           185.23
  2024-01-03           186.45
  2024-01-04           187.12
  ...
==================================================
```

File CSV: `output/plots/AAPL_predictions.csv`

## Langkah 4: Evaluasi Model (Opsional)

Anda dapat mengevaluasi performa model yang sudah dilatih.

```bash
python main.py --mode evaluate --ticker AAPL --model lstm
```

Output akan menampilkan metrik seperti:
- **RMSE**: Root Mean Square Error (semakin kecil semakin baik)
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Koefisien determinasi (mendekati 1 = bagus)
- **Direction Accuracy**: Akurasi arah pergerakan harga

## Langkah 5: Bandingkan Model (Opsional)

Anda dapat membandingkan performa beberapa model sekaligus.

```bash
python main.py --mode compare --ticker AAPL --models lstm,rf,arima
```

## Kode Saham Populer

| Kode | Perusahaan |
|------|------------|
| AAPL | Apple Inc. |
| TSLA | Tesla Inc. |
| MSFT | Microsoft Corp. |
| GOOGL | Alphabet (Google) |
| AMZN | Amazon.com Inc. |
| META | Meta Platforms (Facebook) |
| NVDA | NVIDIA Corp. |
| NFLX | Netflix Inc. |
| BABA | Alibaba Group |
| AMD | Advanced Micro Devices |

## Opsi Tambahan

### Menentukan Rentang Tanggal Data

```bash
python main.py --mode train --ticker AAPL --model lstm --start-date 2020-01-01 --end-date 2024-12-31
```

### Menonaktifkan Visualisasi/Plot

```bash
python main.py --mode train --ticker AAPL --model lstm --no-plot
```

### Menampilkan Plot Secara Interaktif

```bash
python main.py --mode train --ticker AAPL --model lstm --show-plots
```

## Ringkasan Cepat (Quick Start)

Untuk mulai dari awal hingga mendapatkan prediksi, jalankan perintah berikut secara berurutan:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Training model (sekali saja per saham)
python main.py --mode train --ticker AAPL --model lstm --epochs 50

# 3. Prediksi harga
python main.py --mode predict --ticker AAPL --days 30
```

## Troubleshooting

### Error: "Module not found"
Pastikan semua dependencies terinstall:
```bash
pip install -r requirements.txt
```

### Error: "No saved model found"
Anda perlu training model terlebih dahulu sebelum melakukan prediksi:
```bash
python main.py --mode train --ticker AAPL --model lstm
```

### Error: "Invalid ticker"
Pastikan kode saham yang Anda masukkan valid. Kode saham harus sesuai dengan yang terdaftar di Yahoo Finance.

### Training terlalu lama
Kurangi jumlah epochs:
```bash
python main.py --mode train --ticker AAPL --model lstm --epochs 20
```

## Disclaimer

Prediksi harga saham ini hanya untuk tujuan edukasi dan penelitian. Hasil prediksi tidak menjamin akurasi dan tidak boleh dijadikan satu-satunya dasar untuk keputusan investasi. Selalu konsultasikan dengan penasihat keuangan profesional sebelum membuat keputusan investasi.
