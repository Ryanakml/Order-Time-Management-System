# Ojol Order Prediction System

Aplikasi prediksi jumlah order berbasis time-series menggunakan Streamlit dan Facebook Prophet.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*J0DHQIP_jBCAtXI7tDI0sA.png)

## Fitur

- Manajemen data order (tambah, edit, hapus)
- Upload data CSV custom
- Visualisasi data historis dengan Plotly
- Preprocessing data (agregasi per jam, hari, atau minggu)
- Prediksi jumlah order untuk 7-14 hari ke depan menggunakan Facebook Prophet
- Visualisasi hasil prediksi vs data historis
- Filter data berdasarkan rentang tanggal
- Simpan model untuk penggunaan di masa mendatang

## Instalasi

1. Clone repository ini
2. Install dependensi yang diperlukan:

```bash
pip install -r requirements.txt
```

3. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

## Struktur Proyek

- `app.py`: Aplikasi Streamlit utama
- `preprocessing.py`: Fungsi-fungsi untuk preprocessing data
- `requirements.txt`: Daftar dependensi
- `order_data.csv`: File penyimpanan data order

## Penggunaan

1. Pilih sumber data (Generate Dummy Data atau Upload Data)
2. Jika menggunakan data dummy, klik tombol "Generate New Data"
3. Pilih rentang tanggal untuk analisis
4. Pilih level agregasi (Hourly, Daily, Weekly)
5. Pilih model dan periode prediksi
6. Klik "Train Model and Predict" untuk menjalankan prediksi
7. Lihat hasil prediksi dan komponen forecast
8. Opsional: Simpan model untuk penggunaan di masa mendatang

## Catatan

Aplikasi ini menggunakan Facebook Prophet untuk pemodelan time-series. Prophet adalah library yang dikembangkan oleh Facebook untuk forecasting time-series dengan performa yang baik untuk data dengan pola musiman dan tren.