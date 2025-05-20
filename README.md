# Real-time Object Detection

Aplikasi deteksi objek real-time menggunakan webcam dan model Hugging Face DETR (DEtection TRansformer).


## Deskripsi

Aplikasi ini menggunakan model DETR dari Facebook (melalui Hugging Face) untuk mendeteksi berbagai objek dalam gambar atau feed webcam real-time. Dengan antarmuka pengguna yang dibuat menggunakan Streamlit, aplikasi ini menawarkan cara yang mudah dan interaktif untuk melihat kemampuan computer vision modern.

### Fitur Utama

- **Deteksi objek real-time** melalui webcam
- **Upload gambar** untuk deteksi objek
- **Pemilihan device kamera** untuk kebutuhan multi-kamera
- **Pengaturan performa** untuk menyesuaikan dengan kemampuan perangkat:
  - Batas kepercayaan (confidence threshold)
  - Skip frame untuk performa lebih baik
  - Resize factor untuk optimasi
  - Batas jumlah deteksi

### Model dan Kategori

Aplikasi menggunakan model DETR yang dapat mendeteksi 80 kategori objek berbeda, termasuk:
- Orang
- Binatang (kucing, anjing, burung, dll)
- Kendaraan (mobil, bus, sepeda, dll)
- Perabotan (kursi, sofa, meja, dll)
- Elektronik (laptop, TV, handphone, dll)
- Dan banyak lagi

## Cara Instalasi

### Prasyarat
- Python 3.7 atau lebih baru
- Pip (package manager Python)
- Webcam (untuk fitur deteksi real-time)

### Langkah Instalasi

1. Clone repository ini atau download file-filenya:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Instalasi paket yang diperlukan:
   ```bash
   pip install streamlit torch torchvision transformers pillow opencv-python numpy
   ```

   Atau jika Anda ingin membuat virtual environment terlebih dahulu:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk macOS/Linux
   # atau
   venv\Scripts\activate  # Untuk Windows
   pip install streamlit torch torchvision transformers pillow opencv-python numpy
   ```

3. Waktu proses instalasi paket bisa memakan waktu beberapa menit tergantung kecepatan internet dan performa komputer Anda.

## Cara Menjalankan

1. Jalankan aplikasi dengan perintah:
   ```bash
   streamlit run object-detection.py
   ```

2. Aplikasi akan terbuka di browser Anda secara otomatis (biasanya di alamat http://localhost:8501)

3. Pilih mode deteksi dari sidebar:
   - **Webcam (Real-time)**: Untuk deteksi objek melalui webcam
   - **Upload Image**: Untuk mendeteksi objek pada gambar yang diunggah

4. Jika menggunakan mode webcam:
   - Klik "Start Camera"
   - Pilih nomor device kamera jika default tidak berfungsi
   - Gunakan slider di sidebar untuk menyesuaikan performa

5. Jika menggunakan mode upload gambar:
   - Unggah gambar melalui uploader
   - Klik "Detect Objects" untuk memproses gambar

## Versi Aplikasi

Ada beberapa versi aplikasi yang tersedia:

1. **object-detection.py**: Versi lengkap dengan semua fitur
2. **simple-detection.py**: Versi dengan UI yang lebih sederhana
3. **realtime-detection.py**: Versi standar tanpa custom CSS

## Troubleshooting

### Masalah Webcam
- Jika webcam tidak terdeteksi, coba pilih nomor device yang berbeda dari dropdown
- Pastikan tidak ada aplikasi lain yang menggunakan webcam Anda
- Jika aplikasi crash, jalankan `cleanup_script.py` untuk memastikan semua resource webcam dilepaskan

### Masalah Performa
- Jika aplikasi lambat, coba tingkatkan nilai "Skip Frames"
- Kurangi "Resize Factor" untuk meningkatkan performa
- Jika menggunakan perangkat dengan GPU, aplikasi akan secara otomatis menggunakannya

### Kebutuhan Memori
- Model bisa memerlukan memori yang cukup besar
- Jika mengalami masalah memori, coba restart aplikasi atau komputer Anda

## Pengembangan Lanjutan

Beberapa ide untuk pengembangan lebih lanjut:
- Implementasi tracking objek
- Menambahkan lebih banyak model yang dapat dipilih
- Fitur perekaman video dengan deteksi objek
- Ekspor hasil deteksi dalam format JSON atau CSV

## Lisensi dan Kredit
- Model DETR: Facebook AI Research
- Transformers library: Hugging Face
- Streamlit: Streamlit, Inc.
