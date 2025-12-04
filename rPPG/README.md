# Real-time Remote Photoplethysmography (rPPG) Implementation

# Link Repository Github : https://github.com/RafkiHaykhalAlif/Hands-on_Mulmed/tree/main/rPPG
# Referensi Teknis Pengerjaan : https://chatgpt.com/share/69317083-b96c-800a-9dd2-fca014de2adc

## Deskripsi Tugas

Implementasi sistem deteksi detak jantung secara real-time menggunakan webcam dengan teknologi Remote Photoplethysmography (rPPG). Sistem ini menangkap perubahan warna kulit pada wajah yang berkorelasi dengan aliran darah untuk mengestimasi Heart Rate (BPM - Beats Per Minute). Sistem yang dikembangkan mampu memproses video secara real-time dari webcam dan menampilkan hasil estimasi BPM secara langsung di layar dengan akurasi yang tinggi.

---

## Pipeline rPPG

### 1. Deteksi Wajah (Face Detection)

Implementasi ini menggunakan MediaPipe Face Mesh, sebuah library untuk deteksi wajah real-time. MediaPipe dapat mendeteksi dan melacak 468 landmark points pada wajah. Pada saat program dimulai, MediaPipe diinisialisasi dengan parameter tertentu untuk memastikan deteksi. Dalam setiap frame video dari webcam, sistem mengkonversi format warna dari BGR (format OpenCV) menjadi RGB yang merupakan format standar untuk pemrosesan MediaPipe. Setelah itu, MediaPipe melakukan face detection untuk mengidentifikasi landmark-landmark pada wajah. Jika wajah terdeteksi dengan baik, maka landmark-landmark tersebut akan digunakan untuk menentukan lokasi Region of Interest (ROI) yang akan digunakan untuk ekstraksi sinyal. Apabila wajah tidak terdeteksi, program akan menampilkan pesan "No face detected" dan tetap menunggu sampai wajah terdeteksi kembali.

### 2. Ekstraksi Sinyal (Signal Extraction)

Setelah wajah terdeteksi, tahapan berikutnya adalah mengekstraksi sinyal photopletysmographic dari area tertentu pada wajah. Implementasi ini menyediakan berbagai pilihan ROI (Region of Interest) untuk fleksibilitas maksimal. Pengguna dapat memilih antara empat opsi: Forehead (dahi), Cheeks (pipi), Nose (hidung), atau Combined (kombinasi semua area). Setiap ROI memiliki karakteristik unik dalam hal kekuatan sinyal dan ketahanan terhadap gangguan. Proses ekstraksi sinyal dilakukan dengan membuat mask binary untuk setiap ROI. Mask ini kemudian digunakan untuk mengekstraksi pixel-pixel yang berada dalam region tersebut. Dari pixel-pixel yang diseleksi, sistem menghitung nilai rata-rata untuk setiap channel warna (Red, Green, Blue). Nilai-nilai ini direpresentasikan dalam format RGB dan disimpan dalam buffer untuk pemrosesan lebih lanjut. Fokus utama adalah pada Green channel karena channel hijau memiliki penetrasi cahaya terbaik pada jaringan kulit manusia dan paling sensitif terhadap perubahan blood flow yang terjadi saat detak jantung.

### 3. Pemrosesan Sinyal (Signal Processing)

Setelah sinyal diekstraksi, sinyal diproses melalui beberapa tahapan untuk menghilangkan noise dan artifact yang tidak diinginkan. Tahapan pertama adalah resampling, dimana sinyal yang diekstraksi dengan sampling rate yang tidak uniform diinterpolasi kembali ke grid yang uniform dengan target sampling rate 30 Hz. Resampling ini penting untuk memastikan bahwa analisis FFT yang dilakukan nantinya memiliki basis yang konsisten. Tahapan kedua adalah detrending, yaitu menghilangkan low-frequency trend dari sinyal. Low-frequency trend dapat disebabkan oleh perubahan iluminasi, gerakan kepala, atau perubahan gradual lainnya yang tidak terkait dengan detak jantung. Detrending dilakukan dengan menghitung moving average dari sinyal menggunakan kernel dengan panjang 30 frame, kemudian mengurangi sinyal asli dengan moving average tersebut. Hasil dari detrending adalah sinyal yang lebih bersih dengan trend dihilangkan tetapi komponen pulse masih terjaga. Tahapan ketiga adalah aplikasi bandpass filter Butterworth. Filter untuk melewatkan frekuensi dalam rentang 0.67 Hz hingga 4.0 Hz, yang sesuai dengan range detak jantung normal yaitu 40 hingga 240 BPM. Filter dilakukan menggunakan zero-phase filtering (filtfilt) untuk memastikan bahwa tidak ada pergeseran fase yang terjadi pada sinyal. Frekuensi di bawah 0.67 Hz (drift lambat) dan di atas 4.0 Hz (noise cepat) akan difilter keluar. Tahapan keempat adalah analisis FFT (Fast Fourier Transform) untuk mengidentifikasi frekuensi dominan dalam sinyal. FFT mengubah sinyal dari domain waktu menjadi domain frekuensi, sehingga kita dapat melihat spektrum frekuensi dari sinyal. Dalam spektrum frekuensi, frekuensi yang paling dominan (dengan magnitude terbesar) sesuai dengan frekuensi detak jantung.

### 4. Estimasi BPM (Beats Per Minute)

Tahapan terakhir adalah mengkonversi frekuensi dominan yang ditemukan melalui FFT menjadi BPM. Konversi ini sangat sederhana: karena BPM adalah jumlah detak per menit, maka frekuensi (dalam Hz, yaitu detak per detik) harus dikalikan dengan 60 untuk mendapatkan BPM. Sebagai contoh, jika frekuensi dominan yang terdeteksi adalah 1.2 Hz, maka BPM = 1.2 × 60 = 72 BPM. Untuk meningkatkan stabilitas estimasi BPM, sistem mengimplementasikan smoothing dengan menyimpan lima estimasi BPM terakhir dalam buffer dan mengambil rata-rata dari kelima nilai tersebut. Smoothing ini sangat penting karena FFT analysis pada setiap window dapat menghasilkan estimasi yang sedikit berbeda, dan rata-rata ini membantu menghasilkan BPM yang lebih stabil dan tidak berfluktuasi terlalu banyak. Hasil BPM akhir yang telah diperhalus inilah yang ditampilkan kepada pengguna secara real-time di layar.

---

## Perbedaan dengan Demonstrasi Kelas

### Pemprosesan Real-time

Demonstrasi di kelas menggunakan video sudah direkam sebelumnya dan disimpan dalam file. Program membaca seluruh video dari file, memproses setiap frame dari awal hingga akhir, dan menampilkan hasil setelah semua frame selesai diproses. Pendekatan ini cocok untuk analisis retrospektif namun tidak cocok untuk aplikasi real-time. Implementasi yang sekarang dilakukan menggunakan pendekatan real-time streaming, di mana program membaca input langsung dari webcam dan memproses setiap frame dengan segera. Setiap frame yang masuk dari webcam langsung diproses, dianalisis, dan hasilnya langsung ditampilkan di layar tanpa harus menunggu semua frame diproses terlebih dahulu. Keunggulan dari pendekatan real-time adalah bahwa pengguna dapat melihat hasil estimasi BPM secara langsung, mendapatkan instant feedback tentang positioning wajah, dan dapat langsung melihat bagaimana BPM berubah seiring waktu. Hal ini membuat sistem jauh lebih interaktif dan user-friendly.

### Pemilihan ROI Pada Bagian Tertentu di Wajah

Demonstrasi di kelas hanya menggunakan keseluruhan bagian muka dengan melakukan crop pada vidio yang digunakan jadi fokus pada seluruh bagian wajah saja. Implementasi yang sekarang terdapat empat pilihan ROI: Forehead (dahi), Cheeks (pipi), Nose (hidung), dan Combined (kombinasi semua area). Setiap ROI memiliki karakteristik unik. Forehead merupakan bagian kulit paling besar di area wajah. Cheeks (pipi) merupakan pilihan yang direkomendasikan karena area pipi sangat kaya akan pembuluh darah (highly vascular), menghasilkan sinyal yang jauh lebih kuat. Nose menyediakan alternatif untuk pengujian pada kondisi khusus. Combined menggabungkan semua area untuk menghasilkan sinyal dengan maksimal. Fleksibilitas ini memungkinkan untuk beradaptasi dengan berbagai kondisi pencahayaan dan posisi wajah.

### BPM Ditampilkan Real-time Pada Video Overlay

Pada demonstrasi di kelas, hasil BPM ditampilkan sebagai output text di terminal atau console, Implementasi sekarang menampilkan BPM value secara langsung di video frame yang ditangkap dari webcam, sebagai overlay text di bagian bawah layar. Video overlay juga menampilkan informasi tambahan seperti nilai Green channel, jumlah ROI yang aktif, dan FPS. Setiap ROI digambar dengan green box di atas video, sehingga pengguna dapat melihat dengan jelas area mana yang digunakan untuk ekstraksi sinyal. Display real-time ini memberikan feedback yang jauh lebih baik kepada pengguna.

### Visualisasi Grafik Sinyal dan Plot Spektrum Frekuensi Real-time

Demonstrasi di kelas menampilkan hasil analisis sinyal dalam bentuk static plots yang dihasilkan setelah processing selesai. Implementasi sekarang menggunakan real-time interactive plotting yang update secara continuous seiring dengan processing. Tiga plot dihasilkan secara bersamaan dalam satu figure: pertama, Signal plot menampilkan waveform sinyal yang telah diproses dengan detrending dan bandpass filtering, diupdate setiap kali ada data baru; kedua, FFT Spectrum plot menampilkan spektrum frekuensi dari sinyal dalam range bandpass (0.67-4.0 Hz), highlight frekuensi dominan yang sesuai dengan detak jantung; ketiga, BPM History plot menampilkan riwayat estimasi BPM dalam 30 frame terakhir, memvisualisasikan bagaimana BPM berfluktuasi dari waktu ke waktu. Plot-plot ini diupdate setiap 2 detik untuk menjaga frame rate tetap smooth. Visualisasi real-time ini sangat membantu dalam debugging dan understanding bagaimana sistem bekerja.