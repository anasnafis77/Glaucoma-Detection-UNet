# Deteksi Glaukoma berdasarkan karakteristik Optic Disc dan Optic Cup yang terkuantifikasi 
Main Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Kode/Deteksi%20Glaukoma.ipynb#scrollTo=00PZZrqMYiId)

Segmentation Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Semantic_segmentation.ipynb)

Inferencing Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Glaucoma_inference.ipynb)

Intoduction:
Glaukoma merupakan penyakit mata penybebab kebutaan nomor dua setelah katarak. Sayangnya di Indonesia, pendeteksian dini penyakit glaukoma masih merupakan hal yang tidak biasa di lingkungan masyarakat. Selain itu, pendeteksian glaukoma manual merupakan pekerjaan yang tidak mudah oleh dokter mata karena membutuhkan ketelitian dan waktu yang cukup lama. Belum lagi adanya pengaruh subjektifitas terhadap penilaian glaukoma oleh setiap dokter mata membuat metode manual menjadi kurang dapat diandalkan. Oleh karena itu, saya mencoba membuat algoritma deteksi glaukoma berdasarkan karakteristik optic disc dan optic cup yang dapat diimplementasikan pada smartphone. 

Pada projek kali ini, saya akan membuat algoritma deteksi glaukoma berdasarkan karakteristik optic disc dan optic cup dengan input berupa citra retina. Secara umum, pendeteksian ini dilakukan dengan langkah berikut:
1. Preprocessing
2. Lokalisasi Optic Disc
3. Segmentasi Optic Disc dan Cup
4. Ekstraksi Fitur Glaukoma
5. Klasifikasi
 

