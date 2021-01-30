# FR-IoT
IoT dengan Face Recognition 

## Library
*   OpenCv
*   Scipy
*   Tensorflow
*   Scikit-learn
*   Numpy
*   Six

## Persiapan
  * Download dataset facenet [20170512-110547](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) dan simpan di foldern utama (FR_IoT).<br>
  * Biarkan nama folder dan sub folder nya (20170512-110547)<br>
  * Masukkan Data foto ke dalam folder ids dan buat sub folder dengan nama custom (nama custom ini yang akan muncul di frame vidio streaming)<br>
### Contoh


```bash
├── ids
│   ├── Amar
│   │   ├── Amar0.png
│   │   ├── Amar1.png
│   ├── Nurdin
│   │   ├── Nurdin0.png
│   │   ├── Nurdin1.png
```
## Cara Menjalankan:
* masukan command `sh run.sh ids` di folder utama

## Struktur Folder

```bash
├── 20170512-110547
├── ids
│   ├── Amar
│   │   ├── Amar0.png
│   │   ├── Amar1.png
│   ├── Nurdin
│   │   ├── Nurdin0.png
│   │   ├── Nurdin1.png
├── det1.npy
├── det2.npy
├── det3.npy
├── detect_and_align.py
├── main.py
├── README.md
├── requirements.txt
├── run.sh
```
