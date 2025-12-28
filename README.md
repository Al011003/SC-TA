# Financial Performance Modeling using XGBoost

Repository ini berisi implementasi **machine learning berbasis XGBoost** untuk melakukan **pemodelan dan analisis kinerja keuangan perusahaan**, meliputi:

1. **Regresi Net Profit Margin (NPM)**
2. **Klasifikasi Revenue Negative (REVNEG)**
3. **Klasifikasi Net Profit Negative (NETPROFNEG)**

Model dikembangkan menggunakan **Python**, dengan pendekatan **time-based data splitting** dan **manual hyperparameter tuning (grid search)** untuk menjaga validitas evaluasi dan menghindari data leakage.

---

## 📌 Tujuan Penelitian
- Memprediksi **nilai Net Profit Margin (NPM)** menggunakan metode regresi
- Mendeteksi potensi **risiko keuangan perusahaan** melalui klasifikasi kondisi:
  - Pendapatan negatif (REVNEG)
  - Laba bersih negatif (NETPROFNEG)
- Mengimplementasikan proses pelatihan model yang **reproducible** dan **akademis**

---

## 🧠 Metodologi

### 🔹 Algoritma
- **XGBoost Regressor** → Prediksi nilai NPM
- **XGBoost Classifier** → Klasifikasi REVNEG & NETPROFNEG

### 🔹 Fitur Input
- Tahun
- Kuartal
- Kode Emiten (encoded)
- Indeks LQ45
- IHSG

### 🔹 Target
- `NPM_winsor` (Regression)
- `revneg` (Classification)
- `netprofneg` (Classification)

---

## ⏱️ Data Splitting (Time-Based)
Data dibagi berdasarkan urutan waktu untuk mencegah kebocoran data (data leakage):

| Dataset | Periode |
|-------|--------|
| Train | 2022 – 2024 Q2 |
| Validation | 2024 Q3 – Q4 |
| Test | 2025 Q1 – Q2 |

---

## ⚙️ Hyperparameter Tuning
Proses tuning dilakukan menggunakan **manual grid search** (tanpa `GridSearchCV`) dengan kombinasi parameter berikut:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `reg_alpha`
- `reg_lambda`
- `min_child_weight`
- `subsample`
- `colsample_bytree`

Untuk mengatasi **class imbalance**, digunakan parameter:
- `scale_pos_weight` (dihitung otomatis dari data training)

---

## 📁 Struktur Folder
├───backend
│   │   .env
│   │   main.py
│   │   requirements.txt
│   │   runtime.txt
│   │   
│   ├───ml
│   │   │   check_npm_range.py
│   │   │   connect.py
│   │   │   predict.py
│   │   │   preprocessing.py
│   │   │   save_encoder.py
│   │   │   save_scaler.py
│   │   │   test_load.py
│   │   │   test_preprocess.py
│   │   │   train_class.py
│   │   │   train_regression.py
│   │   │   validate_model.py
│   │   │
│   │   └───__pycache__
│   │           connect.cpython-312.pyc
│   │           preprocessing.cpython-312.pyc
│   │
│   └───__pycache__
│           main.cpython-312.pyc
│
└───database_NPM
    │   docker-compose.yml
    │   NPM.xlsx
    │   NPM_202512281546.sql
    │
    └───init
---

## 🖥️ Spesifikasi Sistem

### 💻 Perangkat Lunak
- Python ≥ 3.9
- Google Colab / Visual Studio Code
- Library:
  - xgboost
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn

### 💽 Perangkat Keras (Rekomendasi)
- CPU: Intel i5 / AMD Ryzen 5 atau setara
- RAM: ≥ 8 GB
- Storage: ≥ 5 GB free space

---

## 🚀 Cara Menjalankan Program

### 1️⃣ Install dependency
```bash
pip install -r requirements.txt

2️⃣ Training model klasifikasi
python train_class.py

3️⃣ Training model regresi NPM
python train_regression.py

Model yang telah dilatih akan disimpan dalam format .pkl.


📊 Output

Model regresi NPM (model_npm.pkl)

Model klasifikasi REVNEG (model_revneg.pkl)

Model klasifikasi NETPROFNEG (model_netprofneg.pkl)

Evaluasi performa model (accuracy, R², MAE, RMSE)

🎓 Konteks Akademik

Repository ini dikembangkan untuk keperluan:

Tugas Akhir / Skripsi

Penelitian analisis kinerja keuangan

Eksperimen machine learning pada data time-series keuangan

👤 Author

Al Farhad
Machine Learning & Backend Enthusiast
