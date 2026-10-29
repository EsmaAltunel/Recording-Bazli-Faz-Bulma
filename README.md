# Full-Recording Zıplama Fazı Tespiti ve IMU–Force Plate Senkronizasyonu

Bu proje, daha önceden Force Plate ile senkronize edilmiş ve zıplama fazları etiketlenmiş sporcuların **IMU + EMG** verilerini kullanarak; **yeni (etiketsiz / senkronize edilmemiş) kayıtların zıplama faz sınırlarını otomatik tespit etmek** ve **kayıt bazlı (recording-level) zaman senkronizasyonu sağlamak** amacıyla geliştirilmiş bir Makine Öğrenmesi + HMM (Viterbi) boru hattıdır (pipeline).

Desteklenen sıçrama tipleri:
- **CMJ** (*Countermovement Jump*)
- **SJ** (*Squat Jump*)
- **DJ** (*Drop Jump*)

---

## Yöntem ve Mimari

1. **Pencere Bazlı Özellik Çıkarımı (50 ms):**
   - Tüm kayıt (*full recording*) boyunca 8 adet Trigno sensöründen alınan **IMU** (`ACC_X`, `ACC_Y`, `ACC_Z`, `ACC_MAG`, `GYRO_MAG`) ve **EMG** sinyalleri 50 ms'lik pencerelere bölünür.
   - Her pencere için `mean`, `std`, `rms` ve `jerk` öznitelikleri hesaplanır (toplam **192 öznitelik**).
2. **6 Durumlu Faz Sınıflandırma (CatBoost + Empirical HMM Viterbi):**
   - Her pencere 6 fazdan birine sınıflandırılır:
     - `0`: **Unweighting** *(CMJ)*
     - `1`: **Propulsion** *(CMJ, SJ, DJ)*
     - `2`: **Flight** *(CMJ, SJ, DJ)*
     - `3`: **Landing** *(CMJ, SJ, DJ)*
     - `4`: **Absorption** *(DJ)*
     - `5`: **Waiting / Background**
   - Sadece eğitim verisindeki faz geçişlerinden **ampirik HMM geçiş matrisi (transition matrix)** öğrenilir ve CatBoost olasılık çıktıları **Viterbi algoritması** ile tüm kayıt boyunca zamansal olarak düzeltilir.
3. **Otomatik Faz Segmentasyonu ve Senkronizasyon:**
   - Tahmin edilen faz dizisinde **Flight** blokları referans (anchor) alınarak her sıçramanın `Absorption`, `Unweighting`, `Propulsion`, `Flight` ve `Landing` başlangıç/bitiş zamanları çıkarılır.
   - Eğer yeni kaydın Force Plate verisi mevcutsa, kayıt içindeki sıçramaların `Landing` başlangıç farklarının medyanı (`recording_offset`) hesaplanarak tüm IMU kaydı tek bir kaydırma ile Force Plate'e senkronize edilir.

---

## Gereksinimler

### Veri Gereksinimleri
Pipeline'ın çalışması için aşağıdaki veriler gereklidir:
1. **Eğitim (Referans) Verisi:**
   - [imu_faz_zamanlari_tum_recording.csv](imu_faz_zamanlari_tum_recording.csv): Önceden senkronize edilmiş sporcuların faz başlangıç ve bitiş zamanlarını içeren etiket dosyası.
   - `joblibler/*.joblib`: Senkronize edilmiş sporculara ait ham sensör kayıtlarını içeren `.joblib` dosyaları (veya önceden üretilmiş `full_recording_ml_training_data.csv`).
2. **Yeni Kişiler (Tahmin / Senkronizasyon) Verisi — İki Seçenekten Biri:**
   - `joblibler_2/*.joblib`: Yeni kişilerin `.joblib` formatındaki kayıtları **VEYA**
   - `ham_sensor_verileri/*.csv`: Trigno Discover'dan doğrudan dışa aktarılmış ham sensör CSV dosyaları.

### Python Kütüphaneleri
```bash
pip install numpy pandas scikit-learn catboost xgboost matplotlib seaborn joblib
```

---

## Dosya Yapısı ve Çalıştırma Sırası

### A) Model Doğruluk ve Zamanlama Değerlendirmesi (Opsiyonel)
Yeni veriler üzerinde tahmin yapmadan önce, mevcut etiketli sporcular üzerinde **LOSO (Leave-One-Subject-Out)** çapraz doğrulama ile algoritmanın başarımını ölçebilirsiniz:

1. **Eğitim Özelliklerini Çıkarma:**
   ```bash
   python feature_from_joblib.py
   ```
   *Çıktı:* `full_recording_ml_training_data.csv`

2. **Sınıflandırma Performansı ve Karmaşıklık Matrisleri (LOSO):**
   ```bash
   python matris.py
   ```
   *Çıktılar:*
   - `full_recording_loso_fold_results.csv` (CatBoost, RandomForest, XGBoost F1 skorları)
   - `full_recording_phase_accuracy.csv` (Faz bazlı doğruluk tablosu)
   - `confusion_matrix_full_recording.png`, `confusion_matrix_full_CMJ.png`, `confusion_matrix_full_SJ.png`, `confusion_matrix_full_DJ.png`

3. **Zamanlama Hatası (MAE, ms) Analizi:**
   ```bash
   python zamanlama_degerlendirme.py
   ```
   *Çıktı:* `zamanlama_degerlendirme_sonuclari.csv` (Faz başlangıç/bitiş zamanlarının Force Plate ground-truth ile milisaniye bazında karşılaştırması)

---

### B) Yeni Kişiler İçin Faz Tespiti ve Senkronizasyon

1. **Yeni Kişilerin Özelliklerini Çıkarma:**
   - **Seçenek 1 — Joblib dosyalarından (`joblibler_2/`):**
     ```bash
     python feature_2_from_joblib.py
     ```
     *Çıktı:* `unlabeled_full_recording_features.csv`
   - **Seçenek 2 — Doğrudan ham Trigno CSV dosyalarından (`ham_sensor_verileri/`):**
     ```bash
     python feature_2_from_csv.py
     ```
     *Çıktı:* `unlabeled_full_recording_features_csvden.csv` *(Not: `train.py` içinde `TEST_PATH` olarak bu dosyayı verebilirsiniz).*

2. **Model Eğitimi ve Faz Tahmini:**
   ```bash
   python train.py
   ```
   Etiketli tüm verilerle her sıçrama tipi (`CMJ`, `SJ`, `DJ`) için ayrı `CatBoost + HMM` modeli eğitir ve yeni kayıtların 50 ms pencereleri için faz tahmini yapar.  
   *Çıktı:* `predicted_phases.csv`

3. **Sıçrama ve Faz Sınırlarının Çıkarılması:**
   ```bash
   python phase.py
   ```
   Force Plate veya ground-truth kullanmadan, yalnızca modelin tahmin ettiği faz dizisinden her sıçramanın faz başlangıç ve bitiş saniyelerini çıkarır.  
   *Çıktı:* `predicted_phase_segments.csv`

4. **Kayıt Bazlı Medyan Senkronizasyon ve Görselleştirme (Opsiyonel):**
   ```bash
   python grafik.py
   ```
   Tahmin edilen `Landing` başlangıçları ile Force Plate `Landing` başlangıçları arasındaki farkların medyanını (`recording_offset`) hesaplar, tüm IMU sinyalini senkronize eder ve karşılaştırma grafiklerini `Recording_Bazli_Landing_Median_Sync/` klasörüne kaydeder.
