# 🔋 Data-Driven Battery Cycle Life Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-ElasticNet-green)
![Status](https://img.shields.io/badge/Status-Success-brightgreen)

---

## 📌 Project Overview

This project replicates and optimizes the End-of-Life (EOL) prediction capabilities demonstrated in the Nature Energy paper:

> **"Data-driven prediction of battery cycle life before capacity degradation"**  
> *Severson et al., 2019*

Using only the **first 100 charge/discharge cycles**, this ML model predicts the total lifespan of Lithium-Ion batteries (typically **500–2000+ cycles**).

### ⭐ Key Technical Innovations
- Uses **Summary Statistics** instead of 10GB raw curves  
- Applies **Rolling Mean (window = 9)** for noise reduction  
- Physics-based feature engineering  
- Lightweight + laptop-friendly training pipeline
### 🔑 Key Achievements

- **Replication:** Parsed and processed the complex MIT/Toyota HDF5 dataset.
- **Optimization:** Improved the correlation (ρ) of capacity-based features from the paper’s baseline of **≈ −0.06** to **≈ −0.45** by combining datasets and applying noise filtering.
- **Lightweight Model:** Trained an ElasticNet model achieving **~275 cycles RMSE** using only **3MB** of CSV data, compared to the original **GBs** of raw waveform data.

---

## 📊 Results & Performance

| Metric | Result | Notes |
|-------|--------|-------|
| **Model Type** | ElasticNet Regression | L1/L2 regularization |
| **Input Data** | Cycles **10 → 100** | Early-life data only |
| **R² Score** | **0.61** | Good predictive strength |
| **RMSE** | **~275 cycles** | Avg. absolute error |

> The original paper used 10GB+ raw waveform data (R²=0.91).  
> Here, 3MB summary data achieves R²≈0.61 — with better row value due to smoothing

---

# 🛠️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/battery-eol-prediction.git
cd battery-eol-prediction
```
---
## 2️⃣ Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn h5py
```
## 3️⃣ ⚠️ Download Raw Dataset (Required)
Raw .mat files are NOT included in this repo due to size limits.

>Download the following from the Toyota/Stanford Battery Data Portal:

-2017-05-12_batchdata_updated_struct_errorcorrect.mat

-2017-06-30_batchdata_updated_struct_errorcorrect.mat

-2018-04-12_batchdata_updated_struct_errorcorrect.mat

Create a directory named raw_data/ and place all .mat files inside.
```bash
battery-eol-prediction/
├── raw_data/
│   ├── 2017-05-12_....mat
│   ├── 2017-06-30_....mat
│   └── 2018-04-12_....mat
├── 01_process_data.py
├── 02_train_model.py
└── README.md
```
---
# 🚀 How to Run
## ✅ Step 1 — Process Raw Data
```bash
python 01_process_data.py
```
Output: battery_data_combined.csv (<10MB)
---
## ✅ Step 2 — Train & Visualize
```bash
python 02_train_model.py
```
Output: A results/ folder containing generated graphs.
---
# 🧪 Methodology & Feature Engineering
🔹 1. Log(ΔQ)

Difference in discharge capacity between Cycle 100 and Cycle 10.
(Log transforms stabilize nonlinear degradation.)

🔹 2. Curvature

Measures how non-linear the degradation path is.
```SCSS
Q55 - LinearAvg(Q10, Q100)
```
🔹 3. Internal Resistance

Resistance measured at Cycle 100.
# 🔧 Noise Handling
Dataset batch 2017-05-12 has heavy sensor jitter.
Solution: Rolling Mean smoothing with window=9

❌ Without smoothing → R² < 0

✔ With smoothing → clean degradation curve
---
# 📈 Visualizations
## 1️⃣ Predicted vs Actual Life
```bash
results/2_prediction.png
```
## 2️⃣ Capacity Fade Curves
```bash
results/1_curves.png
```
---
📚 References

Severson, K.A., Attia, P.M., Jin, N., et al.
“Data-driven prediction of battery cycle life before capacity degradation.”
Nature Energy 4, 383–391 (2019).
---
# 📝 Notes
.mat files are ignored via .gitignore

All plots are reproducible by running the scripts

Repository is optimized for clarity & reproducibility

