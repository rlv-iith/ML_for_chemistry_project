# 🔋 Data-Driven Battery Cycle Life Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-ElasticNet-green)
![Status](https://img.shields.io/badge/Status-Success-brightgreen)

---

## 📌 Project Overview

This project replicates the *End-of-Life (EOL)* prediction capabilities demonstrated in the **Nature Energy** paper:

> **"Data-driven prediction of battery cycle life before capacity degradation"**  
> *Severson et al., 2019*

Using only the **first 100 charge/discharge cycles**, this ML model predicts the total lifespan of Lithium-Ion batteries (typically **500–2000+ cycles**).

### ⭐ Key Technical Innovations
- Uses **Summary Statistics** instead of 10GB raw curves  
- Applies **Rolling Mean (window = 9)** for noise reduction  
- Physics-based feature engineering  
- Lightweight + laptop-friendly training pipeline  

---

## 📊 Results & Performance

| Metric | Result | Notes |
|-------|--------|-------|
| **Model Type** | ElasticNet Regression | L1/L2 regularization |
| **Input Data** | Cycles **10 → 100** | Early-life data only |
| **R² Score** | **0.61** | Good predictive strength |
| **RMSE** | **~275 cycles** | Avg. absolute error |

> The original paper used 10GB+ raw waveform data (R²=0.91).  
> Here, 3MB summary data achieves R²≈0.61 — proving simpler lab outputs are enough.

---

# 🛠️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<YOUR_USERNAME>/battery-eol-prediction.git
cd battery-eol-prediction
---
##2️⃣ Install Dependencies
