\# 🔋 Data-Driven Battery Cycle Life Prediction



!\[Python](https://img.shields.io/badge/Python-3.8%2B-blue)

!\[Machine Learning](https://img.shields.io/badge/ML-ElasticNet-green)

!\[Status](https://img.shields.io/badge/Status-Success-brightgreen)



\## 📌 Project Overview

This project replicates the "End-of-Life" (EOL) prediction capabilities demonstrated in the Nature Energy paper: \*Data-driven prediction of battery cycle life before capacity degradation (Severson et al., 2019)\*.



Using only data from the \*\*first 100 charge/discharge cycles\*\*, this Machine Learning model predicts the total lifespan of Lithium-Ion batteries (which typically ranges from 500 to 2000+ cycles).



\*\*Key Technical Challenge:\*\*

The dataset consists of huge MATLAB files (GBs of voltage curves). This project demonstrates an optimized \*\*Lightweight Approach\*\*:

1\. Extraction of "Summary Statistics" (Capacity \& Resistance) rather than raw waveforms.

2\. Signal Processing (Smoothing) to handle sensor noise in older battery testers.

3\. Feature Engineering based on electrochemical degradation physics.



\## 📊 Results \& Performance



| Metric | Result | Notes |

| :--- | :--- | :--- |

| \*\*Model Type\*\* | ElasticNet Regression | Linear model with Regularization (L1/L2) |

| \*\*Input Data\*\* | Cycles 10 to 100 | Early-life data only |

| \*\*$R^2$ Score\*\* | \*\*0.61\*\* | Strong predictive power using only Summary Data |

| \*\*RMSE\*\* | \*\*~275 Cycles\*\* | Average prediction error |



\*Note: The original paper achieved $R^2=0.91$ by processing 10GB+ of raw voltage curves. Achieving 0.61 with only 3MB of capacity data confirms that standard lab measurements are sufficient for reasonable estimation.\*



---



\## 🛠️ Installation \& Setup



\### 1. Clone the Repository

```bash

git clone https://github.com/<YOUR\_USERNAME>/battery-eol-prediction.git

cd battery-eol-prediction

