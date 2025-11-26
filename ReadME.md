🔋 Data-Driven Battery Cycle Life Prediction






📌 Project Overview

This project replicates the End-of-Life (EOL) prediction capabilities demonstrated in the Nature Energy paper:

"Data-driven prediction of battery cycle life before capacity degradation"
Severson et al., 2019

Using only the first 100 charge/discharge cycles, this ML model predicts the total lifespan of Lithium-Ion batteries (typically 500–2000+ cycles).

⭐ Key Technical Innovations

Use of Summary Statistics (Capacity & Resistance) instead of large 10GB waveform data.

Signal processing via Rolling Mean (window = 9) to handle noise in older testers.

Feature engineering based on electrochemical degradation physics.

📊 Results & Performance
Metric	Result	Notes
Model Type	ElasticNet Regression	Linear with L1/L2 regularization
Input Data	Cycles 10 → 100	Early-life data only
R² Score	0.61	Strong predictive ability with only summary data
RMSE	≈ 275 cycles	Avg. prediction error

🔎 The original paper achieved R² = 0.91 using 10GB+ of raw voltage curves.
Achieving 0.61 with ~3MB summary data shows standard lab outputs are enough for estimation.

🛠️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/<YOUR_USERNAME>/battery-eol-prediction.git
cd battery-eol-prediction

2️⃣ Install Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn h5py

3️⃣ ⚠️ Download Raw Dataset (Important)

Raw .mat files cannot be uploaded to GitHub due to size limits.

Visit the Toyota/Stanford Battery Data Portal.

Download these 3 files:

2017-05-12_batchdata_updated_struct_errorcorrect.mat

2017-06-30_batchdata_updated_struct_errorcorrect.mat

2018-04-12_batchdata_updated_struct_errorcorrect.mat

Create a folder named raw_data/ and place them inside.

Directory Structure
battery-eol-prediction/
├── raw_data/
│   ├── 2017-05-12_....mat
│   ├── 2017-06-30_....mat
│   └── 2018-04-12_....mat
├── 01_process_data.py
├── 02_train_model.py
└── README.md

🚀 How to Run
Step 1 — Process Raw Data

Extracts data from MATLAB files, cleans it, and merges all batches.

python 01_process_data.py


Output: battery_data_combined.csv (~3MB)

Step 2 — Train & Visualize

Computes engineered features, smooths noise, trains the model, and generates graphs.

python 02_train_model.py


Output: results/ folder with generated plots.

🧪 Methodology & Feature Engineering
🔹 1. Log(ΔQ)

Difference in discharge capacity between Cycle 100 and Cycle 10.
(Degradation often follows a power-law → log useful.)

🔹 2. Curvature

How curved the degradation line is.
Formula:

Q55 – LinearAvg(Q10, Q100)

🔹 3. Internal Resistance

Value measured at Cycle 100.

🔧 Noise Handling

Batch 1 (2017-05-12) contains high sensor jitter.
A Rolling Mean — window=9 removes noise effectively.

Without smoothing → model collapses (R² < 0).
With smoothing → clear, usable signal.

📈 Visualizations
1️⃣ Predicted vs Actual Life

Shows model's ability to separate short-life (~500 cycles) and long-life (~2000 cycles) batteries.

results/2_prediction.png

2️⃣ Capacity Fade Curves

Red = short life, Blue = long life.

results/1_curves.png

📚 References

Severson, K.A., Attia, P.M., Jin, N. et al.
“Data-driven prediction of battery cycle life before capacity degradation.”
Nature Energy 4, 383–391 (2019).
