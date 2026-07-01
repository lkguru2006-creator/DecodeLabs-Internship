# Data Science Project 1 — Advanced EDA & Feature Engineering
**Industrial Training Kit | Batch: 2026 | Powered by DecodeLabs**

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (all 3 phases)
python main.py

# 3. Generate EDA report
python generate_eda_report.py

# 4. Run all tests
python3 -m unittest tests/test_pipeline.py -v

# 5. Run notebooks
python3 notebooks/01_eda_exploration.py
python3 notebooks/02_feature_engineering.py
```

---

## 🏗️ Architecture: Input → Process → Output (IPO)

```
Phase 1 (INPUT)       Phase 2 (PROCESS)        Phase 3 (OUTPUT)
──────────────────    ─────────────────────    ──────────────────────
Missing data       →  Date vectorization    →  Feature engineering
Outlier IQR        →  Revenue features     →  Schema contracts
Distribution       →  OHE + Label Enc      →  Temporal split
variance guard     →  Collinearity fix     →  Feature store push
```

---

## 📁 Folder Structure

```
ds_project_1/
├── main.py                          # Pipeline orchestrator — run this
├── generate_eda_report.py           # Generates reports/eda_report.html
├── requirements.txt
├── setup.py
├── README.md
│
├── data/
│   ├── raw/dataset.xlsx             # Original immutable dataset (1200×14)
│   ├── interim/dataset_cleaned.csv  # After Phase 1 (missing + outliers)
│   └── processed/
│       ├── dataset_final.csv        # Final encoded + engineered dataset
│       ├── offline_feature_store.csv # Feast offline store (train set)
│       └── online_feature_cache.csv  # Feast online cache (per entity)
│
├── notebooks/
│   ├── 01_eda_exploration.py        # Raw data analysis & findings
│   └── 02_feature_engineering.py   # 3 engineered features with validation
│
├── src/
│   ├── input/
│   │   ├── missing_data.py          # Decision matrix imputation
│   │   ├── outlier_handler.py       # IQR + numpy.clip winsorization
│   │   └── distribution_guard.py   # Std/skew variance checks
│   ├── process/
│   │   ├── vectorizer.py            # NumPy SIMD vectorized transforms
│   │   ├── encoder.py               # Label + One-Hot encoding
│   │   └── collinearity.py          # Pearson matrix, feature pruning
│   └── output/
│       ├── schema_contracts.py      # Runtime data validation contracts
│       ├── feature_store.py         # Offline/online feature store push
│       └── point_in_time.py         # Temporal joins, leakage prevention
│
├── features/
│   ├── feature_registry.yaml        # Full feature metadata registry
│   └── engineered/
│       └── features.py              # 3 engineered predictive features
│
├── configs/
│   ├── pipeline_config.yaml         # Paths and global settings
│   └── imputation_config.yaml       # Thresholds, KNN params, IQR mult.
│
├── contracts/
│   └── failure_logs/failure_cases.log
│
├── tests/
│   └── test_pipeline.py             # 18 unit tests (all passing)
│
├── reports/
│   ├── eda_report.html              # Self-contained EDA report (open in browser)
│   └── figures/                     # 10 auto-generated charts
│
└── logs/
    └── pipeline.log                 # Full pipeline execution log
```

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| File | `data/raw/dataset.xlsx` |
| Records | 1,200 orders |
| Features | 14 raw → 34 after pipeline |
| Target | `TotalPrice` |
| Date range | Jan 2023 – Jun 2025 |
| Missing | `CouponCode` — 309 values (25.8%) |
| Outliers | `TotalPrice` — 8 values (IQR winsorized) |

---

## 🔬 Engineered Features

| Feature | Formula | Correlation with Target |
|---------|---------|------------------------|
| `RevenueIntensityScore` | `TotalPrice / (Qty × CartItems)` | r = +0.07 |
| `PriceDeviation` | `UnitPrice − CategoryMean` | r = +0.72 |
| `SpendTier` | `pd.cut(TotalPrice, 4 bins)` | r = +0.96 |

---

## 🧪 Test Results

```
18 tests run | 0 failures | 0 errors
Leakage check   : PASSED ✓
Schema contracts: PASSED ✓
Train/Test split: 960 / 240 (point-in-time correct)
```

---

## 📦 Dependencies

```
pandas>=2.0      numpy>=1.26      scikit-learn>=1.4
scipy>=1.11      matplotlib>=3.7  seaborn>=0.12
openpyxl>=3.1    pyyaml>=6.0
```

---

*DecodeLabs Industrial Training Kit — Batch 2026*
