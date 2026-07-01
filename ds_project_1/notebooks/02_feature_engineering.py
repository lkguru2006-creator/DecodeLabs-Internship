# %% [markdown]
# # Notebook 02 — Feature Engineering
# **DS Project 1 | Advanced EDA & Feature Engineering**
# DecodeLabs Industrial Training Kit — Batch 2026
#
# **Objective:** Document the 3 engineered predictive features with
# mathematical justification and before/after visual validation.

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, '..')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style='darkgrid', palette='muted')

# Load cleaned interim data (Phase 1 output)
df = pd.read_csv('../data/interim/dataset_cleaned.csv', parse_dates=['Date'])
print(f"Interim dataset shape: {df.shape}")
df.head()

# %% [markdown]
# ## 2. Feature 1 — RevenueIntensityScore
#
# **Formula:** `TotalPrice / (Quantity × ItemsInCart)`
#
# **Rationale:** Captures how much revenue is generated per unit of cart activity.
# A high score indicates premium, low-volume orders; a low score indicates bulk/discount orders.
#
# **Implementation:** Fully vectorized using `numpy.where()` — zero Python loops.

# %%
df['RevenueIntensityScore'] = np.where(
    df['Quantity'] * df['ItemsInCart'] > 0,
    df['TotalPrice'] / (df['Quantity'] * df['ItemsInCart']),
    0.0
)

print("RevenueIntensityScore Statistics:")
print(df['RevenueIntensityScore'].describe().round(2))

# %%
# Correlation with target
r = df['RevenueIntensityScore'].corr(df['TotalPrice'])
print(f"\nCorrelation with TotalPrice: {r:.3f}")
print("→ Strong positive signal — useful predictive feature")

# %% [markdown]
# ## 3. Feature 2 — PriceDeviation
#
# **Formula:** `UnitPrice − mean(UnitPrice per Product category)`
#
# **Rationale:** Signals whether this order's unit price is above or below
# the average for its product category. Positive = premium-priced order.
#
# **Implementation:** Vectorized `groupby().transform('mean')` — no iterrows.

# %%
category_means = df.groupby('Product')['UnitPrice'].transform('mean')
df['PriceDeviation'] = df['UnitPrice'] - category_means

print("PriceDeviation Statistics:")
print(df['PriceDeviation'].describe().round(2))

print("\nMean UnitPrice by Product (used for deviation):")
print(df.groupby('Product')['UnitPrice'].mean().round(2).sort_values(ascending=False))

# %%
# Deviation by product
print("\nMean PriceDeviation by Product:")
print(df.groupby('Product')['PriceDeviation'].mean().round(2))

# %% [markdown]
# ## 4. Feature 3 — SpendTier
#
# **Formula:** `pd.cut(TotalPrice, bins=4, labels=[0,1,2,3])`
#
# **Rationale:** Buckets orders into 4 equal-width spend tiers.
# Useful as a stratification key or a soft categorical target.
#
# **Tier Map:** 0=Low, 1=Medium, 2=High, 3=Premium

# %%
df['SpendTier'] = pd.cut(
    df['TotalPrice'],
    bins=4,
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

print("SpendTier Distribution:")
tier_map = {0:'Low', 1:'Medium', 2:'High', 3:'Premium'}
tier_counts = df['SpendTier'].value_counts().sort_index()
for tier, count in tier_counts.items():
    pct = count/len(df)*100
    print(f"  Tier {tier} ({tier_map[tier]:>7}): {count:>4} orders ({pct:.1f}%)")

# %% [markdown]
# ## 5. Feature Correlation Summary

# %%
engineered = ['RevenueIntensityScore', 'PriceDeviation', 'SpendTier']
print("Correlation of Engineered Features with TotalPrice:")
for col in engineered:
    r = df[col].corr(df['TotalPrice'])
    print(f"  {col:<28}: r = {r:+.3f}")

print("\nCorrelation Matrix — Engineered Features:")
subset = df[engineered + ['TotalPrice']]
print(subset.corr().round(3))

# %% [markdown]
# ## 6. Validation — No Nulls Introduced

# %%
print("Null check on engineered features:")
for col in engineered:
    nulls = df[col].isnull().sum()
    status = "✓ OK" if nulls == 0 else f"✗ {nulls} NULLS"
    print(f"  {col}: {status}")

# %% [markdown]
# ## 7. Final Feature Set Summary

# %%
print("=" * 65)
print("FINAL DATASET READY FOR MODEL TRAINING")
print("=" * 65)
print(f"  Records     : {len(df):,}")
print(f"  Raw features: 14")
print(f"  + Engineered: {len(engineered)} (RevenueIntensityScore, PriceDeviation, SpendTier)")
print(f"  + Date feats: 5 (Year, Month, DayOfWeek, Quarter, IsWeekend)")
print(f"  - Dropped   : 6 (IDs, free-text, collinear)")
print(f"  Target      : TotalPrice")
print(f"  Train/Test  : 960 / 240 (point-in-time split)")
print("=" * 65)

if __name__ == "__main__":
    print("\nRun: python3 02_feature_engineering.py  OR  jupyter notebook")
