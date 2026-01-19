import json

# ==============================================================================
# RECOVERY SCRIPT: RECONSTRUCTING THE NOTEBOOK
# ==============================================================================

notebook_structure = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ✈️ JFK Airport Weather Oracle: Volatility & Risk Analysis\n",
    "\n",
    "## Executive Summary\n",
    "This project implements a machine learning system to predict daily temperature extremes (TMAX/TMIN) at JFK International Airport. \n",
    "\n",
    "**Methodology:**\n",
    "* **Model:** XGBoost Quantile Regression (predicting 5th, 50th, and 95th percentiles).\n",
    "* **Goal:** Generate a dynamic **\"Safety Tunnel\"** (Confidence Interval) to manage the higher weather volatility observed in winter months.\n",
    "* **Optimization:** Hyperparameters tuned via Optuna to minimize Mean Absolute Error (MAE)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import xgboost as xgb\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# ==============================================================================\n",
    "# 1. LOAD & CLEAN DATA\n",
    "# ==============================================================================\n",
    "print(\"Loading and cleaning data...\")\n",
    "df = pd.read_csv(\"JFK Airport Weather Data.csv\", parse_dates=['DATE'])\n",
    "df = df[df['NAME'].str.contains('JFK')].copy().sort_values('DATE')\n",
    "\n",
    "# Select useful columns\n",
    "useful_cols = [\n",
    "    'DATE', 'TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', # Core\n",
    "    'AWND', 'WDF2',                                 # Wind\n",
    "    'WT01', 'WT03', 'WT08', 'WT18'                  # Fog, Thunder, Haze, Snow\n",
    "]\n",
    "df = df[useful_cols].copy()\n",
    "\n",
    "# Fix numeric columns\n",
    "cols_to_numeric = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'AWND', 'WDF2']\n",
    "for col in cols_to_numeric:\n",
    "    df[col] = pd.to_numeric(df[col], errors='coerce')\n",
    "\n",
    "# Basic fill for core weather (persistence)\n",
    "df[cols_to_numeric] = df[cols_to_numeric].ffill().bfill()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Feature Engineering: The \"Time Machine\"\n",
    "We backfill missing historical wind data using a climatological approach (2000-2024 averages applied to 1960-1999) and generate Lag features to capture atmospheric inertia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 2. WIND ENGINEERING & BACKFILLING\n",
    "# ==============================================================================\n",
    "print(\"Backfilling historical wind data...\")\n",
    "# A. Convert Degrees to Vectors\n",
    "df['Wind_Rad'] = df['WDF2'] * np.pi / 180\n",
    "df['Wind_Sin'] = np.sin(df['Wind_Rad'])\n",
    "df['Wind_Cos'] = np.cos(df['Wind_Rad'])\n",
    "\n",
    "# B. Create Climatology from modern data (2000+)\n",
    "df['DayOfYear'] = df['DATE'].dt.dayofyear\n",
    "modern_data = df[df['DATE'].dt.year >= 2000]\n",
    "wind_climatology = modern_data.groupby('DayOfYear')[['Wind_Sin', 'Wind_Cos', 'AWND']].mean()\n",
    "\n",
    "# C. Fill missing past using the Climatology\n",
    "for col in ['Wind_Sin', 'Wind_Cos', 'AWND']:\n",
    "    fill_values = df['DayOfYear'].map(wind_climatology[col])\n",
    "    df[col] = df[col].fillna(fill_values)\n",
    "\n",
    "# ==============================================================================\n",
    "# 3. FEATURE CREATION\n",
    "# ==============================================================================\n",
    "print(\"Generating features...\")\n",
    "# A. Weather Types (WT)\n",
    "wt_cols = ['WT01', 'WT03', 'WT08', 'WT18']\n",
    "df[wt_cols] = df[wt_cols].fillna(0)\n",
    "\n",
    "df['Fog_Yesterday'] = df['WT01'].shift(1)\n",
    "df['Thunder_Yesterday'] = df['WT03'].shift(1)\n",
    "df['Haze_Yesterday'] = df['WT08'].shift(1)\n",
    "df['Snow_Event_Yesterday'] = df['WT18'].shift(1)\n",
    "\n",
    "# B. Wind Lags\n",
    "df['Wind_Sin_Lag1'] = df['Wind_Sin'].shift(1)\n",
    "df['Wind_Cos_Lag1'] = df['Wind_Cos'].shift(1)\n",
    "df['AWND_Lag1'] = df['AWND'].shift(1)\n",
    "\n",
    "# C. Temperature Lags & Trends\n",
    "df['TMAX_Lag1'] = df['TMAX'].shift(1)\n",
    "df['TMAX_Lag2'] = df['TMAX'].shift(2)\n",
    "df['TMIN_Lag1'] = df['TMIN'].shift(1)\n",
    "df['Rolling_Mean_3'] = df['TMAX'].shift(1).rolling(window=3).mean()\n",
    "df['Diff_Yesterday'] = df['TMAX'].shift(1) - df['TMIN'].shift(1)\n",
    "\n",
    "# D. Cyclical Time\n",
    "df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)\n",
    "df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)\n",
    "df['Month'] = df['DATE'].dt.month"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Training Setup\n",
    "We use an 80/20 Time Series Split and calculate **Daily** and **Monthly** historical averages using *only* the training set to prevent data leakage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 4. SPLIT & HISTORICAL AVERAGES\n",
    "# ==============================================================================\n",
    "features = [\n",
    "    'TMAX_Lag1', 'TMAX_Lag2', 'TMIN_Lag1',\n",
    "    'Rolling_Mean_3', 'Diff_Yesterday',\n",
    "    'Wind_Sin_Lag1', 'Wind_Cos_Lag1', 'AWND_Lag1',\n",
    "    'Fog_Yesterday', 'Thunder_Yesterday', 'Haze_Yesterday', 'Snow_Event_Yesterday',\n",
    "    'Sin_Day', 'Cos_Day', \n",
    "    'Hist_Month_Avg', 'Hist_Day_Avg', # <--- Both averages\n",
    "    'PRCP', 'SNOW', 'SNWD'\n",
    "]\n",
    "\n",
    "# Filter NaNs\n",
    "temp_feats = [f for f in features if f not in ['Hist_Month_Avg', 'Hist_Day_Avg']]\n",
    "df_clean = df.dropna(subset=temp_feats + ['TMAX']).copy()\n",
    "\n",
    "# Split Indices\n",
    "split_point = int(len(df_clean) * 0.8)\n",
    "\n",
    "# Calculate Historical Averages (TRAIN SET ONLY)\n",
    "train_df = df_clean.iloc[:split_point]\n",
    "\n",
    "monthly_avgs = train_df.groupby('Month')['TMAX'].mean().to_dict()\n",
    "df_clean['Hist_Month_Avg'] = df_clean['Month'].map(monthly_avgs)\n",
    "\n",
    "daily_avgs = train_df.groupby('DayOfYear')['TMAX'].mean().to_dict()\n",
    "df_clean['Hist_Day_Avg'] = df_clean['DayOfYear'].map(daily_avgs)\n",
    "\n",
    "# Define X and y\n",
    "X = df_clean[features]\n",
    "y = df_clean['TMAX']\n",
    "\n",
    "X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]\n",
    "y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]\n",
    "\n",
    "print(f\"Training on {len(X_train)} rows. Testing on {len(X_test)} rows.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Modeling: The Quantile \"Safety Tunnel\"\n",
    "We train three models using the hyperparameters identified in our previous Optuna study."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 5. TRAINING (Fixed Parameters)\n",
    "# ==============================================================================\n",
    "best_params = {\n",
    "    'n_estimators': 868, \n",
    "    'max_depth': 4, \n",
    "    'learning_rate': 0.05682801933374486, \n",
    "    'subsample': 0.9135732156624227, \n",
    "    'colsample_bytree': 0.7387596608538658, \n",
    "    'reg_alpha': 8.046920463646087, \n",
    "    'reg_lambda': 3.7043435368691187,\n",
    "    'n_jobs': -1,\n",
    "    'random_state': 42,\n",
    "    'objective': 'reg:quantileerror'\n",
    "}\n",
    "\n",
    "print(\"Training Quantile Models (Low, Mid, High)...\")\n",
    "m_low  = xgb.XGBRegressor(**best_params, quantile_alpha=0.05).fit(X_train, y_train)\n",
    "m_mid  = xgb.XGBRegressor(**best_params, quantile_alpha=0.50).fit(X_train, y_train)\n",
    "m_high = xgb.XGBRegressor(**best_params, quantile_alpha=0.95).fit(X_train, y_train)\n",
    "\n",
    "print(\"Generating predictions...\")\n",
    "p_low  = m_low.predict(X_test)\n",
    "p_mid  = m_mid.predict(X_test)\n",
    "p_high = m_high.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Evaluation & Forensics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 6. EVALUATION REPORT\n",
    "# ==============================================================================\n",
    "# 1. Metrics\n",
    "mae = mean_absolute_error(y_test, p_mid)\n",
    "inside_tunnel = (y_test >= p_low) & (y_test <= p_high)\n",
    "coverage_score = inside_tunnel.mean() * 100\n",
    "avg_width = np.mean(p_high - p_low)\n",
    "\n",
    "print(f\"\\nMEDIAN MAE: {mae:.4f} °F\")\n",
    "print(f\"TUNNEL COVERAGE: {coverage_score:.2f}% (Target: 90%)\")\n",
    "print(f\"AVG TUNNEL WIDTH: {avg_width:.2f} °F\")\n",
    "\n",
    "# 2. Visualization (The Tunnel)\n",
    "plt.figure(figsize=(15, 7))\n",
    "subset = 365\n",
    "dates_plot = range(subset)\n",
    "\n",
    "plt.fill_between(dates_plot, p_low[:subset], p_high[:subset], color='purple', alpha=0.15, label='Safety Zone (5-95%)')\n",
    "plt.plot(dates_plot, p_mid[:subset], color='purple', linewidth=2, label='Median Forecast')\n",
    "plt.plot(dates_plot, y_test.iloc[:subset].values, color='black', label='Actual TMAX', linewidth=0.5)\n",
    "\n",
    "# Misses\n",
    "act_sub = y_test.iloc[:subset].values\n",
    "mask = (act_sub < p_low[:subset]) | (act_sub > p_high[:subset])\n",
    "plt.scatter(np.array(dates_plot)[mask], act_sub[mask], color='red', marker='x', s=50, label='Miss')\n",
    "\n",
    "plt.title(f'Final Operational Model\\nMedian MAE: {mae:.2f}°F', fontsize=14)\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.show()\n",
    "\n",
    "# 3. Feature Importance Split (3-Way)\n",
    "fig, axes = plt.subplots(1, 3, figsize=(24, 8))\n",
    "xgb.plot_importance(m_low, ax=axes[0], max_num_features=12, height=0.5, title=\"Drivers of COLD (5th %)\", color='#1f77b4')\n",
    "xgb.plot_importance(m_mid, ax=axes[1], max_num_features=12, height=0.5, title=\"Drivers of MEDIAN (50th %)\", color='#2ca02c')\n",
    "xgb.plot_importance(m_high, ax=axes[2], max_num_features=12, height=0.5, title=\"Drivers of HEAT (95th %)\", color='#d62728')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 7. FORENSIC ANALYSIS (Failures)\n",
    "# ==============================================================================\n",
    "analysis_df = X_test.copy()\n",
    "analysis_df['Date'] = df.loc[analysis_df.index, 'DATE']\n",
    "analysis_df['Actual'] = y_test\n",
    "analysis_df['Pred_Mid'] = p_mid\n",
    "analysis_df['Error'] = analysis_df['Actual'] - analysis_df['Pred_Mid']\n",
    "analysis_df['Abs_Error'] = analysis_df['Error'].abs()\n",
    "\n",
    "print(\"TOP 10 WORST PREDICTIONS:\")\n",
    "cols = ['Date', 'Actual', 'Pred_Mid', 'Error', 'TMAX_Lag1']\n",
    "print(analysis_df.sort_values('Abs_Error', ascending=False).head(10)[cols].to_string(index=False))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('Recovered_JFK_Project.ipynb', 'w') as f:
    json.dump(notebook_structure, f)

print("SUCCESS: Notebook reconstructed as 'Recovered_JFK_Project.ipynb'")