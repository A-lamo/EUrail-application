import json

# The content of the notebook (Markdown and Code cells)
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ✈️ JFK Airport Weather Oracle: A Volatility-Aware Approach\n",
    "\n",
    "## Executive Summary\n",
    "This project implements a machine learning system to predict daily temperature extremes (TMAX) at JFK International Airport. \n",
    "\n",
    "Unlike standard forecasting models that output a single point prediction, this system acknowledges the **increasing volatility of weather** by generating a **90% Confidence Interval (The \"Safety Tunnel\")** alongside the median forecast. This allows flight planners to understand not just the *expected* temperature, but the *range of probable outcomes*, significantly improving risk management for de-icing and heat safety protocols.\n",
    "\n",
    "**Methodology:**\n",
    "* **Model:** XGBoost Regressor (Gradient Boosting).\n",
    "* **Technique:** Quantile Regression (predicting the 5th, 50th, and 95th percentiles).\n",
    "* **Optimization:** Hyperparameters tuned via Optuna to minimize Mean Absolute Error (MAE)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup & Data Ingestion\n",
    "We begin by loading the historical weather data from JFK Airport. The dataset includes core metrics such as Temperature, Precipitation, Snowfall, and Wind direction/speed."
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
    "from sklearn.metrics import mean_absolute_error\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# ==============================================================================\n",
    "# 1. LOAD & CLEAN DATA\n",
    "# ==============================================================================\n",
    "print(\"Loading and cleaning data...\")\n",
    "df = pd.read_csv(\"JFK Airport Weather Data.csv\", parse_dates=['DATE'])\n",
    "df = df[df['NAME'].str.contains('JFK')].copy().sort_values('DATE')\n",
    "\n",
    "# Select only the columns we determined are useful\n",
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
    "\n",
    "A critical challenge with historical weather data is missing wind records prior to the modern digital era. To solve this, we implement a **Climatological Backfill**:\n",
    "\n",
    "1.  **Vectorization:** Wind direction (Degrees) is converted into `Sin` and `Cos` components to handle the $0^\\circ/360^\\circ$ discontinuity.\n",
    "2.  **Imputation:** We calculate the daily average wind vector from modern data (2000-2024) and map it back to fill gaps in the historical record (1960-1999)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 2. WIND ENGINEERING & BACKFILLING (The \"Time Machine\")\n",
    "# ==============================================================================\n",
    "print(\"Backfilling historical wind data...\")\n",
    "# A. Convert Degrees to Vectors\n",
    "df['Wind_Rad'] = df['WDF2'] * np.pi / 180\n",
    "df['Wind_Sin'] = np.sin(df['Wind_Rad'])\n",
    "df['Wind_Cos'] = np.cos(df['Wind_Rad'])\n",
    "\n",
    "# B. Create Climatology from modern data (2000-2014)\n",
    "df['DayOfYear'] = df['DATE'].dt.dayofyear\n",
    "modern_data = df[df['DATE'].dt.year >= 2000]\n",
    "wind_climatology = modern_data.groupby('DayOfYear')[['Wind_Sin', 'Wind_Cos', 'AWND']].mean()\n",
    "\n",
    "# C. Fill missing past (1960-1999) using the Climatology\n",
    "for col in ['Wind_Sin', 'Wind_Cos', 'AWND']:\n",
    "    fill_values = df['DayOfYear'].map(wind_climatology[col])\n",
    "    df[col] = df[col].fillna(fill_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2.1 Lag Features & Cyclical Time\n",
    "To capture the \"state\" of the atmosphere, we generate features representing:\n",
    "* **Inertia:** Yesterday's temperature and wind conditions.\n",
    "* **Seasonality:** Sine/Cosine transformations of the Day of Year.\n",
    "* **Events:** Boolean flags for Fog, Thunder, or Snow events."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 3. FEATURE ENGINEERING\n",
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
    "## 3. Training Configuration\n",
    "We utilize an **80/20 Time Series Split** to ensure we are testing on \"future\" data the model has never seen. \n",
    "\n",
    "We also calculate **Historical Averages** solely on the training set to prevent data leakage."
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
    "    'Sin_Day', 'Cos_Day', 'Hist_Month_Avg',\n",
    "    'PRCP', 'SNOW', 'SNWD'\n",
    "]\n",
    "\n",
    "df_clean = df.dropna(subset=[f for f in features if f != 'Hist_Month_Avg'] + ['TMAX']).copy()\n",
    "\n",
    "# Split Indices\n",
    "split_point = int(len(df_clean) * 0.8)\n",
    "\n",
    "# Calculate Historical Averages (Train Set Only)\n",
    "train_df = df_clean.iloc[:split_point]\n",
    "monthly_avgs = train_df.groupby('Month')['TMAX'].mean().to_dict()\n",
    "df_clean['Hist_Month_Avg'] = df_clean['Month'].map(monthly_avgs)\n",
    "\n",
    "X = df_clean[features]\n",
    "y = df_clean['TMAX']\n",
    "\n",
    "X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]\n",
    "y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]\n",
    "\n",
    "print(f\"Data Ready. Training on {len(X_train)} rows.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Modeling: The Volatility-Aware Tunnel\n",
    "\n",
    "Instead of a standard regression, we train **three distinct XGBoost models** to create a probabilistic forecast:\n",
    "\n",
    "1.  **Lower Bound (5th Percentile):** The \"Best Case\" for Cold / \"Worst Case\" for Heat.\n",
    "2.  **Median (50th Percentile):** The most likely outcome.\n",
    "3.  **Upper Bound (95th Percentile):** The \"Worst Case\" for Heat / \"Best Case\" for Cold.\n",
    "\n",
    "We use the **optimized hyperparameters** discovered during our earlier Optuna tuning session."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 5. CONFIGURE PARAMETERS (HARDCODED)\n",
    "# ==============================================================================\n",
    "# Optimized parameters derived from Optuna study\n",
    "best_params = {\n",
    "    'n_estimators': 868, \n",
    "    'max_depth': 4, \n",
    "    'learning_rate': 0.05682801933374486, \n",
    "    'subsample': 0.9135732156624227, \n",
    "    'colsample_bytree': 0.7387596608538658, \n",
    "    'reg_alpha': 8.046920463646087, \n",
    "    'reg_lambda': 3.7043435368691187,\n",
    "    'n_jobs': -1,\n",
    "    'random_state': 42\n",
    "}\n",
    "\n",
    "print(\"Using Fixed Parameters:\", best_params)\n",
    "\n",
    "# ==============================================================================\n",
    "# 6. TRAIN QUANTILE REGRESSION (The \"Tunnel\")\n",
    "# ==============================================================================\n",
    "print(\"Training Final Volatility-Aware Tunnel...\")\n",
    "\n",
    "# Copy params and set objective for Quantile Regression\n",
    "q_params = best_params.copy()\n",
    "q_params['objective'] = 'reg:quantileerror'\n",
    "\n",
    "# Train 3 models: 5th percentile (Low), 50th (Median), 95th (High)\n",
    "m_low  = xgb.XGBRegressor(**q_params, quantile_alpha=0.05).fit(X_train, y_train)\n",
    "m_mid  = xgb.XGBRegressor(**q_params, quantile_alpha=0.50).fit(X_train, y_train)\n",
    "m_high = xgb.XGBRegressor(**q_params, quantile_alpha=0.95).fit(X_train, y_train)\n",
    "\n",
    "# Predict\n",
    "p_low  = m_low.predict(X_test)\n",
    "p_mid  = m_mid.predict(X_test)\n",
    "p_high = m_high.predict(X_test)\n",
    "\n",
    "# Calculate Final MAE (using the Median forecast)\n",
    "final_mae = mean_absolute_error(y_test, p_mid)\n",
    "print(f\"Final Median MAE: {final_mae:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visual Analysis\n",
    "\n",
    "The visualization below displays the **Prediction Tunnel**. \n",
    "* **Purple Zone:** The 90% confidence interval. If the black line (Actual Temp) stays within this zone, the model is successfully managing risk.\n",
    "* **Red X:** A \"Miss\" where the weather was more extreme than our 95th percentile prediction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ==============================================================================\n",
    "# 7. VISUALIZATION\n",
    "# ==============================================================================\n",
    "\n",
    "# Plot Feature Importance (Using the Median Model)\n",
    "plt.figure(figsize=(10, 5))\n",
    "xgb.plot_importance(m_mid, max_num_features=10, height=0.5, \n",
    "                   title=\"Final Model Drivers (Weight)\", \n",
    "                   importance_type='weight') \n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Plot The Final Tunnel\n",
    "plt.figure(figsize=(15, 7))\n",
    "subset = 365\n",
    "dates_plot = range(subset)\n",
    "\n",
    "# Plot the Tunnel\n",
    "plt.fill_between(dates_plot, p_low[:subset], p_high[:subset], color='purple', alpha=0.15, label='Volatility-Aware Safety Zone (5%-95%)')\n",
    "plt.plot(dates_plot, p_mid[:subset], color='purple', linewidth=2, label='Median Forecast')\n",
    "plt.plot(dates_plot, y_test.iloc[:subset].values, color='black', label='Actual TMAX', linewidth=0.5)\n",
    "\n",
    "# Highlight Misses\n",
    "actuals = y_test.iloc[:subset].values\n",
    "mask = (actuals < p_low[:subset]) | (actuals > p_high[:subset])\n",
    "plt.scatter(np.array(dates_plot)[mask], actuals[mask], color='red', marker='x', s=50, label='Miss (Outside Tunnel)')\n",
    "\n",
    "plt.title(f'Final Operational Model (Fixed Parameters)\\nMedian MAE: ~{final_mae:.2f}°F', fontsize=14)\n",
    "plt.legend(loc='upper left')\n",
    "plt.grid(True, alpha=0.3)\n",
    "plt.xlabel(\"Days into Test Set\")\n",
    "plt.ylabel(\"Temperature (°F)\")\n",
    "plt.show()"
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

# Write the file
with open('JFK_Weather_Forecast_Final.ipynb', 'w') as f:
    json.dump(notebook_content, f)

print("Notebook generated successfully as 'JFK_Weather_Forecast_Final.ipynb'")