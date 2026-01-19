# EUrail-application

# ✈️ JFK Weather Oracle: Volatility-Aware Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

A machine learning system designed to predict daily temperature extremes (Highs & Lows) at JFK International Airport. Unlike standard forecasts that output a single number, this system generates a **"Safety Tunnel"** (Confidence Interval) to quantify weather volatility, crucial for aviation risk management.



## 📖 Project Overview

Aviation operations require more than just average temperature predictions; they need to know the *extremes*. A sudden freeze or unexpected heatwave impacts de-icing schedules and runway safety. 

This project solves that by moving from **Point Predictions** to **Probabilistic Forecasting**.

### Key Features
* **Quantile Regression:** Predicts the 5th, 50th (Median), and 95th percentiles to create a dynamic confidence interval.
* **Volatility Awareness:** The model automatically widens its "Safety Tunnel" during volatile winter months and narrows it during stable summer months.
* **Hybrid Feature Engineering:** Combines "Kitchen Sink" lagging (brute force) with domain-specific physics (Wind Vectors, Atmospheric Inertia).
* **Interactive GUI:** A Streamlit app that acts as a "Smart Almanac," allowing users to input partial data and filling the gaps with historical climatology.

## 📊 Methodology

The full research process is documented in `analysis.ipynb`.

1.  **Data Forensics:** Handled 60 years of NOAA data, fixing sparse wind records from the 1960s using a "Climatological Backfill" strategy.
2.  **Feature Engineering:** * Converted Wind Direction (Degrees) $\to$ Sin/Cos Vectors.
    * Calculated "Atmospheric Inertia" (Rolling Means, Lags).
3.  **Model Selection:** * Hypothesis testing confirmed **Heteroscedasticity** (changing variance) in the data.
    * *Decision:* Rejected standard Linear Regression in favor of **XGBoost Quantile Regression** to explicitly model this variance.
4.  **Evaluation:** Achieved a Median MAE of ~3-4°F, with the Safety Tunnel successfully capturing >90% of actual weather events.

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* pip

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Data Requirement:**
    Ensure `JFK Airport Weather Data.csv` is in the root directory. (Source: NOAA GHCN Daily).

### Running the App

Launch the interactive forecasting tool:

```bash
streamlit run app.py
