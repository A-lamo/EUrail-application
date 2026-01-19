import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta

# ==============================================================================
# 1. HELPER: FLEXIBLE PREDICTION ENGINE
# ==============================================================================
def predict_weather_flexible(user_input, models, historical_df):
    """
    Predicts tomorrow's weather using whatever data is available today.
    See logic in 'predict_flexible' from previous steps.
    """
    # 1. Parse Date
    if 'DATE' not in user_input: return None, None
    current_date = pd.to_datetime(user_input['DATE'])
    doy = current_date.dayofyear
    
    # 2. Generate Baseline (Climatology)
    if 'DayOfYear' not in historical_df.columns:
        historical_df['DayOfYear'] = pd.to_datetime(historical_df['DATE']).dt.dayofyear

    day_stats = historical_df[historical_df['DayOfYear'] == doy]
    
    defaults = {
        'TMAX': day_stats['TMAX'].mean(),
        'TMIN': day_stats['TMIN'].mean(),
        'AWND': day_stats['AWND'].mean(),
        'WDF2': day_stats['WDF2'].median(),
        'PRCP': 0.0, 'SNOW': 0.0, 'SNWD': 0.0,
        'WT01': 0, 'WT03': 0, 'WT08': 0, 'WT18': 0
    }
    for k,v in defaults.items():
        if pd.isna(v): defaults[k] = 0.0

    # 3. Merge User Input
    current_conditions = defaults.copy()
    current_conditions.update(user_input)
    
    # 4. Feature Engineering
    wind_rad = current_conditions['WDF2'] * np.pi / 180
    
    features = {
        'TMAX_Lag1': current_conditions['TMAX'],
        'TMAX_Lag2': current_conditions['TMAX'],
        'TMIN_Lag1': current_conditions['TMIN'],
        'Rolling_Mean_3': current_conditions['TMAX'],
        'Diff_Yesterday': current_conditions['TMAX'] - current_conditions['TMIN'],
        'Wind_Sin_Lag1': np.sin(wind_rad),
        'Wind_Cos_Lag1': np.cos(wind_rad),
        'AWND_Lag1': current_conditions['AWND'],
        'Fog_Yesterday': int(current_conditions['WT01'] > 0),
        'Thunder_Yesterday': int(current_conditions['WT03'] > 0),
        'Haze_Yesterday': int(current_conditions['WT08'] > 0),
        'Snow_Event_Yesterday': int(current_conditions['WT18'] > 0),
        'Sin_Day': np.sin(2 * np.pi * doy / 365.0),
        'Cos_Day': np.cos(2 * np.pi * doy / 365.0),
        'PRCP': current_conditions.get('PRCP', 0),
        'SNOW': current_conditions.get('SNOW', 0),
        'SNWD': current_conditions.get('SNWD', 0),
        'Hist_Month_Avg': 0.0, 'Hist_Day_Avg': 0.0
    }

    # 5. Prediction Prep
    input_df = pd.DataFrame([features])
    feature_order = [
        'TMAX_Lag1', 'TMAX_Lag2', 'TMIN_Lag1', 'Rolling_Mean_3', 'Diff_Yesterday',
        'Wind_Sin_Lag1', 'Wind_Cos_Lag1', 'AWND_Lag1',
        'Fog_Yesterday', 'Thunder_Yesterday', 'Haze_Yesterday', 'Snow_Event_Yesterday',
        'Sin_Day', 'Cos_Day', 'Hist_Month_Avg', 'Hist_Day_Avg',
        'PRCP', 'SNOW', 'SNWD'
    ]
    
    target_date = current_date + timedelta(days=1)
    hist_month = historical_df[historical_df['DATE'].dt.month == target_date.month]
    hist_day = historical_df[historical_df['DayOfYear'] == target_date.dayofyear]
    
    # Predict TMAX
    input_df['Hist_Month_Avg'] = hist_month['TMAX'].mean()
    input_df['Hist_Day_Avg'] = hist_day['TMAX'].mean()
    tmax_preds = (
        models['tmax_low'].predict(input_df[feature_order])[0],
        models['tmax_mid'].predict(input_df[feature_order])[0],
        models['tmax_high'].predict(input_df[feature_order])[0]
    )

    # Predict TMIN
    input_df['Hist_Month_Avg'] = hist_month['TMIN'].mean()
    input_df['Hist_Day_Avg'] = hist_day['TMIN'].mean()
    tmin_preds = (
        models['tmin_low'].predict(input_df[feature_order])[0],
        models['tmin_mid'].predict(input_df[feature_order])[0],
        models['tmin_high'].predict(input_df[feature_order])[0]
    )
    
    return tmax_preds, tmin_preds

# ==============================================================================
# 2. CACHED MODEL TRAINING
# ==============================================================================
@st.cache_resource
def load_and_train_system():
    # A. Load Data
    df = pd.read_csv("JFK Airport Weather Data.csv", parse_dates=['DATE'])
    df = df[df['NAME'].str.contains('JFK')].copy().sort_values('DATE')
    
    cols_to_numeric = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'SNWD', 'AWND', 'WDF2']
    for col in cols_to_numeric: df[col] = pd.to_numeric(df[col], errors='coerce')
    df[cols_to_numeric] = df[cols_to_numeric].ffill().bfill()
    
    # B. Wind Engineering (Backfill)
    df['Wind_Rad'] = df['WDF2'] * np.pi / 180
    df['Wind_Sin'] = np.sin(df['Wind_Rad'])
    df['Wind_Cos'] = np.cos(df['Wind_Rad'])
    
    df['DayOfYear'] = df['DATE'].dt.dayofyear
    modern = df[df['DATE'].dt.year >= 2000]
    wind_clim = modern.groupby('DayOfYear')[['Wind_Sin', 'Wind_Cos', 'AWND']].mean()
    for c in ['Wind_Sin', 'Wind_Cos', 'AWND']:
        df[c] = df[c].fillna(df['DayOfYear'].map(wind_clim[c]))

    # C. Feature Engineering
    wt_cols = ['WT01', 'WT03', 'WT08', 'WT18']
    df[wt_cols] = df[wt_cols].fillna(0)
    
    df['Fog_Yesterday'] = df['WT01'].shift(1)
    df['Thunder_Yesterday'] = df['WT03'].shift(1)
    df['Haze_Yesterday'] = df['WT08'].shift(1)
    df['Snow_Event_Yesterday'] = df['WT18'].shift(1)
    
    df['Wind_Sin_Lag1'] = df['Wind_Sin'].shift(1)
    df['Wind_Cos_Lag1'] = df['Wind_Cos'].shift(1)
    df['AWND_Lag1'] = df['AWND'].shift(1)
    
    df['TMAX_Lag1'] = df['TMAX'].shift(1)
    df['TMAX_Lag2'] = df['TMAX'].shift(2)
    df['TMIN_Lag1'] = df['TMIN'].shift(1)
    df['Rolling_Mean_3'] = df['TMAX'].shift(1).rolling(window=3).mean()
    df['Diff_Yesterday'] = df['TMAX'].shift(1) - df['TMIN'].shift(1)
    
    df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
    df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
    df['Month'] = df['DATE'].dt.month

    # D. Prepare Training Set
    features = [
        'TMAX_Lag1', 'TMAX_Lag2', 'TMIN_Lag1', 'Rolling_Mean_3', 'Diff_Yesterday',
        'Wind_Sin_Lag1', 'Wind_Cos_Lag1', 'AWND_Lag1',
        'Fog_Yesterday', 'Thunder_Yesterday', 'Haze_Yesterday', 'Snow_Event_Yesterday',
        'Sin_Day', 'Cos_Day', 'Hist_Month_Avg', 'Hist_Day_Avg',
        'PRCP', 'SNOW', 'SNWD'
    ]
    
    df_clean = df.dropna(subset=[f for f in features if 'Hist' not in f] + ['TMAX']).copy()
    split = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split]
    
    # Calculate Histories
    m_avgs_max = train_df.groupby('Month')['TMAX'].mean().to_dict()
    d_avgs_max = train_df.groupby('DayOfYear')['TMAX'].mean().to_dict()
    df_clean['Hist_Month_Avg'] = df_clean['Month'].map(m_avgs_max)
    df_clean['Hist_Day_Avg'] = df_clean['DayOfYear'].map(d_avgs_max)

    X = df_clean[features]
    y = df_clean['TMAX']
    
    # E. Train Models (Using Fixed Params)
    params = {
        'n_estimators': 868, 'max_depth': 4, 'learning_rate': 0.0568, 
        'subsample': 0.913, 'colsample_bytree': 0.738, 
        'reg_alpha': 8.04, 'reg_lambda': 3.70,
        'n_jobs': -1, 'random_state': 42, 'objective': 'reg:quantileerror'
    }
    
    models = {}
    # Train TMAX
    models['tmax_low'] = xgb.XGBRegressor(**params, quantile_alpha=0.05).fit(X, y)
    models['tmax_mid'] = xgb.XGBRegressor(**params, quantile_alpha=0.50).fit(X, y)
    models['tmax_high'] = xgb.XGBRegressor(**params, quantile_alpha=0.95).fit(X, y)
    
    # Train TMIN (Reuse X, change y)
    y_min = df_clean['TMIN']
    models['tmin_low'] = xgb.XGBRegressor(**params, quantile_alpha=0.05).fit(X, y_min)
    models['tmin_mid'] = xgb.XGBRegressor(**params, quantile_alpha=0.50).fit(X, y_min)
    models['tmin_high'] = xgb.XGBRegressor(**params, quantile_alpha=0.95).fit(X, y_min)
    
    return models, df

# ==============================================================================
# 3. STREAMLIT UI
# ==============================================================================
st.set_page_config(page_title="JFK Weather Oracle", page_icon="✈️")
st.title("✈️ JFK Weather Oracle")
st.markdown("### Volatility-Aware Forecasting System")

# Load System
with st.spinner("Initializing Weather Models..."):
    models, history_df = load_and_train_system()

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Select Date")
input_date = st.sidebar.date_input("Forecast for tomorrow relative to:", pd.to_datetime("today"))

# Look up history for defaults
doy = pd.to_datetime(input_date).dayofyear
defaults = history_df[history_df['DayOfYear'] == doy].mean(numeric_only=True)

st.sidebar.header("2. Current Conditions")
st.sidebar.caption("Defaults are based on historical averages for this day. Adjust if you have real-time data.")

tmax_in = st.sidebar.number_input("Today's High (°F)", value=float(defaults.get('TMAX', 50)))
tmin_in = st.sidebar.number_input("Today's Low (°F)", value=float(defaults.get('TMIN', 40)))
precip_in = st.sidebar.number_input("Precipitation (in)", value=0.0, step=0.1)
wind_in = st.sidebar.number_input("Wind Speed (mph)", value=float(defaults.get('AWND', 10)))

st.sidebar.subheader("Events")
c1, c2 = st.sidebar.columns(2)
is_rain = c1.checkbox("Rain/Fog", value=False)
is_snow = c2.checkbox("Snow", value=False)

# --- PREDICT ---
if st.button("Generate Forecast", type="primary"):
    
    # Package Inputs
    user_input = {
        'DATE': str(input_date),
        'TMAX': tmax_in, 'TMIN': tmin_in,
        'AWND': wind_in, 'PRCP': precip_in,
        'WT01': 1 if is_rain else 0, # Fog/Rain
        'WT18': 1 if is_snow else 0  # Snow
    }
    
    # Run Prediction
    tmax_preds, tmin_preds = predict_weather_flexible(user_input, models, history_df)
    
    # --- DISPLAY ---
    st.divider()
    
    # TMAX CARD
    st.subheader(f"Forecast for {input_date + timedelta(days=1)}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🌡️ Maximum Temperature")
        st.metric("Expected High", f"{tmax_preds[1]:.1f}°F")
        st.caption(f"Safety Tunnel (90% Conf): **{tmax_preds[0]:.1f}°F** to **{tmax_preds[2]:.1f}°F**")
        
    with col2:
        st.success("❄️ Minimum Temperature")
        st.metric("Expected Low", f"{tmin_preds[1]:.1f}°F")
        st.caption(f"Safety Tunnel (90% Conf): **{tmin_preds[0]:.1f}°F** to **{tmin_preds[2]:.1f}°F**")

    # Visual Warning
    if (tmax_preds[2] - tmax_preds[0]) > 20:
        st.warning("⚠️ High Volatility Detected: The model is uncertain about tomorrow's weather. Use the Safety Tunnel bounds for planning.")