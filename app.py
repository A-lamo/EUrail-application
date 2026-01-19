import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="JFK Weather Forecast", page_icon=":)", layout="wide")

@st.cache_resource
def load_and_train_system():
    """
    Trains the Robust Quantile System on startup.
    Returns: Dict of 6 models, History DF, and Feature List.
    """
    # ------------------------------------------------------------------
    # A. LOAD DATA
    # ------------------------------------------------------------------
    try:
        df = pd.read_csv("JFK Airport Weather Data.csv", parse_dates=['DATE'])
    except FileNotFoundError:
        st.error("Data file 'JFK Airport Weather Data.csv' not found. Please upload it.")
        return None, None, None

    df = df.sort_values('DATE')
    
    # Force Numeric Conversion
    non_numeric = ['DATE', 'STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
    for col in df.columns:
        if col not in non_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Clean Gaps
    df = df.dropna(axis=1, how='all')
    df = df.ffill().bfill().fillna(0)

    # ------------------------------------------------------------------
    # B. FEATURE ENGINEERING
    # ------------------------------------------------------------------
    # 1. Domain Knowledge: Wind Vectors & Time
    if 'WDF2' in df.columns:
        df['Wind_Sin'] = np.sin(df['WDF2'] * np.pi / 180)
        df['Wind_Cos'] = np.cos(df['WDF2'] * np.pi / 180)
        
    df['Month'] = df['DATE'].dt.month
    df['DayOfYear'] = df['DATE'].dt.dayofyear
    
    df['Sin_Day'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.0)
    df['Cos_Day'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.0)
    
    # 2. Kitchen Sink Lags
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = ['Sin_Day', 'Cos_Day'] 
    
    for col in numeric_cols:
        if df[col].std() > 0 and col not in ['DayOfYear', 'Month']:
            lag_name = f"{col}_Lag1"
            df[lag_name] = df[col].shift(1).fillna(0)
            feature_names.append(lag_name)
            
    # 3. Complex Interactions
    df['Rolling_Mean_3'] = df['TMAX'].shift(1).rolling(window=3).mean().bfill()
    df['Diff_Yesterday'] = df['TMAX'].shift(1) - df['TMIN'].shift(1)
    
    feature_names.extend(['Rolling_Mean_3', 'Diff_Yesterday'])
    
    # 4. Historical Averages
    monthly_avgs = df.groupby('Month')['TMAX'].mean().to_dict()
    daily_avgs = df.groupby('DayOfYear')['TMAX'].mean().to_dict()
    
    df['Hist_Month_Avg'] = df['Month'].map(monthly_avgs)
    df['Hist_Day_Avg'] = df['DayOfYear'].map(daily_avgs)
    
    feature_names.extend(['Hist_Month_Avg', 'Hist_Day_Avg'])
    feature_names = list(set(feature_names))

    # ------------------------------------------------------------------
    # C. TRAIN QUANTILE MODELS
    # ------------------------------------------------------------------
    df_clean = df.dropna(subset=['TMAX'] + feature_names)
    X = df_clean[feature_names]
    y_max = df_clean['TMAX']
    y_min = df_clean['TMIN']
    
    params = {
        'n_estimators': 600, 
        'max_depth': 5, 
        'learning_rate': 0.05, 
        'n_jobs': -1, 
        'objective': 'reg:quantileerror',
        'random_state': 42
    }
    
    models = {}
    with st.spinner("Initializing & Training Weather Models..."):
        # TMAX Models
        models['tmax_low']  = xgb.XGBRegressor(**params, quantile_alpha=0.05).fit(X, y_max)
        models['tmax_mid']  = xgb.XGBRegressor(**params, quantile_alpha=0.50).fit(X, y_max)
        models['tmax_high'] = xgb.XGBRegressor(**params, quantile_alpha=0.95).fit(X, y_max)
        
        # TMIN Models
        models['tmin_low']  = xgb.XGBRegressor(**params, quantile_alpha=0.05).fit(X, y_min)
        models['tmin_mid']  = xgb.XGBRegressor(**params, quantile_alpha=0.50).fit(X, y_min)
        models['tmin_high'] = xgb.XGBRegressor(**params, quantile_alpha=0.95).fit(X, y_min)
        
    return models, df, feature_names

# Load System
system_data = load_and_train_system()
if system_data[0] is None:
    st.stop()
models, history_df, feature_names = system_data

# ==============================================================================
# 2. PREDICTION ENGINE
# ==============================================================================
def predict_with_confidence(user_input, feature_names):
    """
    Fills input gaps with historical averages, then predicts.
    """
    date_obj = pd.to_datetime(user_input.get('DATE', pd.Timestamp.now()))
    doy = date_obj.dayofyear
    month = date_obj.month
    
    # Get Historical Defaults
    day_stats = history_df[history_df['DayOfYear'] == doy]
    
    defaults = {
        'TMAX': day_stats['TMAX'].mean(),
        'TMIN': day_stats['TMIN'].mean(),
        'PRCP': 0.0, 'SNOW': 0.0, 'SNWD': 0.0,
        'AWND': day_stats['AWND'].mean(),
        'WDF2': day_stats['WDF2'].median(),
        'WT01': 0, 'WT03': 0, 'WT08': 0, 'WT18': 0
    }
    
    merged = {k: user_input.get(k, defaults.get(k, 0)) for k in set(user_input) | set(defaults)}
    
    # Recreate Features
    wind_rad = merged['WDF2'] * np.pi / 180
    
    row = {
        'Sin_Day': np.sin(2 * np.pi * doy / 365.0),
        'Cos_Day': np.cos(2 * np.pi * doy / 365.0),
        'TMAX_Lag1': merged['TMAX'],
        'TMIN_Lag1': merged['TMIN'],
        'PRCP_Lag1': merged['PRCP'],
        'SNOW_Lag1': merged['SNOW'],
        'SNWD_Lag1': merged['SNWD'],
        'AWND_Lag1': merged['AWND'],
        'Wind_Sin_Lag1': np.sin(wind_rad),
        'Wind_Cos_Lag1': np.cos(wind_rad),
        'WT01_Lag1': merged['WT01'],
        'WT03_Lag1': merged['WT03'],
        'WT08_Lag1': merged['WT08'],
        'WT18_Lag1': merged['WT18'],
        'Rolling_Mean_3': merged['TMAX'],
        'Diff_Yesterday': merged['TMAX'] - merged['TMIN'],
        'Hist_Month_Avg': history_df[history_df['Month'] == month]['TMAX'].mean(),
        'Hist_Day_Avg': day_stats['TMAX'].mean()
    }
    
    input_df = pd.DataFrame([row])
    for f in feature_names:
        if f not in input_df.columns:
            input_df[f] = 0
            
    input_df = input_df[feature_names] 
    
    tmax_preds = (
        models['tmax_low'].predict(input_df)[0],
        models['tmax_mid'].predict(input_df)[0],
        models['tmax_high'].predict(input_df)[0]
    )
    tmin_preds = (
        models['tmin_low'].predict(input_df)[0],
        models['tmin_mid'].predict(input_df)[0],
        models['tmin_high'].predict(input_df)[0]
    )
    return tmax_preds, tmin_preds

# ==============================================================================
# 3. USER INTERFACE (CENTERED)
# ==============================================================================
st.title(":) JFK Weather Forecast")
st.markdown("### Confidence-Aware Forecasting System")

# Create a 3-column layout: [Empty, Main Content, Empty]
# This centers the "Main Content" column
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    st.info("💡 Enter today's observation below. The model will forecast tomorrow's weather.")
    
    # --- DATE INPUT ---
    input_date = st.date_input("Date of Observation", pd.Timestamp.now())
    
    # Get defaults for placeholders
    doy = pd.to_datetime(input_date).dayofyear
    if 'DayOfYear' in history_df.columns:
        day_defaults = history_df[history_df['DayOfYear'] == doy].mean(numeric_only=True)
    else:
        day_defaults = {}

    st.write("---")

    # --- INPUTS ---
    st.subheader("Temperature")
    c1, c2 = st.columns(2)
    tmax_in = c1.number_input("High (°F)", value=float(day_defaults.get('TMAX', 65.0)))
    tmin_in = c2.number_input("Low (°F)", value=float(day_defaults.get('TMIN', 50.0)))
    
    st.subheader("Wind Conditions")
    c3, c4 = st.columns(2)
    
    awnd_in = c3.number_input("Speed (mph)", value=float(day_defaults.get('AWND', 10.0)))
    
    # [NEW] Simplified 8-Class Wind Direction
    wind_mapping = {
        "N (North)": 0,
        "NE (North-East)": 45,
        "E (East)": 90,
        "SE (South-East)": 135,
        "S (South)": 180,
        "SW (South-West)": 225,
        "W (West)": 270,
        "NW (North-West)": 315
    }
    
    # Default to West (approx median direction at JFK)
    wdf2_label = c4.selectbox("Direction", options=list(wind_mapping.keys()), index=6)
    wdf2_degrees = wind_mapping[wdf2_label]

    st.subheader("Precipitation & Events")
    c5, c6 = st.columns(2)
    precip_in = c5.number_input("Rain (inches)", value=0.0)
    snow_in = c6.number_input("Snow (inches)", value=0.0)
    
    c7, c8 = st.columns(2)
    is_fog = c7.checkbox("Fog or Haze")
    is_thunder = c8.checkbox("Thunderstorm")
    
    st.write("") # Spacer
    run_btn = st.button("Generate Forecast", type="primary", use_container_width=True)

# --- RESULTS SECTION ---
if run_btn:
    
    # 1. Package Input
    user_data = {
        'DATE': input_date,
        'TMAX': tmax_in, 'TMIN': tmin_in,
        'AWND': awnd_in, 
        'WDF2': wdf2_degrees, # Use the converted degree value
        'PRCP': precip_in, 'SNOW': snow_in,
        'WT01': 1 if is_fog else 0,
        'WT08': 1 if is_fog else 0,
        'WT03': 1 if is_thunder else 0
    }
    
    # 2. Predict
    (high_low, high_mid, high_high), (low_low, low_mid, low_high) = predict_with_confidence(user_data, feature_names)
    
    # 3. Display Results (Using the same center column)
    with center_col:
        st.divider()
        st.subheader(f"Forecast for {input_date + timedelta(days=1)}")
        
        res_c1, res_c2 = st.columns(2)
        
        width_max = high_high - high_low
        width_min = low_high - low_low
        
        with res_c1:
            st.error("🌡️ Predicted High")
            st.metric("Median", f"{high_mid:.1f}°F")
            st.markdown(f"**Safety Tunnel (90%):** \n`{high_low:.1f}°F` —— `{high_high:.1f}°F`")
            if width_max > 20:
                st.warning(f"⚠️ High Volatility (+/- {width_max/2:.0f}°F)")
                
        with res_c2:
            st.info("❄️ Predicted Low")
            st.metric("Median", f"{low_mid:.1f}°F")
            st.markdown(f"**Safety Tunnel (90%):** \n`{low_low:.1f}°F` —— `{low_high:.1f}°F`")
            if width_min > 20:
                st.warning(f"⚠️ High Volatility (+/- {width_min/2:.0f}°F)")


        