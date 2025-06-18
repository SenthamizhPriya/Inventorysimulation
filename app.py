import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta
import numpy as np

# Load model and encoder
model = joblib.load("xgb_best_model.pkl")
encoder = joblib.load("target_encoder.pkl")
df_hist = pd.read_csv("final_feature_engineered_data.csv")

# Optional: Load buffer table (precomputed)
buffer_df = pd.DataFrame({
    'Brand Name': ['Barefoot', 'Captain Morgan', 'Jim Beam', 'Coors', 'Sutter Home', 'Jack Daniels', 'Smirnoff', 'Jameson', 'Absolut', 'Budweiser', 'Heineken', 'Malibu', 'Grey Goose', 'Yellow Tail', 'Bacardi', 'Miller'],
    'Buffer_Percent': [18.80, 11.36, 9.45, 8.41, 7.83, 7.35, 6.05, 5.51, 4.40, 3.42, 3.21, 1.79, 1.34, 1.29, 0.92, -5.51]
})

# UI Inputs
st.title("🍸 Inventory Forecast & PAR Level Recommender")
st.sidebar.header("Input Parameters")

start_date = st.sidebar.date_input("Select Start Date", datetime.today())
n_days = st.sidebar.number_input("Forecast for how many days?", min_value=1, max_value=30, value=5)

bar_options = df_hist["Bar Name"].unique()
alcohol_options = df_hist["Alcohol Type"].unique()
brand_options = df_hist["Brand Name"].unique()

bar = st.sidebar.selectbox("Bar Name", bar_options)
alcohol = st.sidebar.selectbox("Alcohol Type", alcohol_options)
brand = st.sidebar.selectbox("Brand Name", brand_options)

current_stock = st.sidebar.number_input("Current Stock (in ml)", min_value=0.0, step=10.0)

# Generate future dates
def generate_future_dates(start, n):
    return [start + timedelta(days=i) for i in range(1, n+1)]

# Build future feature rows
def build_features(start_date, n_days, bar, alcohol, brand):
    future_dates = generate_future_dates(start_date, n_days)
    df = pd.DataFrame({"Date": future_dates})
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # Placeholder values
    df["Opening Balance (ml)"] = 1000
    df["Purchase (ml)"] = 0
    df["lag_1"] = 200
    df["lag_2"] = 200
    df["rolling_3"] = 200
    df["rolling_7"] = 200
    df["std_7"] = 50

    # Categorical
    df["Bar Name"] = bar
    df["Alcohol Type"] = alcohol
    df["Brand Name"] = brand

    # Encoding
    cat_df = df[["Bar Name", "Alcohol Type", "Brand Name"]]
    transformed = encoder.transform(cat_df)
    df["Bar Name_TE"] = transformed["Bar Name"].values
    df["Alcohol Type_TE"] = transformed["Alcohol Type"].values
    df["Brand Name_TE"] = transformed["Brand Name"].values

    return df

# Run forecast
if st.sidebar.button("Run Forecast"):
    df_future = build_features(pd.to_datetime(start_date), n_days, bar, alcohol, brand)
    feature_cols = ['Opening Balance (ml)', 'Purchase (ml)', 'Date', 'Month', 'Year', 'DayOfWeek',
                    'Weekend', 'Bar Name_TE', 'Alcohol Type_TE', 'Brand Name_TE',
                    'lag_1', 'lag_2', 'rolling_3', 'rolling_7', 'std_7']

    X_future = df_future[feature_cols].copy()
    X_future["Date"] = pd.to_numeric(X_future["Date"])  # convert datetime to int
    y_pred = model.predict(X_future)
    df_future["Forecasted Consumption (ml)"] = y_pred

    # Total demand
    total_forecast = y_pred.sum()

    # Use brand-specific buffer if available
    buffer_pct = buffer_df.loc[buffer_df['Brand Name'] == brand, 'Buffer_Percent']
    buffer_ratio = buffer_pct.values[0] / 100 if not buffer_pct.empty else 0.10  # fallback to 10%

    par_level = round(total_forecast * (1 + buffer_ratio), 2)
    restock_qty = max(0, round(par_level - current_stock, 2))

    first_day = df_future["Date"].min().strftime("%B %d")
    last_day = df_future["Date"].max().strftime("%B %d")

    st.subheader("📊 Forecast Summary")
    st.metric("Total Forecast (Next {} days)".format(n_days), f"{total_forecast:.0f} ml")
    st.metric("Recommended PAR Level", f"{par_level:.0f} ml")
    st.metric("Current Stock", f"{current_stock:.0f} ml")

    if restock_qty > 0:
        st.warning(f"⚠️ Please restock at least {restock_qty} ml by {first_day} to maintain inventory through {last_day}.")
    else:
        st.success(f"✅ Stock is sufficient to cover demand through {last_day}.")

    # Optional: show forecast table
    with st.expander("🔍 View Daily Forecast Table"):
        df_show = df_future[["Date", "Forecasted Consumption (ml)"]]
        st.dataframe(df_show.style.format({"Forecasted Consumption (ml)": "{:.2f}"}))
