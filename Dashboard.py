import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Dashboard Analisis Harga Emas Dunia",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Dashboard Analisis & Peramalan Harga Emas Dunia")
st.markdown("""
Dashboard ini digunakan untuk menganalisis tren historis harga emas dunia
dan melakukan peramalan menggunakan model ARIMA(0,1,0).
""")

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_excel("harga_emas_GCF_2020_today.xlsx")

    # Hapus header Yahoo Finance
    df = df.iloc[2:].reset_index(drop=True)

    # Konversi tipe data
    df["Price"] = pd.to_datetime(df["Price"])

    numeric_cols = [
        "Adj Close",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.sort_values("Price").reset_index(drop=True)

    return df


try:
    df = load_data()

except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Pengaturan Dashboard")

min_date = df["Price"].min().date()
max_date = df["Price"].max().date()

start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

forecast_steps = st.sidebar.slider(
    "Jumlah Hari Forecast",
    min_value=7,
    max_value=60,
    value=31
)

st.sidebar.markdown("---")

st.sidebar.subheader("🏆 Model Terbaik")

st.sidebar.success("ARIMA (0,1,0)")
st.sidebar.write("MAE : 1068.12")
st.sidebar.write("RMSE : 1263.62")
st.sidebar.write("MAPE : 23.57%")

# ==================================================
# FILTER DATA
# ==================================================

filtered_df = df[
    (df["Price"] >= pd.to_datetime(start_date))
    &
    (df["Price"] <= pd.to_datetime(end_date))
].copy()

# ==================================================
# MOVING AVERAGE
# ==================================================

filtered_df["MA_7"] = (
    filtered_df["Close"]
    .rolling(7)
    .mean()
)

filtered_df["MA_30"] = (
    filtered_df["Close"]
    .rolling(30)
    .mean()
)

# ==================================================
# KPI
# ==================================================

st.subheader("📊 Indikator Utama (KPI)")

col1, col2, col3, col4 = st.columns(4)

latest_price = filtered_df["Close"].iloc[-1]

prev_price = (
    filtered_df["Close"].iloc[-2]
    if len(filtered_df) > 1
    else latest_price
)

price_diff = latest_price - prev_price

col1.metric(
    "Harga Terakhir",
    f"${latest_price:,.2f}",
    f"{price_diff:+,.2f}"
)

col2.metric(
    "Harga Tertinggi",
    f"${filtered_df['High'].max():,.2f}"
)

col3.metric(
    "Harga Terendah",
    f"${filtered_df['Low'].min():,.2f}"
)

col4.metric(
    "Jumlah Data",
    len(filtered_df)
)

# ==================================================
# STATISTIK DESKRIPTIF
# ==================================================

st.subheader("📋 Statistik Deskriptif")

st.dataframe(
    filtered_df[
        [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    ]
    .describe()
    .round(2),
    use_container_width=True
)

# ==================================================
# GRAFIK HARGA
# ==================================================

st.subheader("📈 Grafik Harga Emas Dunia")

fig1, ax1 = plt.subplots(figsize=(14,6))

ax1.plot(
    filtered_df["Price"],
    filtered_df["Close"],
    linewidth=2,
    label="Close Price"
)

ax1.set_title("Harga Penutupan Emas Dunia")
ax1.set_xlabel("Tanggal")
ax1.set_ylabel("Harga (USD)")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)

# ==================================================
# MOVING AVERAGE
# ==================================================

st.subheader("📉 Moving Average (MA7 & MA30)")

fig2, ax2 = plt.subplots(figsize=(14,6))

ax2.plot(
    filtered_df["Price"],
    filtered_df["Close"],
    label="Close"
)

ax2.plot(
    filtered_df["Price"],
    filtered_df["MA_7"],
    label="MA 7"
)

ax2.plot(
    filtered_df["Price"],
    filtered_df["MA_30"],
    label="MA 30"
)

ax2.set_title("Close Price dengan Moving Average")
ax2.grid(True)
ax2.legend()

st.pyplot(fig2)

# ==================================================
# EVALUASI MODEL
# ==================================================

st.subheader("🏆 Hasil Evaluasi Model Terbaik")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Model", "ARIMA(0,1,0)")
c2.metric("MAE", "1068.12")
c3.metric("RMSE", "1263.62")
c4.metric("MAPE", "23.57%")

st.success(
    "MAPE 23.57% → Model termasuk kategori Layak/Cukup Baik untuk forecasting."
)

# ==================================================
# FORECAST
# ==================================================

st.subheader("🔮 Forecast Harga Emas")

forecast_df = None

try:

    ts_data = (
        filtered_df
        .set_index("Price")["Close"]
        .asfreq("B")
        .ffill()
    )

    model = ARIMA(
        ts_data,
        order=(0,1,0)
    )

    model_fit = model.fit()

    forecast_res = model_fit.get_forecast(
        steps=forecast_steps
    )

    forecast_index = forecast_res.predicted_mean.index
    forecast_values = forecast_res.predicted_mean.values

    forecast_df = pd.DataFrame({
        "Tanggal Forecast": forecast_index,
        "Prediksi Harga (USD)": np.round(
            forecast_values,
            2
        )
    })

    fig3, ax3 = plt.subplots(figsize=(14,6))

    ax3.plot(
        ts_data.index,
        ts_data.values,
        color="royalblue",
        linewidth=2,
        label="Data Historis"
    )

    ax3.plot(
        forecast_index,
        forecast_values,
        color="red",
        linestyle="--",
        linewidth=3,
        label="Forecast"
    )

    ax3.set_title(
        "Forecast Harga Emas Menggunakan ARIMA(0,1,0)"
    )

    ax3.grid(True)
    ax3.legend()

    st.pyplot(fig3)

except Exception as e:

    st.error(f"Forecast gagal: {e}")

# ==================================================
# TABS
# ==================================================

tab1, tab2 = st.tabs([
    "📋 Hasil Forecast",
    "🔍 Data Historis"
])

with tab1:

    if forecast_df is not None:

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

with tab2:

    st.dataframe(
        filtered_df[
            [
                "Price",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ],
        use_container_width=True
    )

# ==================================================
# KESIMPULAN
# ==================================================

st.subheader("📖 Kesimpulan")

st.info("""
Model terbaik berdasarkan hasil penelitian adalah ARIMA(0,1,0).

MAE : 1068.12

RMSE : 1263.62

MAPE : 23.57%

Model termasuk kategori layak digunakan untuk melakukan
peramalan harga emas dunia.
""")

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Dashboard Penambangan Data - Kelompok 10 © 2026"
)