import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime

# 1. KONTEN & KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Dashboard Analisis Harga Emas Dunia",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Dashboard Analisis & Peramalan Harga Emas Dunia (Metode ARIMA)")
st.markdown("Aplikasi ini digunakan untuk menganalisis tren historis harga emas dunia dan melakukan peramalan (*forecasting*) berbasis data time series.")

# 2. LOAD DATA
@st.cache_data
def load_data():
    # Membaca dataset sesuai file yang diunggah
    df = pd.read_excel("harga_emas_GCF_2020_today.xlsx")
    
    # Membersihkan baris header duplikat/metadata bawaan jika ada di indeks 0 dan 1
    if isinstance(df.iloc[0,0], str) and "Ticker" in df.iloc[0,0]:
        df = df.iloc[2:].reset_index(drop=True)
        
    # Mapping kolom
    df['Price'] = pd.to_datetime(df['Price'])
    df['Close'] = pd.to_numeric(df['Close'])
    df = df.sort_values('Price').reset_index(drop=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Gagal memuat dataset. Pastikan file 'harga_emas_GCF_2020_today.xlsx' berada di folder yang sama. Error: {e}")
    st.stop()

# 3. SIDEBAR (Kontrol Filter & Parameter)
st.sidebar.header("⚙️ Pengaturan Dashboard")

# Filter Rentang Tanggal
min_date = df['Price'].min().to_pydatetime()
max_date = df['Price'].max().to_pydatetime()
start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Waktu Historis:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter data berdasarkan tanggal yang dipilih
filtered_df = df[(df['Price'] >= pd.to_datetime(start_date)) & (df['Price'] <= pd.to_datetime(end_date))]

# Parameter ARIMA
st.sidebar.subheader("🔮 Parameter Model ARIMA")
p = st.sidebar.number_input("Order p (AR):", min_value=0, max_value=5, value=1)
d = st.sidebar.number_input("Order d (Differencing):", min_value=0, max_value=2, value=1)
q = st.sidebar.number_input("Order q (MA):", min_value=0, max_value=5, value=1)

forecast_steps = st.sidebar.slider("Jumlah Hari Peramalan ke Depan:", min_value=5, max_value=60, value=30)

# 4. RINGKASAN METRIK (KPI)
st.subheader("📊 Indikator Utama (KPI)")
col1, col2, col3, col4 = st.columns(4)

latest_price = filtered_df['Close'].iloc[-1]
prev_price = filtered_df['Close'].iloc[-2] if len(filtered_df) > 1 else latest_price
price_diff = latest_price - prev_price

col1.metric("Harga Terakhir (Close)", f"${latest_price:,.2f}", f"{price_diff:+,.2f}")
col2.metric("Harga Tertinggi", f"${filtered_df['High'].astype(float).max():,.2f}")
col3.metric("Harga Terendah", f"${filtered_df['Low'].astype(float).min():,.2f}")
col4.metric("Total Hari Data", f"{len(filtered_df)} Hari")

# 5. GRAFIK HISTORIS & HASIL FORECASTING
st.subheader("📈 Visualisasi Tren & Peramalan ARIMA")

# Latih Model ARIMA dengan data yang difilter
with st.spinner("Melatih model ARIMA dan melakukan peramalan..."):
    # Set indeks waktu untuk model
    ts_data = filtered_df.set_index('Price')['Close']
    ts_data = ts_data.asfreq('B').ffill() # Mengisi hari libur bursa (Business days)
    
    # Fit Model
    try:
        model = ARIMA(ts_data, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast ke depan
        forecast_res = model_fit.get_forecast(steps=forecast_steps)
        forecast_index = forecast_res.predicted_mean.index
        forecast_values = forecast_res.predicted_mean.values
        confidence_intervals = forecast_res.conf_int()
        
        # Plotting
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(ts_data.index, ts_data.values, label="Data Historis Aktual", color="royalblue", lw=2)
        ax.plot(forecast_index, forecast_values, label="Hasil Peramalan", color="darkorange", linestyle="--", lw=2)
        ax.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='orange', alpha=0.15, label="Interval Kepercayaan")
        
        ax.set_title(f"Tren Harga Emas Dunia & Proyeksi ke Depan (ARIMA({p},{d},{q}))", fontsize=14)
        ax.set_xlabel("Tanggal", fontsize=12)
        ax.set_ylabel("Harga (USD)", fontsize=12)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        
        st.pyplot(fig)
        
    except Exception as error:
        st.error(f"Model ARIMA({p},{d},{q}) tidak dapat converge dengan data ini. Silakan sesuaikan parameter p, d, atau q di sidebar. Error: {error}")

# 6. TABEL DATA HILIR
tab1, tab2 = st.tabs(["📋 Data Hasil Peramalan", "🔍 Jelajahi Data Historis"])

with tab1:
    st.write(f"Proyeksi Nilai Harga Emas untuk {forecast_steps} Hari Kerja ke Depan:")
    forecast_df = pd.DataFrame({
        "Tanggal Proyeksi": forecast_index.strftime('%Y-%m-%d'),
        "Estimasi Harga Peramalan (USD)": np.round(forecast_values, 2)
    }).reset_index(drop=True)
    st.dataframe(forecast_df, width="stretch")

with tab2:
    st.write("Data historis mentah berdasarkan filter tanggal:")
    st.dataframe(filtered_df[['Price', 'Open', 'High', 'Low', 'Close', 'Volume']], width="stretch")

# 7. FOOTER
st.markdown("---")
st.caption("Dashboard ini dikembangkan untuk pemenuhan Tugas Besar Mata Kuliah Penambangan Data - Kelompok 10 © 2026.")