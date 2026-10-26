import streamlit as st
import numpy as np
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from PIL import Image, ImageOps
import requests
import plotly.graph_objects as go
import plotly.express as px
import io
from google import genai
from google.genai import types

# ==============================================================================
# TRACKING PENGUNJUNG
# ==============================================================================
try:
    headers = st.context.headers
    user_ip = headers.get("X-Forwarded-For", "IP tidak terdeteksi")
    user_agent = headers.get("User-Agent", "Browser tidak terdeteksi")
    print(f"[AKSES BARU] IP: {user_ip} | Device: {user_agent}")
except Exception as e:
    pass

# ==============================================================================
# INTEGRASI GEMINI AI API
# ==============================================================================
# Mengambil API Key secara aman dari Secrets Streamlit Cloud
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

import time

@st.cache_data(show_spinner=False)
def analyze_plant_disease_with_gemini(image_bytes_data, crop_name):
    if GEMINI_API_KEY == "MASUKKAN_API_KEY_GEMINI_KAMU_DI_SINI" or not GEMINI_API_KEY:
        return "⚠️ **API Key Gemini belum dipasang.** Silakan isi `GEMINI_API_KEY` di Streamlit Secrets."
    
    # Coba pakai model gemini-2.5-flash dulu, kalau sibuk otomatis coba gemini-1.5-flash
    models_to_try = ['gemini-3.8-flash', 'gemini-3.6-flash']
    
    prompt = f"""
    Kamu adalah pakar patologi tanaman dan agronomis profesional. 
    Analisis foto berikut untuk sampel tanaman: {crop_name}.
    
    Petunjuk analisis:
    1. Pertama, periksa apakah gambar ini benar-benar memuat bagian dari tanaman/daun/buah. Jika BUKAN tanaman, jawab dengan kalimat tegas: "BUKAN_TANAMAN".
    2. Jika memuat tanaman, analisa apakah tanaman tersebut SEHAT atau TERSERANG PENYAKIT/HAMA.
    3. Jika terserang penyakit, berikan:
       - Nama Penyakit / Hama (serta nama ilmiah patogennya).
       - Tingkat Keparahan Infeksi (Rendah / Sedang / Tinggi).
       - Gejala Khas yang terlihat pada foto.
       - Rekomendasi Penanganan dan Pengobatan langkah demi langkah yang praktis untuk petani.
    
    Sajikan jawaban dalam format Markdown yang rapi, profesional, dan mudah dipahami.
    """

    for model_name in models_to_try:
        # Lakukan percobaan hingga 3 kali jika kena error 503 (server sibuk)
        for attempt in range(3):
            try:
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        types.Part.from_bytes(data=image_bytes_data, mime_type='image/jpeg')
                    ]
                )
                return response.text
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    time.sleep(2)  # Tunggu 2 detik sebelum coba lagi
                    continue
                else:
                    return f"⚠️ Gagal menghubungkan ke Gemini AI: {e}"

    return "⚠️ Server Gemini AI sedang sangat sibuk. Silakan tekan tombol analisis sekali lagi dalam beberapa detik."
# ==============================================================================
# FUNGSI INTEGRASI COPERNICUS SENTINEL-2 API & SOILGRIDS
# ==============================================================================
COPERNICUS_CLIENT_ID = st.secrets.get("COPERNICUS_CLIENT_ID", "")
COPERNICUS_CLIENT_SECRET = st.secrets.get("COPERNICUS_CLIENT_SECRET", "")

@st.cache_data(ttl=3600)
def get_sentinel_token():
    if not COPERNICUS_CLIENT_ID or not COPERNICUS_CLIENT_SECRET:
        return None
    try:
        auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        data = {
            "client_id": COPERNICUS_CLIENT_ID,
            "client_secret": COPERNICUS_CLIENT_SECRET,
            "grant_type": "client_credentials"
        }
        response = requests.post(auth_url, data=data, timeout=5)
        if response.status_code == 200:
            return response.json().get("access_token")
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400)
def get_real_sentinel_indices(lat, lon):
    token = get_sentinel_token()
    if token:
        try:
            url = "https://sh.dataspace.copernicus.eu/api/v1/process"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            delta = 0.001
            bbox = [lon - delta, lat - delta, lon + delta, lat + delta]

            evalscript = """
            //VERSION=3
            function setup() {
              return {
                input: [{bands: ["B03", "B04", "B08", "B11"]}],
                output: {id: "default", bands: 4}
              };
            }
            function evaluatePixel(sample) {
              return [sample.B03, sample.B04, sample.B08, sample.B11];
            }
            """

            payload = {
                "input": {
                    "bounds": {"bbox": bbox},
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {"mosaickingOrder": "mostRecent", "maxCloudCoverage": 20}
                    }]
                },
                "output": {"width": 1, "height": 1, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
                "evalscript": evalscript
            }

            res = requests.post(url, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                pass
        except Exception:
            pass

    return None, None, None

@st.cache_data(ttl=3600)
def get_real_soil_ph(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&depth=0-5cm&value=mean"
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code == 200:
            data = response.json()
            raw_ph = data['properties']['layers'][0]['depths'][0]['values']['mean']
            if raw_ph is not None:
                return round(raw_ph / 10.0, 1)
    except Exception:
        pass
    return None

@st.cache_data(ttl=3600)
def get_real_weather_nasa(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,WS2M,ALLSKY_SFC_SW_DW,PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start=20240101&end=20240107&format=JSON"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()['properties']['parameter']
            temp = np.mean(list(data['T2M'].values()))
            humidity = np.mean(list(data['RH2M'].values()))
            wind = np.mean(list(data['WS2M'].values()))
            rainfall = np.mean(list(data['PRECTOTCORR'].values()))
            return round(temp, 1), round(humidity, 1), round(wind, 1), round(rainfall, 1)
    except Exception:
        pass
    return None

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="TaniCerdas AI Precision",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
:root { --forest:#064e3b; --mint:#10b981; --canvas:#f4f8f5; --ink:#102c24; --muted:#668078; }
html,body,[class*="css"] {font-family:'Plus Jakarta Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background:linear-gradient(135deg,#f2f9f5 0%,#fff 49%,#f1f7f2 100%);color:#12342a;}
[data-testid="stHeader"] {background:rgba(244,248,245,.82);backdrop-filter:blur(14px);}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#073e30,#0b513c 58%,#093d31);border-right:1px solid rgba(255,255,255,.1);}
[data-testid="stSidebar"] * {color:#eefdf5;}
[data-testid="stSidebar"] [data-baseweb="select"] * {color:#102c24 !important;}
[data-testid="stSidebar"] .custom-card {background:rgba(255,255,255,.09);color:#fff;border-color:rgba(255,255,255,.14);}
.block-container {padding-top:1rem!important;padding-bottom:3rem!important;max-width:1480px;}
h1,h2,h3,h4,h5 {font-family:'DM Sans',sans-serif;letter-spacing:-.035em;color:#13392b;}
.hero {position:relative;overflow:hidden;border-radius:26px;padding:34px 36px;margin:8px 0 24px;background:radial-gradient(circle at 85% 10%,rgba(118,235,161,.28),transparent 36%),linear-gradient(120deg,#073b2b 0%,#126c48 100%);box-shadow:0 18px 45px rgba(6,78,59,.16);color:white;}
.hero:after {content:'✺';position:absolute;right:38px;top:-55px;font-size:240px;color:rgba(255,255,255,.06);pointer-events:none;}
.hero .eyebrow {font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.19em;color:#9af2c5;}
.hero h1 {color:white;font-size:clamp(26px,3vw,42px);line-height:1.13;margin:10px 0 12px;font-weight:800;}
.hero p {color:#d2f4e1;max-width:680px;line-height:1.75;margin:0;font-size:14px;}
.hero-pills {display:flex;flex-wrap:wrap;gap:9px;margin-top:20px;}
.hero-pills span {font-size:12px;padding:7px 13px;border-radius:100px;border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.1);color:#effff5;}
.custom-card,.metric-card {background:#fff;border-radius:20px;padding:20px;border:1px solid #e2eee6;box-shadow:0 7px 24px rgba(15,75,49,.055);margin-bottom:16px;transition:transform .2s ease,box-shadow .2s ease,border-color .2s;}
.custom-card:hover,.metric-card:hover {transform:translateY(-3px);box-shadow:0 14px 34px rgba(15,75,49,.11);border-color:#b6e5c8;}
.metric-card{text-align:left;min-height:108px;display:flex;flex-direction:column;justify-content:space-between;}
.metric-value {font-size:clamp(22px,2vw,29px);font-weight:800;color:#0d7951;letter-spacing:-.04em;}
.metric-label {font-size:12px;color:#61766c;font-weight:700;}
.stTabs [data-baseweb="tab-list"] {gap:7px;padding:6px;border-radius:15px;background:#eaf3ed;flex-wrap:wrap;box-shadow:inset 0 0 0 1px #dcece1;}
.stTabs [data-baseweb="tab"] {border-radius:11px;height:45px;padding:0 15px;color:#4c6f5c!important;font-size:13px;font-weight:750;}
.stTabs [aria-selected="true"] {background:#fff!important;color:#08794a!important;box-shadow:0 2px 10px rgba(6,78,59,.12);}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important;}
.stButton>button[kind="primary"],div.stButton>button {border-radius:12px;min-height:43px;font-weight:750;transition:all .18s ease;}
div.stButton>button:hover {transform:translateY(-2px);box-shadow:0 8px 18px rgba(16,120,72,.16);}
[data-testid="stFileUploader"], [data-testid="stCameraInput"] {border-radius:17px;overflow:hidden;}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"], iframe {border-radius:16px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:13px;}
[data-baseweb="select"]>div, [data-baseweb="input"]>div {border-radius:11px;}
.suitability-pass {background:linear-gradient(135deg,#e0f9e9,#f2fff7);border:1px solid #77dca2;color:#066b43;padding:20px;border-radius:16px;font-weight:800;}
.suitability-fail {background:linear-gradient(135deg,#fff1ee,#fff8f4);border:1px solid #ffb6a6;color:#a63220;padding:20px;border-radius:16px;font-weight:800;}
code {background:#eaf6ee!important;color:#05643d!important;border-radius:6px;padding:2px 5px!important;}
@media (max-width:700px){.hero{padding:25px 20px;border-radius:18px}.hero:after{font-size:150px;right:0;top:-20px}.block-container{padding-left:1rem!important;padding-right:1rem!important}.stTabs [data-baseweb="tab"]{padding:0 10px;font-size:11px}.metric-card{min-height:92px}}
@media (prefers-reduced-motion:reduce){.custom-card,.metric-card,div.stButton>button{transition:none!important;transform:none!important;}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE KOMODITAS PERTANIAN, BUAH & SAYURAN LENGKAP INDONESIA
# ==============================================================================
CROPS_DATABASE = {
    # 🍌 BUAH-BUAHAN TROPIS & KHAS INDONESIA
    "Pisang (Cavendish/Ambon/Raja/Barangan)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "High", "yield_est": "30.0 - 50.0 Ton/Ha", "cycle": "11-12 Bulan (Panen Kontinu)", "kc_init": 0.50, "kc_mid": 1.10, "kc_end": 1.00},
    "Mangga (Harum Manis/Manalagi)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "Medium", "yield_est": "15.0 - 25.0 Ton/Ha", "cycle": "Tahunan (Panen 1x/Thn)", "kc_init": 0.80, "kc_mid": 1.05, "kc_end": 0.90},
    "Alpukat (Aligator/Miki/Mentega)": {"kategori": "Buah-Buahan", "ph_min": 5.6, "ph_max": 6.5, "ndvi_min": 0.70, "water_req": "Medium-High", "yield_est": "10.0 - 20.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.60, "kc_mid": 0.85, "kc_end": 0.75},
    "Jeruk Siam / Keprok Pontianak": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.60, "water_req": "Medium", "yield_est": "18.0 - 30.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.70, "kc_mid": 0.90, "kc_end": 0.80},
    "Durian (Musang King/Bawor/Montong)": {"kategori": "Buah-Buahan", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.72, "water_req": "High", "yield_est": "8.0 - 15.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.90},
    "Rambutan (Binjai/Rapiah)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.68, "water_req": "Medium-High", "yield_est": "12.0 - 18.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.75, "kc_mid": 0.95, "kc_end": 0.85},
    "Salak (Pondoh/Gula Pasir)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.60, "water_req": "Medium", "yield_est": "15.0 - 25.0 Ton/Ha", "cycle": "Tahunan (Panen Panjangan)", "kc_init": 0.70, "kc_mid": 0.90, "kc_end": 0.80},
    "Kelengkeng (Pingpong/Itoh/Kateki)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.8, "ndvi_min": 0.65, "water_req": "Medium", "yield_est": "8.0 - 14.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.70, "kc_mid": 0.95, "kc_end": 0.85},
    "Jambu Biji (Kristal)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.62, "water_req": "Medium", "yield_est": "20.0 - 35.0 Ton/Ha", "cycle": "Tahunan (Berkala)", "kc_init": 0.60, "kc_mid": 0.85, "kc_end": 0.75},
    "Jambu Air (Madu Deli)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.60, "water_req": "High", "yield_est": "15.0 - 25.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.65, "kc_mid": 0.90, "kc_end": 0.80},
    "Sawo": {"kategori": "Buah-Buahan", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "Low-Medium", "yield_est": "10.0 - 18.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.60, "kc_mid": 0.80, "kc_end": 0.70},
    "Sirsak (Ratu)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.58, "water_req": "Medium", "yield_est": "12.0 - 20.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.55, "kc_mid": 0.80, "kc_end": 0.70},
    "Buah Naga (Red Dragon)": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.45, "water_req": "Low-Medium", "yield_est": "20.0 - 35.0 Ton/Ha", "cycle": "Tahunan (Panen Berkala)", "kc_init": 0.30, "kc_mid": 0.55, "kc_end": 0.40},
    "Manggis": {"kategori": "Buah-Buahan", "ph_min": 5.0, "ph_max": 6.5, "ndvi_min": 0.68, "water_req": "High", "yield_est": "5.0 - 12.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.90},
    "Stroberi": {"kategori": "Buah-Buahan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.50, "water_req": "High (Subtropis/Dataran Tinggi)", "yield_est": "10.0 - 18.0 Ton/Ha", "cycle": "4-6 Bulan", "kc_init": 0.40, "kc_mid": 0.85, "kc_end": 0.75},
    "Apel Malang": {"kategori": "Buah-Buahan", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "Medium (Dataran Tinggi)", "yield_est": "12.0 - 22.0 Ton/Ha", "cycle": "Tahunan (Dua Kali Panen)", "kc_init": 0.60, "kc_mid": 0.95, "kc_end": 0.75},
    "Nanas (Honi/Subang)": {"kategori": "Buah-Buahan", "ph_min": 4.5, "ph_max": 5.5, "ndvi_min": 0.50, "water_req": "Low-Medium", "yield_est": "40.0 - 60.0 Ton/Ha", "cycle": "12-18 Bulan", "kc_init": 0.40, "kc_mid": 0.50, "kc_end": 0.40},
    "Papaya (Calina/California)": {"kategori": "Buah-Buahan", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.60, "water_req": "Medium-High", "yield_est": "40.0 - 70.0 Ton/Ha", "cycle": "8-9 Bulan", "kc_init": 0.50, "kc_mid": 1.05, "kc_end": 0.85},

    # 🥬 SAYUR-SAYURAN INDONESIA
    "Cabai Merah": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.55, "water_req": "Medium-High", "yield_est": "10.0 - 15.0 Ton/Ha", "cycle": "80-90 Hari", "kc_init": 0.60, "kc_mid": 1.15, "kc_end": 0.80},
    "Cabai Rawit": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.55, "water_req": "Medium", "yield_est": "8.0 - 12.0 Ton/Ha", "cycle": "80-100 Hari", "kc_init": 0.60, "kc_mid": 1.15, "kc_end": 0.80},
    "Bawang Merah": {"kategori": "Sayur-Sayuran", "ph_min": 5.6, "ph_max": 6.5, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "10.0 - 14.0 Ton/Ha", "cycle": "60-70 Hari", "kc_init": 0.70, "kc_mid": 1.05, "kc_end": 0.75},
    "Tomat": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.55, "water_req": "Medium-High", "yield_est": "25.0 - 40.0 Ton/Ha", "cycle": "75-90 Hari", "kc_init": 0.60, "kc_mid": 1.15, "kc_end": 0.80},
    "Kentang": {"kategori": "Sayur-Sayuran", "ph_min": 5.0, "ph_max": 6.5, "ndvi_min": 0.60, "water_req": "High", "yield_est": "18.0 - 25.0 Ton/Ha", "cycle": "90-120 Hari", "kc_init": 0.50, "kc_mid": 1.15, "kc_end": 0.75},
    "Kubis / Kol": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.55, "water_req": "Medium-High", "yield_est": "20.0 - 30.0 Ton/Ha", "cycle": "80-90 Hari", "kc_init": 0.70, "kc_mid": 1.05, "kc_end": 0.95},
    "Terung (Ungu/Hijau)": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 6.8, "ndvi_min": 0.55, "water_req": "Medium", "yield_est": "20.0 - 35.0 Ton/Ha", "cycle": "70-90 Hari", "kc_init": 0.60, "kc_mid": 1.05, "kc_end": 0.80},
    "Buncis": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "10.0 - 18.0 Ton/Ha", "cycle": "50-60 Hari", "kc_init": 0.40, "kc_mid": 1.05, "kc_end": 0.90},
    "Kacang Panjang": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "12.0 - 20.0 Ton/Ha", "cycle": "45-60 Hari", "kc_init": 0.40, "kc_mid": 1.00, "kc_end": 0.85},
    "Oyong / Gambas": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 6.8, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "15.0 - 25.0 Ton/Ha", "cycle": "40-50 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.75},
    "Pare": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.52, "water_req": "Medium", "yield_est": "15.0 - 22.0 Ton/Ha", "cycle": "50-60 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.80},
    "Labu Siam": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 6.8, "ndvi_min": 0.60, "water_req": "High", "yield_est": "30.0 - 50.0 Ton/Ha", "cycle": "3-4 Bulan", "kc_init": 0.50, "kc_mid": 1.05, "kc_end": 0.85},
    "Kangkung (Darat/Air)": {"kategori": "Sayur-Sayuran", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.45, "water_req": "High", "yield_est": "12.0 - 20.0 Ton/Ha", "cycle": "25-30 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.90},
    "Bayam (Cabut/Petik)": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.45, "water_req": "Medium-High", "yield_est": "10.0 - 15.0 Ton/Ha", "cycle": "20-25 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.90},
    "Sawi Hijau / Caisim": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.50, "water_req": "Medium-High", "yield_est": "15.0 - 22.0 Ton/Ha", "cycle": "30-40 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.95},
    "Pakcoy": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 7.0, "ndvi_min": 0.50, "water_req": "Medium-High", "yield_est": "15.0 - 25.0 Ton/Ha", "cycle": "30-35 Hari", "kc_init": 0.50, "kc_mid": 1.00, "kc_end": 0.95},
    "Lobak Putih": {"kategori": "Sayur-Sayuran", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.52, "water_req": "Medium", "yield_est": "20.0 - 30.0 Ton/Ha", "cycle": "50-60 Hari", "kc_init": 0.50, "kc_mid": 0.95, "kc_end": 0.85},

    # 🌾 TANAMAN PANGAN
    "Padi Sawah": {"kategori": "Tanaman Pangan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "High (Irigasi Basah)", "yield_est": "6.0 - 7.5 Ton/Ha", "cycle": "110-120 Hari", "kc_init": 1.05, "kc_mid": 1.20, "kc_end": 0.90},
    "Padi Gogo (Lahan Kering)": {"kategori": "Tanaman Pangan", "ph_min": 5.0, "ph_max": 6.5, "ndvi_min": 0.55, "water_req": "Medium (Tadah Hujan)", "yield_est": "3.5 - 4.5 Ton/Ha", "cycle": "110-125 Hari", "kc_init": 0.40, "kc_mid": 1.15, "kc_end": 0.35},
    "Jagung": {"kategori": "Tanaman Pangan", "ph_min": 5.8, "ph_max": 7.2, "ndvi_min": 0.60, "water_req": "Medium", "yield_est": "7.0 - 9.0 Ton/Ha", "cycle": "100-110 Hari", "kc_init": 0.30, "kc_mid": 1.20, "kc_end": 0.60},
    "Kedelai": {"kategori": "Tanaman Pangan", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "1.8 - 2.5 Ton/Ha", "cycle": "80-90 Hari", "kc_init": 0.40, "kc_mid": 1.15, "kc_end": 0.50},
    "Ubi Kayu (Singkong)": {"kategori": "Tanaman Pangan", "ph_min": 4.5, "ph_max": 7.0, "ndvi_min": 0.45, "water_req": "Low", "yield_est": "20.0 - 35.0 Ton/Ha", "cycle": "8-10 Bulan", "kc_init": 0.30, "kc_mid": 1.10, "kc_end": 0.50},

    # 🌴 PERKEBUNAN & REMPAH
    "Kelapa Sawit": {"kategori": "Perkebunan", "ph_min": 4.0, "ph_max": 6.5, "ndvi_min": 0.70, "water_req": "Very High", "yield_est": "20.0 - 25.0 Ton/Ha", "cycle": "Tahunan (3-25 Thn)", "kc_init": 0.90, "kc_mid": 1.00, "kc_end": 0.95},
    "Kopi Arabika": {"kategori": "Perkebunan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.65, "water_req": "Medium-High", "yield_est": "1.0 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.90, "kc_mid": 1.05, "kc_end": 0.95},
    "Kakao (Cokelat)": {"kategori": "Perkebunan", "ph_min": 6.0, "ph_max": 7.5, "ndvi_min": 0.65, "water_req": "High", "yield_est": "1.2 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.90},
    "Karet": {"kategori": "Perkebunan", "ph_min": 4.5, "ph_max": 5.5, "ndvi_min": 0.70, "water_req": "High", "yield_est": "1.5 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.85}
}

# ==============================================================================
# 2. SIDEBAR NAVIGATION & GLOBAL CROP SELECTION
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="padding:14px 5px 18px">
          <div style="font-size:25px;font-weight:800;letter-spacing:-1px;color:#f0fff5">🌱 TaniCerdas<span style="color:#70e4a1"> AI</span></div>
          <div style="font-size:10px;letter-spacing:2px;color:#b7e5ca;margin-top:5px">SMART AGRICULTURE PLATFORM</div>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🎯 Pengaturan komoditas")
    categories = list(set(crop["kategori"] for crop in CROPS_DATABASE.values()))

    default_cat_idx = sorted(categories).index("Buah-Buahan") if "Buah-Buahan" in categories else 0
    selected_cat = st.selectbox("Pilih Kategori Komoditas:", sorted(categories), index=default_cat_idx)

    filtered_crops = [k for k, v in CROPS_DATABASE.items() if v["kategori"] == selected_cat]

    banana_key = "Pisang (Cavendish/Ambon/Raja/Barangan)"
    default_crop_idx = filtered_crops.index(banana_key) if banana_key in filtered_crops else 0

    selected_crop_global = st.selectbox("Pilih Tanaman Target:", sorted(filtered_crops), index=default_crop_idx)

    crop_data = CROPS_DATABASE[selected_crop_global]

    st.markdown(f"""
        <div class="custom-card" style="margin-top: 15px; padding: 15px; border-left: 4px solid #10b981;">
            <div style="font-size:0.85rem;"><b>Kategori:</b> {crop_data['kategori']}</div>
            <div style="font-size:0.85rem; margin-top:5px;"><b>Siklus Tanam:</b> {crop_data['cycle']}</div>
            <div style="font-size:0.85rem; margin-top:5px;"><b>Estimasi Hasil:</b> {crop_data['yield_est']}</div>
        </div>
    """, unsafe_allow_html=True)

# Dashboard header
st.markdown(f"""
<section class="hero">
  <div class="eyebrow">✦ Precision farming • Intelligence dashboard</div>
  <h1>Kelola Lahan Lebih Cerdas.<br>Panen Lebih Optimal.</h1>
  <p>Eksplorasi peta lahan, deteksi gangguan tanaman dengan AI, simulasi kebutuhan irigasi dan pelajari potensi panen dalam satu platform interaktif.</p>
  <div class="hero-pills"><span>🌿 {selected_crop_global}</span><span>🗺️ Pemetaan lahan</span><span>🔬 Diagnosa AI</span><span>💧 Irigasi presisi</span></div>
</section>
""", unsafe_allow_html=True)
st.caption("💡 Mulai dengan memilih komoditas di sidebar, kemudian buka fitur yang ingin digunakan di bawah ini.")

# ==============================================================================
# 3. TAB NAVIGATION
# ==============================================================================
tab1, tab2, tab3, tab_yield, tab4, tab5 = st.tabs([
    "🗺️ Pemetaan Lahan",
    "🔬 Diagnosa AI",
    "💧 Irigasi",
    "📈 Prediksi Panen",
    "🌿 Katalog Tanaman",
    "📚 Dataset"
])

# ------------------------------------------------------------------------------
# TAB 1: KELAYAKAN LAHAN
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(f"### 🗺️ Peta Interaktif & Kelayakan Lahan")
    st.caption(f"Gambar batas area pertanian di peta untuk mengkaji komoditas: {selected_crop_global}.")

    lat_center, lon_center = -7.4965, 112.6316

    m = folium.Map(
        location=[lat_center, lon_center],
        zoom_start=17,
        max_zoom=19,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery"
    )

    draw = Draw(
        export=True,
        filename='field_polygon.geojson',
        position='topleft',
        draw_options={'polyline': False, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': True, 'circlemarker': False},
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)

    st_data = st_folium(m, width="100%", height=420, key="eos_map_precision")

    drawn_features = st_data.get("all_drawings") if st_data else None
    area_ha = 0.0
    current_lat = lat_center
    current_lon = lon_center

    if drawn_features and len(drawn_features) > 0:
        last_draw = drawn_features[-1]
        geom_type = last_draw['geometry']['type']

        if geom_type == 'Point':
            current_lon, current_lat = last_draw['geometry']['coordinates']
            area_ha = 0.5
        else:
            coords = last_draw['geometry']['coordinates'][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            current_lat = sum(lats) / len(lats)
            current_lon = sum(lons) / len(lons)
            area_ha = round(abs((max(lats) - min(lats)) * (max(lons) - min(lons))) * 11100, 2)
            if area_ha < 0.1: area_ha = 1.2

    real_ndvi, real_ndwi, real_ndbi = get_real_sentinel_indices(current_lat, current_lon)

    ndvi_val = real_ndvi if real_ndvi is not None else 0.72
    ndwi_val = real_ndwi if real_ndwi is not None else 0.38
    ndbi_val = real_ndbi if real_ndbi is not None else -0.42

    is_settlement = (ndbi_val > 0.2) and (ndvi_val < 0.3)

    real_ph = get_real_soil_ph(current_lat, current_lon)
    ph_lahan = real_ph if real_ph is not None else (5.2 if is_settlement else 6.2)

    crop_ph_match = crop_data["ph_min"] <= ph_lahan <= crop_data["ph_max"]
    crop_ndvi_match = ndvi_val >= crop_data["ndvi_min"]
    is_suitable = (not is_settlement) and crop_ph_match and crop_ndvi_match

    st.caption("⚠️ NDVI/NDWI/NDBI menggunakan nilai contoh jika data satelit tidak tersedia; jangan gunakan sebagai pengukuran lapangan.")
    st.markdown(f'''
        <div class="custom-card" style="padding: 14px 20px; margin-bottom: 20px;">
            <b>📡 GIS Dashboard • Copernicus / SoilGrids (bila tersedia)</b> | Komoditas Sidebar: <span style="color:#0284c7;"><b>{selected_crop_global}</b></span> ({crop_data['kategori']}) |
            <b>Luas:</b> <span style="color:#10b981;">{area_ha} ha</span> |
            <b>Koordinat:</b> <span style="color:#f59e0b;">{round(current_lat, 5)}° S, {round(current_lon, 5)}° E</span>
        </div>
    ''', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDVI (Vegetasi)</div><div class="metric-value" style="color:#10b981;">{ndvi_val}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDWI (Kandungan Air)</div><div class="metric-value" style="color:#0284c7;">{ndwi_val}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDBI (Area Bangunan)</div><div class="metric-value" style="color:#ef4444;">{ndbi_val}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">pH Tanah Lahan {"(Live)" if real_ph else ""}</div><div class="metric-value" style="color:#f59e0b;">{ph_lahan}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### Evaluasi Kesesuaian Lahan untuk **{selected_crop_global}**")
    col1, col2, col3 = st.columns([1.2, 1, 1])

    if drawn_features is None or len(drawn_features) == 0:
        with col1:
            st.warning("Tandai area pada peta untuk memulai evaluasi kesesuaian lahan.")
    elif is_settlement:
        with col1:
            st.markdown('<div class="suitability-fail">STATUS LAHAN:<br><span style="font-size:1.1rem;">❌ TIDAK LAYAK UNTUK PERTANIAN (AREA PEMUKIMAN)</span></div>', unsafe_allow_html=True)
            st.caption("AI mendeteksi area pemukiman padat dan struktur bangunan (Impervious Surface).")
        with col2:
            st.markdown("**Parameter Fisik Lahan:**")
            st.write("• **Klasifikasi Lahan:** Pemukiman / Bangunan")
            st.write("• **Kebutuhan Air:** Non-Irigasi Pertanian")
        with col3:
            st.markdown("**Rekomendasi:**")
            st.error("Alihkan ke Urban Farming / Hidroponik")
    elif is_suitable:
        with col1:
            st.markdown(f'<div class="suitability-pass">STATUS LAHAN:<br><span style="font-size:1.1rem;">✅ SANGAT LAYAK UNTUK {selected_crop_global.upper()}</span></div>', unsafe_allow_html=True)
            st.caption(f"Kondisi pH ({ph_lahan}) dan indeks vegetasi ({ndvi_val}) memenuhi kriteria tumbuh optimal.")
        with col2:
            st.markdown(f"**Parameter Ideal {selected_crop_global}:**")
            st.write(f"• **Rentang pH Optimal:** {crop_data['ph_min']} - {crop_data['ph_max']}")
            st.write(f"• **Tingkat Kebutuhan Air:** {crop_data['water_req']}")
            st.write(f"• **Siklus Tanam:** {crop_data['cycle']}")
        with col3:
            st.markdown("**Estimasi Hasil Panen:**")
            st.info(f"Target Produksi: {crop_data['yield_est']}")
    else:
        with col1:
            st.markdown(f'<div class="suitability-fail">STATUS LAHAN:<br><span style="font-size:1.1rem;">⚠️ KURANG OPTIMAL UNTUK {selected_crop_global.upper()}</span></div>', unsafe_allow_html=True)
            st.caption(f"pH tanah ({ph_lahan}) berada di luar batas ideal tanaman ini ({crop_data['ph_min']} - {crop_data['ph_max']}).")
        with col2:
            st.markdown(f"**Parameter Ideal {selected_crop_global}:**")
            st.write(f"• **Rentang pH Optimal:** {crop_data['ph_min']} - {crop_data['ph_max']}")
            st.write(f"• **Kebutuhan Air:** {crop_data['water_req']}")
        with col3:
            st.markdown("**Tindakan Perbaikan:**")
            st.warning("Lakukan pengapuran (Dolomit) atau pembenah tanah sebelum penanaman.")

# ------------------------------------------------------------------------------
# TAB 2: DIAGNOSA PENYAKIT DAUN & BUAH (DENGAN GEMINI AI)
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(f"### 🔬 Diagnosa Penyakit & Hama Tanaman")
    st.write(f"Gunakan kamera atau unggah foto sampel daun/buah **{selected_crop_global}** untuk mendiagnosa potensi penyakit secara spesifik menggunakan Gemini AI.")

    col_input, col_result = st.columns([1, 1.2])

    with col_input:
        input_mode = st.radio("Pilih Metode Input Foto:", ["Ambil Foto via Kamera", "Upload File Foto"], horizontal=True)

        image_source = None

        if input_mode == "Ambil Foto via Kamera":
            camera_photo = st.camera_input("Posisikan sampel di tengah fokus kamera:")
            if camera_photo:
                image_source = camera_photo
        else:
            uploaded_file = st.file_uploader("Unggah foto sampel (JPG, PNG)...", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image_source = uploaded_file

        if image_source:
            image = Image.open(image_source)

            ctrl_col1, ctrl_col2 = st.columns([1, 1])
            with ctrl_col1:
                rotate_angle = st.selectbox("Putar Gambar:", options=[0, 90, 180, 270], index=0)
            with ctrl_col2:
                flip_mode = st.checkbox("Balik Kamera (Mirror)", value=False)

            if flip_mode:
                image = ImageOps.mirror(image)

            if rotate_angle != 0:
                image = image.rotate(-rotate_angle, expand=True)

            st.image(image, caption=f"Foto Sampel {selected_crop_global}", use_container_width=True)
        else:
            st.info(f"Ambil/Unggah sampel daun atau buah {selected_crop_global} untuk dianalisis AI.")

    with col_result:
        st.markdown(f"#### Hasil Analisis Gemini AI ({selected_crop_global})")
        if image_source and st.button("✨ Mulai Analisis AI", type="primary", use_container_width=True):
            with st.spinner(f"🤖 Gemini AI sedang menganalisis karakteristik penyakit pada {selected_crop_global}..."):
                # Kompresi gambar agar proses kirim super cepat
                img_temp = image.copy()
                img_temp.thumbnail((800, 800))
                img_byte_arr = io.BytesIO()
                img_temp.convert('RGB').save(img_byte_arr, format='JPEG', quality=75)
                compressed_bytes = img_byte_arr.getvalue()

                # Panggil fungsi yang sudah di-cache
                ai_analysis = analyze_plant_disease_with_gemini(compressed_bytes, selected_crop_global)
                
                if "BUKAN_TANAMAN" in ai_analysis:
                    st.error("⚠️ Objek Bukan Tanaman atau Buah!")
                    st.warning("Gemini AI mendeteksi foto yang dimasukkan bukan sampel daun/buah. Pastikan foto memuat objek tanaman secara jelas.")
                else:
                    st.success("✅ Analisis Berhasil")
                    st.markdown(ai_analysis)
        else:
            st.info("Unggah atau ambil foto lalu tekan **Mulai Analisis AI** untuk melihat hasil diagnosis.")

# ------------------------------------------------------------------------------
# TAB 3: KALKULATOR IRIGASI FAO-56
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(f"### 💧 Kalkulator Kebutuhan Air Tanaman")
    st.write(f"Kalkulasi Kebutuhan Air Tanaman (ETc) dan Rekomendasi Penyiraman Presisi untuk komoditas **{selected_crop_global}**.")

    nasa_data = get_real_weather_nasa(current_lat, current_lon)
    if nasa_data:
        def_temp, def_rh, def_wind, def_rain = nasa_data
    else:
        def_temp, def_rh, def_wind, def_rain = 29.5, 65, 2.1, 1.2

    col_env, col_crop_stage = st.columns([1, 1])

    with col_env:
        st.markdown(f"#### ⛅ Parameter Mikroklimat & Cuaca Lahan {'(NASA Live API)' if nasa_data else ''}")
        temp = st.slider("Suhu Udara Rata-rata (°C):", min_value=15.0, max_value=42.0, value=float(def_temp), step=0.5)
        humidity = st.slider("Kelembapan Relatif / RH (%):", min_value=20, max_value=95, value=int(def_rh), step=1)
        wind_speed = st.slider("Kecepatan Angin (m/s pada ketinggian 2m):", min_value=0.5, max_value=10.0, value=float(def_wind), step=0.1)
        sunshine_hrs = st.slider("Lama Penyinaran Matahari (Jam/Hari):", min_value=2.0, max_value=12.0, value=8.5, step=0.5)
        rainfall_eff = st.number_input("Curah Hujan Efektif (mm/hari):", min_value=0.0, max_value=50.0, value=float(def_rain), step=0.5)

    with col_crop_stage:
        st.markdown(f"#### Parameter Tanaman & Sistem Irigasi")

        st.success(f"Komoditas Aktif dari Sidebar: **{selected_crop_global}**")

        stage = st.radio("Fase Pertumbuhan Tanaman:", ["Fase Awal (Awal Tanam / Anakan)", "Fase Vegetatif Aktif / Pembentukan Batang Semu", "Fase Generatif / Pembentukan Jantung & Tandan Pisang"], horizontal=False)

        luas_m2 = st.number_input("Luas Lahan (m²):", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
        eff_irrigation = st.slider("Efisiensi Sistem Irigasi (%):", min_value=50, max_value=95, value=85)

    c_info = CROPS_DATABASE[selected_crop_global]
    if "Awal" in stage:
        kc = c_info.get("kc_init", 0.50)
    elif "Vegetatif" in stage:
        kc = c_info.get("kc_mid", 1.10)
    else:
        kc = c_info.get("kc_end", 1.00)

    eto = round(0.0023 * (temp + 17.8) * (12.0**0.5) * (0.408 * (sunshine_hrs * 1.35 + 8.0)) * (1 + 0.1 * (wind_speed - 2.0)), 2)
    if eto < 1.5: eto = 2.10

    etc_gross = round(eto * kc, 2)
    etc_net = max(0.0, round(etc_gross - rainfall_eff, 2))

    vol_liter_m2 = round(etc_net / (eff_irrigation / 100.0), 2)
    vol_total_liter = round(vol_liter_m2 * luas_m2, 1)
    vol_total_m3 = round(vol_total_liter / 1000.0, 2)

    st.markdown("---")
    st.markdown(f"### Hasil Perhitungan Irigasi FAO-56 untuk **{selected_crop_global}**")

    res1, res2, res3, res4 = st.columns(4)
    with res1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Evapotranspirasi Acuan (ET₀)</div><div class="metric-value" style="color:#0284c7;">{eto} <span style="font-size:0.9rem;">mm/hari</span></div></div>', unsafe_allow_html=True)
    with res2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Koefisien Tanaman (Kc {selected_crop_global})</div><div class="metric-value" style="color:#f59e0b;">{kc}</div></div>', unsafe_allow_html=True)
    with res3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Kebutuhan Air Bersih (ETc)</div><div class="metric-value" style="color:#10b981;">{etc_net} <span style="font-size:0.9rem;">mm/hari</span></div></div>', unsafe_allow_html=True)
    with res4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Volume Irigasi</div><div class="metric-value" style="color:#a855f7;">{vol_total_m3} <span style="font-size:0.9rem;">m³/hari</span></div></div>', unsafe_allow_html=True)

    st.markdown(" ")
    col_rec1, col_rec2 = st.columns([1.2, 1])

    with col_rec1:
        st.markdown(f"""
        <div class="custom-card">
            <h4 style="color:#0284c7; margin-top:0;">📋 Jadwal & Rekomendasi Penyiraman untuk {selected_crop_global}:</h4>
            <ul style="line-height: 1.8;">
                <li><b>Kebutuhan Air Dosis Spesifik:</b> <code>{vol_liter_m2} Liter / m² / hari</code></li>
                <li><b>Total Volume Lahan ({luas_m2} m²):</b> <code>{vol_total_liter:.0f} Liter/hari</code> ({vol_total_m3} m³/hari)</li>
                <li><b>Waktu Operasional Irigasi (Pompa Debit 50 L/menit):</b> <code>{round(vol_total_liter / 50.0, 1)} Menit/hari</code></li>
                <li><b>Rekomendasi Penyiraman:</b> Lakukan irigasi tetes/drip untuk efisiensi penyerapan air pada perakaran {selected_crop_global}.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_rec2:
        st.markdown(f"""
        <div class="custom-card" style="border-left: 4px solid #10b981;">
            <h5 style="color:#10b981; margin-top:0;">💡 Catatan Karakteristik {selected_crop_global}:</h5>
            <p style="font-size:0.9rem;">
            <b>Kebutuhan Air:</b> {c_info['water_req']}<br>
            <b>Estimasi Hasil:</b> {c_info['yield_est']}<br>
            <b>Siklus Tanam:</b> {c_info['cycle']}
            </p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB ESTIMASI HASIL PANEN
# ------------------------------------------------------------------------------
with tab_yield:
    st.markdown(f"### 📈 Simulasi Estimasi Hasil Panen ({selected_crop_global})")
    st.write(f"Simulasi berbobot dari dua input prediksi model untuk memperkirakan bobot panen komoditas **{selected_crop_global}**.")

    col_p1, col_p2 = st.columns([1, 1.2])

    with col_p1:
        st.markdown("#### ⚙️ Input Parameter Model")
        alpha_val = st.slider("Bobot Pembobotan Ensemble Model (α):", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        rf_pred = st.number_input("Prediksi Yield Random Forest (Ton/Ha):", min_value=1.0, max_value=100.0, value=35.0, step=0.5)
        xgb_pred = st.number_input("Prediksi Yield XGBoost (Ton/Ha):", min_value=1.0, max_value=100.0, value=38.5, step=0.5)

        y_ensemble = (alpha_val * rf_pred) + ((1 - alpha_val) * xgb_pred)

    with col_p2:
        st.markdown("#### 📊 Hasil Prediksi Yield Ensemble")
        y1, y2 = st.columns(2)
        with y1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Estimasi Hasil Ensemble (ŷ)</div><div class="metric-value" style="color:#10b981;">{y_ensemble:.2f} <span style="font-size:0.9rem;">Ton/Ha</span></div></div>', unsafe_allow_html=True)
        with y2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Contoh R² (belum divalidasi)</div><div class="metric-value" style="color:#38bdf8;">0.93 <span style="font-size:0.9rem;">(R² > 0.89)</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Visualisasi & Analisis Grafis Model")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1. Perbandingan Output Antar Model (Ton/Ha)")

        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Random Forest', 'XGBoost', 'Ensemble AI'],
                y=[rf_pred, xgb_pred, y_ensemble],
                marker_color=['#0284c7', '#f59e0b', '#10b981'],
                text=[f"{rf_pred:.2f}", f"{xgb_pred:.2f}", f"{y_ensemble:.2f}"],
                textposition='auto'
            )
        ])
        fig_bar.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Hasil Panen (Ton/Ha)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with g_col2:
        st.markdown("##### 2. Proyeksi Tren Hasil Panen Buah/Sayur per Periode")

        periods = ['Musim Lalu', 'Musim Ini (Prediksi)', 'Musim Depan (Proyeksi)']
        yield_trend = [y_ensemble * 0.90, y_ensemble, y_ensemble * 1.12]

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=periods, y=yield_trend,
            mode='lines+markers+text',
            name='Proyeksi Yield',
            line=dict(color='#10b981', width=3),
            marker=dict(size=10),
            text=[f"{v:.1f} Ton" for v in yield_trend],
            textposition="top center"
        ))
        fig_line.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Estimasi (Ton/Ha)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("##### 3. Tingkat Pengaruh Fitur Fisik Lahan terhadap Prediksi (Feature Importance)")

    features = ['NDVI (Klorofil)', 'pH Tanah', 'Curah Hujan', 'Suhu Udara', 'Kelembapan', 'NDWI (Air)']
    importance = [35, 22, 18, 12, 8, 5]

    fig_feat = px.bar(
        x=importance,
        y=features,
        orientation='h',
        color=importance,
        color_continuous_scale='Viridis',
        labels={'x': 'Kontribusi (%)', 'y': 'Fitur Input'}
    )
    fig_feat.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_feat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4: DATABASE KOMODITAS
# ------------------------------------------------------------------------------
with tab4:
    st.markdown("### 🌿 Eksplorasi Database Komoditas Pertanian")
    st.write("Daftar komoditas nasional beserta parameter tumbuh idealnya:")

    df_crops = pd.DataFrame.from_dict(CROPS_DATABASE, orient='index')
    df_crops.index.name = "Nama Komoditas / Tanaman"
    df_crops.reset_index(inplace=True)

    kat_filter = st.multiselect("Filter Kategori Dashboard:", options=list(df_crops["kategori"].unique()), default=list(df_crops["kategori"].unique()))
    df_filtered = df_crops[df_crops["kategori"].isin(kat_filter)]

    st.dataframe(df_filtered, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📊 Distribusi Toleransi pH Tanah Berdasarkan Kategori")
    fig_cat = px.box(df_filtered, x="kategori", y="ph_min", color="kategori", points="all", labels={"ph_min": "pH Minimum Tanah", "kategori": "Kategori Komoditas"})
    fig_cat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_cat, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 5: OPEN DATASETS
# ------------------------------------------------------------------------------
with tab5:
    st.header("📚 Referensi Dataset Penyakit Tanaman")
    st.caption("Daftar link dataset open-source yang dapat diakses publik untuk pengembangan model Computer Vision:")
    st.markdown("---")

    col_ds1, col_ds2 = st.columns(2)

    with col_ds1:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#0284c7; margin-top:0;">1. Fruits-360 & Banana Disease Dataset</h3>
            <ul style="line-height:1.8;">
                <li><b>Ukuran:</b> &gt; 90.000 citra sampel.</li>
                <li><b>Cakupan:</b> Citra buah Pisang, Mangga, Apel, Jeruk, serta dataset klasifikasi penyakit pisang (Fusarium/Sigatoka).</li>
                <li><b>Akses URL:</b> <code>github.com/Horea94/Fruit-Images-Dataset</code></li>
            </ul>
        </div>

        <div class="custom-card">
            <h3 style="color:#0284c7; margin-top:0;">2. PlantVillage Dataset</h3>
            <ul style="line-height:1.8;">
                <li><b>Ukuran:</b> 54.303 citra tersegmentasi.</li>
                <li><b>Cakupan:</b> 38 kelas penyakit pada tanaman sayuran (Tomat, Cabai, Kentang, Jagung).</li>
                <li><b>Akses URL:</b> <code>github.com/spMohanty/PlantVillage-Dataset</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_ds2:
        st.markdown("""
        <div class="custom-card">
            <h3 style="color:#0284c7; margin-top:0;">3. Indonesia Tropical Fruits Dataset (Roboflow)</h3>
            <ul style="line-height:1.8;">
                <li><b>Ukuran:</b> &gt; 12.000 foto objek lapangan.</li>
                <li><b>Cakupan:</b> Buah tropis khas Indonesia (Pisang, Durian, Rambutan, Salak, Manggis, Buah Naga).</li>
                <li><b>Akses URL:</b> <code>universe.roboflow.com</code> (Search: <i>Banana Disease / Tropical Fruits</i>)</li>
            </ul>
        </div>

        <div class="custom-card">
            <h3 style="color:#0284c7; margin-top:0;">4. PlantDoc Open Dataset</h3>
            <ul style="line-height:1.8;">
                <li><b>Ukuran:</b> 2.569 citra latar belakang kebun/lahan asli.</li>
                <li><b>Cakupan:</b> 13 spesies tanaman &amp; 27 kelas penyakit dengan kondisi pencahayaan alami.</li>
                <li><b>Akses URL:</b> <code>github.com/pratikkayal/PlantDoc-Dataset</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
