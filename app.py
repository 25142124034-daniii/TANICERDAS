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
import os
import json
import html
import hashlib
from datetime import datetime, timedelta, timezone
from google import genai
from google.genai import types

# Credentials are read from environment or Streamlit secrets, never from source code.
def get_credential(name):
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

GEMINI_API_KEY = get_credential("GEMINI_API_KEY")

import time

@st.cache_data(show_spinner=False)
def analyze_plant_disease_with_gemini(image_bytes_data, crop_name):
    if GEMINI_API_KEY == "MASUKKAN_API_KEY_GEMINI_KAMU_DI_SINI" or not GEMINI_API_KEY:
        return "API Key Gemini belum dipasang.** Silakan isi `GEMINI_API_KEY` di Streamlit Secrets."
    
    # Coba pakai model gemini-3.8-flash dulu, kalau sibuk otomatis coba gemini-3.5-flash
    models_to_try = ['gemini-3.8-flash', 'gemini-3.5-flash-lite']
    
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
                    return f"Gagal menghubungkan ke Gemini AI: {e}"

    return "Server Gemini AI sedang sangat sibuk. Silakan tekan tombol analisis sekali lagi dalam beberapa detik."
# ==============================================================================
# FUNGSI INTEGRASI COPERNICUS SENTINEL-2 API & SOILGRIDS
# ==============================================================================
COPERNICUS_CLIENT_ID = get_credential("COPERNICUS_CLIENT_ID")
COPERNICUS_CLIENT_SECRET = get_credential("COPERNICUS_CLIENT_SECRET")

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
        end_date = datetime.now(timezone.utc).date() - timedelta(days=21)
        start_date = end_date - timedelta(days=6)
        url = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
               "parameters=T2M,RH2M,WS2M,ALLSKY_SFC_SW_DW,PRECTOTCORR&community=AG&"
               f"longitude={lon}&latitude={lat}&start={start_date:%Y%m%d}&end={end_date:%Y%m%d}&format=JSON")
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()['properties']['parameter']
            def mean_available(field):
                good = [v for v in data[field].values() if isinstance(v, (int, float)) and v > -900]
                return np.mean(good) if good else float("nan")
            temp = mean_available('T2M')
            humidity = mean_available('RH2M')
            wind = mean_available('WS2M')
            rainfall = mean_available('PRECTOTCORR')
            if not all(np.isfinite(v) for v in [temp, humidity, wind, rainfall]):
                return None
            return round(temp, 1), round(humidity, 1), round(wind, 1), round(rainfall, 1)
    except Exception:
        pass
    return None

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="TaniCerdas | Agriculture Intelligence",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');
:root {
  --bg:#071A16; --sidebar:#091E19; --panel:#102A23; --panel2:#14352C;
  --stroke:#28473E; --stroke-soft:#1E3A30; --ink:#F1F7F3;
  --muted:#9CB8AB; --muted2:#78998A; --accent:#34D399; --cyan:#22D3EE;
  --warning:#FBBF24; --danger:#FB7185; --surface-hover:#193D31;
}
html,body,[class*="css"] {font-family:'DM Sans',sans-serif!important;}
[data-testid="stAppViewContainer"],.stApp {background:var(--bg)!important;color:var(--ink)!important;}
[data-testid="stMain"] {background:transparent!important;}
[data-testid="stHeader"] {background:rgba(7,26,22,.93)!important;backdrop-filter:blur(16px);border-bottom:1px solid var(--stroke-soft);}
[data-testid="stToolbar"] {color:var(--muted)!important;}
.block-container {max-width:1600px!important;padding:5.25rem 3rem 4rem!important;}
/* Streamlit reserves the top 3.75rem for its application toolbar/header.
   Keep the first dashboard row below that area, not behind the header. */
[data-testid="stMain"] [data-testid="stMainBlockContainer"],
[data-testid="stMain"] .block-container {padding-top:5.25rem!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:1.5rem!important;}
[data-testid="stSidebar"] {background:var(--sidebar)!important;border-right:1px solid var(--stroke)!important;}
[data-testid="stSidebar"] .block-container {padding:1.4rem 1.15rem!important;}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.5rem 1.25rem 2rem!important;}
[data-testid="stSidebar"] hr {border-color:var(--stroke)!important;}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:var(--muted)!important;}
h1,h2,h3,h4,h5,h6 {color:var(--ink)!important;font-family:'Manrope',sans-serif!important;letter-spacing:-.035em;}
h3{font-weight:750!important;} p,li {color:#C7D8CF;} strong,b{color:var(--ink)}
small,[data-testid="stCaptionContainer"]{color:var(--muted)!important;}
a {color:var(--accent)!important;text-decoration:none;} a:hover{text-decoration:underline}
hr {border-color:var(--stroke-soft)!important;}
[data-testid="stMarkdownContainer"] code,code {color:var(--accent)!important;background:#0B3B2A!important;border:1px solid #1E5840;border-radius:6px;padding:3px 6px;}
.topbar {display:flex;flex-wrap:wrap;row-gap:12px;gap:16px;align-items:center;justify-content:space-between;border-bottom:1px solid var(--stroke-soft);padding:0 0 18px;margin-bottom:24px;min-height:40px;}
.crumb {font-size:11px;letter-spacing:.11em;font-weight:800;color:var(--muted2);text-transform:uppercase;line-height:1.65;min-width:0;overflow-wrap:anywhere}
.crumb b {color:var(--ink);font-weight:800}
.topright {display:flex;gap:10px;align-items:center;color:var(--muted);font-size:12px;flex-shrink:0}
.topdot {width:7px;height:7px;display:inline-block;border-radius:99px;background:var(--accent);box-shadow:0 0 0 4px rgba(52,211,153,.1)}
.topbadge {border:1px solid var(--stroke);border-radius:99px;padding:8px 13px;color:#CBE1D4;background:#102A23;letter-spacing:.035em}
.brand {display:flex;align-items:center;gap:10px;margin:5px 0 26px;min-width:0;} .brand>span:last-child{min-width:0;flex:1;}
.brand-symbol {display:grid;place-items:center;width:40px;height:40px;background:var(--accent);border-radius:12px;color:#05251B;flex-shrink:0;}
.brand-title {font-family:'Manrope',sans-serif;font-size:20px;color:#F3FFF7;font-weight:800;letter-spacing:-.065em;line-height:1.05}
.brand-sub {font-size:8px;letter-spacing:.09em;line-height:1.55;color:#8DAE9D;text-transform:uppercase;margin-top:5px;font-weight:800;overflow-wrap:normal;}
.sidebar-eyebrow {color:var(--muted2);font-size:10px;letter-spacing:.2em;text-transform:uppercase;font-weight:800;margin:19px 0 10px;}
.sidebar-info {background:#102B22;border:1px solid #27483B;border-radius:14px;padding:16px 16px;margin-top:20px;}
.sidebar-info .tag {text-transform:uppercase;letter-spacing:.12em;font-size:10px;font-weight:800;color:var(--accent);margin-bottom:12px;}
.sidebar-info .crop {font-size:15px;font-weight:700;color:var(--ink);line-height:1.4;margin-bottom:13px;}
.sidebar-row {display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-top:1px solid #214235;font-size:12px;color:var(--muted)}
.sidebar-row b {font-size:12px;font-weight:650;text-align:right;max-width:130px;color:#D5E8DD;}
.sidebar-footer {margin-top:26px;padding-top:20px;border-top:1px solid var(--stroke);color:#789788;font-size:11px;line-height:1.8;}
.hero {position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1.28fr) minmax(290px,.72fr);gap:26px;min-height:304px;background:radial-gradient(circle at 91% 10%,rgba(34,211,238,.09),transparent 38%),linear-gradient(117deg,#10392B 0%,#0B2A23 65%,#08221C 100%);border:1px solid #2B5140;border-radius:22px;padding:36px 40px;margin-bottom:25px;}
.hero:before{content:'';position:absolute;top:0;bottom:0;right:0;width:49%;background-image:linear-gradient(rgba(62,182,135,.075) 1px,transparent 1px),linear-gradient(90deg,rgba(62,182,135,.075) 1px,transparent 1px);background-size:28px 28px;mask-image:linear-gradient(to left,black,transparent);pointer-events:none;}
.hero > * {position:relative;z-index:1;min-width:0;}
.eyebrow {font-size:10px;line-height:1.4;letter-spacing:.20em;color:#7BEAC1;font-weight:800;text-transform:uppercase;}
.hero h1 {font-family:'Manrope',sans-serif!important;color:#F5FDF8!important;font-size:clamp(28px,3.2vw,48px);letter-spacing:-.065em;line-height:1.13;font-weight:800;max-width:640px;margin:14px 0 14px;}
.hero h1 span {color:#76E5B8;}
.hero p{color:#A9C7B8;font-size:14px;line-height:1.75;max-width:605px;margin:0;}
.hero-tags{display:flex;gap:7px;flex-wrap:wrap;margin-top:24px;}
.hero-tags span{font-size:11px;color:#BEE1D0;letter-spacing:.02em;border:1px solid #326047;border-radius:7px;background:#164332;padding:8px 11px;}
.hero-module {align-self:center;border:1px solid #38624D;border-radius:15px;background:rgba(9,32,26,.82);padding:18px 19px 15px;min-height:214px;box-shadow:0 12px 40px rgba(0,0,0,.13);}
.hero-module .label{font-size:10px;text-transform:uppercase;letter-spacing:.17em;color:#95B6A4;font-weight:800;}
.hero-module .value{font-size:34px;font-weight:800;color:#F1FFF6;letter-spacing:-.07em;line-height:1.2;margin-top:10px;}
.hero-module .value small{font-size:13px;color:#8BBDA1;font-weight:600;letter-spacing:0}
.mini-wave{height:65px;display:flex;align-items:end;gap:5px;margin-top:16px;padding-bottom:2px;border-bottom:1px solid #355344;}
.mini-wave i{display:block;flex:1;border-radius:3px 3px 0 0;background:linear-gradient(180deg,#4CCB9B,#226C53);opacity:.85}
.mini-wave i:nth-child(3n){background:linear-gradient(180deg,#38BFC1,#226F77);}
.hero-module .foot{display:flex;gap:8px;justify-content:space-between;color:#9DB6A8;font-size:11px;margin-top:11px}
.hero-module .foot strong{color:#5CE2A6;font-weight:700}
.kpi-row {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:0 0 27px;}
.kpi {background:var(--panel);border:1px solid var(--stroke);padding:17px 19px;border-radius:14px;min-width:0;}
.kpi small{display:block;color:#8BA899;font-size:10px;text-transform:uppercase;letter-spacing:.14em;font-weight:800;margin-bottom:10px;}
.kpi strong{display:block;font-size:15px;font-weight:700;color:#E7F9ED;overflow-wrap:anywhere;}
.kpi .detail{display:block;color:#7C9B8C;font-size:11px;margin-top:6px}
.section-head {display:flex;align-items:end;justify-content:space-between;gap:20px;border-bottom:1px solid var(--stroke);padding-bottom:15px;margin:18px 0 20px;}
.section-head .index{display:inline-block;font-weight:800;font-size:10px;letter-spacing:.2em;color:#58CFA4;margin-bottom:6px}
.section-head h2{margin:0 0 5px;font-size:23px;letter-spacing:-.05em;}
.section-head p{color:var(--muted);margin:0;font-size:12px;line-height:1.55}
.section-chip{display:block;flex-shrink:0;border:1px solid #2A5643;border-radius:99px;background:#143B2C;color:#A5DABF;padding:8px 12px;font-size:11px;font-weight:750;}
.custom-card,.metric-card {background:var(--panel)!important;border:1px solid var(--stroke)!important;box-shadow:none!important;border-radius:14px!important;margin:0 0 16px;padding:20px;transition:border-color .18s,background .18s;}
.custom-card:hover,.metric-card:hover {border-color:#3A6551!important;background:var(--panel2)!important;}
.custom-card h3,.custom-card h4,.custom-card h5 {color:#E8F4EC!important;font-size:16px;line-height:1.55;}
.custom-card li,.custom-card p {color:#BBD0C3!important;font-size:12px;line-height:1.9;}
.metric-card{min-height:112px;display:flex;flex-direction:column;justify-content:space-between;padding:18px 20px;}
.metric-label {font-size:11px;color:#91B3A1;font-weight:750;letter-spacing:.01em;line-height:1.5}
.metric-value {font-size:clamp(24px,2.3vw,31px);font-family:'Manrope',sans-serif;font-weight:800;letter-spacing:-.06em;color:#67E8AE!important;}
.metric-value span {color:#A5C7B3!important;font-weight:650!important;font-size:11px!important;letter-spacing:0;}
.stTabs [data-baseweb="tab-list"] {display:flex;gap:4px;overflow-x:auto;flex-wrap:nowrap;border-bottom:1px solid var(--stroke);background:transparent!important;padding:0 0 4px;margin:0 0 20px;scrollbar-width:thin;}
.stTabs [data-baseweb="tab"] {height:46px;flex-shrink:0;border-radius:7px 7px 0 0;padding:0 17px!important;background:transparent!important;font-size:12px;font-weight:700;white-space:nowrap;color:#9DB3A8!important;}
.stTabs [data-baseweb="tab"]:hover {background:#13372A!important;color:#E0F7EB!important;}
.stTabs [aria-selected="true"] {color:#68DDB0!important;border-bottom:2px solid var(--accent)!important;background:#112F25!important;}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"] {display:none!important;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--stroke)!important;}
[data-testid="stForm"],div[data-testid="stExpander"] {background:var(--panel)!important;border:1px solid var(--stroke)!important;border-radius:13px;}
[data-baseweb="input"] input,[data-baseweb="select"] input,[data-baseweb="textarea"] textarea {color:var(--ink)!important;background:#112F25!important;}
[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div {background:#102D24!important;border:1px solid #2D5142!important;border-radius:10px!important;color:var(--ink)!important;}
[data-baseweb="select"] svg {fill:#89BCA2!important;}
[data-baseweb="select"] [role="combobox"],[data-baseweb="select"] [data-testid="stMarkdownContainer"] {color:#EFFCF5!important;}
[data-baseweb="popover"],[data-baseweb="menu"],[role="listbox"] {background:#122E25!important;color:var(--ink)!important;}
[role="option"] {background:#122E25!important;color:#E5F6EC!important;}
[role="option"]:hover,[aria-selected="true"][role="option"] {background:#245642!important;}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{background:var(--accent)!important;border-color:var(--accent)!important;}
[data-testid="stRadio"] label,[data-testid="stCheckbox"] label {color:var(--ink)!important;}
[data-testid="stButton"] button[kind="primary"],.stDownloadButton button[kind="primary"]{background:var(--accent)!important;border:1px solid var(--accent)!important;border-radius:10px;color:#09251C!important;font-weight:800;}
[data-testid="stButton"] button,.stDownloadButton button {min-height:43px;border:1px solid #365A4A;border-radius:10px;background:#15392D;color:#D1EEDC;font-weight:700;transition:transform .18s,border-color .18s,background .18s;}
[data-testid="stButton"] button:hover,.stDownloadButton button:hover {border-color:var(--accent);background:#1A4A39;transform:translateY(-1px)}
[data-testid="stFileUploader"],[data-testid="stCameraInput"] {background:var(--panel);border:1px solid var(--stroke);border-radius:15px;padding:10px;}
[data-testid="stFileUploaderDropzone"] {background:#0D241E!important;border:1px dashed #3A6552!important;border-radius:11px;}
[data-testid="stDataFrame"],[data-testid="stPlotlyChart"],iframe {border-radius:13px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:11px;border:1px solid var(--stroke);}
.suitability-pass,.suitability-fail {border-radius:12px;padding:19px;font-size:13px;line-height:1.9;font-weight:800;}
.suitability-pass {background:#0F3A2A;border:1px solid #2D7557;color:#7EEBB6;}
.suitability-fail {background:#392923;border:1px solid #7C5547;color:#FDC3A0;}
.data-note {background:#0D281F;border:1px solid #28503D;border-radius:11px;padding:12px 15px;color:#9FC4AD;font-size:11px;line-height:1.7;margin:8px 0 18px;}
.data-note b{color:#C8ECD6}
.page-footer {margin-top:36px;padding:18px 0;border-top:1px solid var(--stroke);display:flex;flex-wrap:wrap;gap:12px;justify-content:space-between;color:#789789;font-size:11px;}
@media(max-width:1100px){.hero{grid-template-columns:1fr}.hero-module{display:none}.block-container{padding:5.25rem 1.6rem 3rem!important;}[data-testid="stMain"] [data-testid="stMainBlockContainer"],[data-testid="stMain"] .block-container{padding:5.25rem 1.6rem 3rem!important;}}
@media(max-width:700px){.block-container{padding:5.1rem 1rem 2.5rem!important}[data-testid="stMain"] [data-testid="stMainBlockContainer"],[data-testid="stMain"] .block-container{padding:5.1rem 1rem 2.5rem!important}.hero{padding:25px 20px;min-height:0}.hero h1{font-size:29px}.hero p{font-size:13px}.topbadge{display:none}.kpi-row{grid-template-columns:1fr;gap:9px}.section-chip{display:none}.section-head h2{font-size:21px}.metric-card{min-height:90px}.stTabs [data-baseweb="tab"]{padding:0 11px!important;font-size:11px}}
[data-testid="stSidebar"] .block-container {padding:1.4rem 1.15rem!important;}
@media(prefers-reduced-motion:reduce){*,*:before,*:after{scroll-behavior:auto!important;transition-duration:0s!important;animation-duration:0s!important}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE KOMODITAS PERTANIAN, BUAH & SAYURAN LENGKAP INDONESIA
# ==============================================================================
CROPS_DATABASE = {
    #  BUAH-BUAHAN TROPIS & KHAS INDONESIA
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

    #  SAYUR-SAYURAN INDONESIA
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

    #  TANAMAN PANGAN
    "Padi Sawah": {"kategori": "Tanaman Pangan", "ph_min": 5.5, "ph_max": 7.0, "ndvi_min": 0.65, "water_req": "High (Irigasi Basah)", "yield_est": "6.0 - 7.5 Ton/Ha", "cycle": "110-120 Hari", "kc_init": 1.05, "kc_mid": 1.20, "kc_end": 0.90},
    "Padi Gogo (Lahan Kering)": {"kategori": "Tanaman Pangan", "ph_min": 5.0, "ph_max": 6.5, "ndvi_min": 0.55, "water_req": "Medium (Tadah Hujan)", "yield_est": "3.5 - 4.5 Ton/Ha", "cycle": "110-125 Hari", "kc_init": 0.40, "kc_mid": 1.15, "kc_end": 0.35},
    "Jagung": {"kategori": "Tanaman Pangan", "ph_min": 5.8, "ph_max": 7.2, "ndvi_min": 0.60, "water_req": "Medium", "yield_est": "7.0 - 9.0 Ton/Ha", "cycle": "100-110 Hari", "kc_init": 0.30, "kc_mid": 1.20, "kc_end": 0.60},
    "Kedelai": {"kategori": "Tanaman Pangan", "ph_min": 6.0, "ph_max": 6.8, "ndvi_min": 0.50, "water_req": "Medium", "yield_est": "1.8 - 2.5 Ton/Ha", "cycle": "80-90 Hari", "kc_init": 0.40, "kc_mid": 1.15, "kc_end": 0.50},
    "Ubi Kayu (Singkong)": {"kategori": "Tanaman Pangan", "ph_min": 4.5, "ph_max": 7.0, "ndvi_min": 0.45, "water_req": "Low", "yield_est": "20.0 - 35.0 Ton/Ha", "cycle": "8-10 Bulan", "kc_init": 0.30, "kc_mid": 1.10, "kc_end": 0.50},

    #  PERKEBUNAN & REMPAH
    "Kelapa Sawit": {"kategori": "Perkebunan", "ph_min": 4.0, "ph_max": 6.5, "ndvi_min": 0.70, "water_req": "Very High", "yield_est": "20.0 - 25.0 Ton/Ha", "cycle": "Tahunan (3-25 Thn)", "kc_init": 0.90, "kc_mid": 1.00, "kc_end": 0.95},
    "Kopi Arabika": {"kategori": "Perkebunan", "ph_min": 5.5, "ph_max": 6.5, "ndvi_min": 0.65, "water_req": "Medium-High", "yield_est": "1.0 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.90, "kc_mid": 1.05, "kc_end": 0.95},
    "Kakao (Cokelat)": {"kategori": "Perkebunan", "ph_min": 6.0, "ph_max": 7.5, "ndvi_min": 0.65, "water_req": "High", "yield_est": "1.2 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.90},
    "Karet": {"kategori": "Perkebunan", "ph_min": 4.5, "ph_max": 5.5, "ndvi_min": 0.70, "water_req": "High", "yield_est": "1.5 - 2.0 Ton/Ha", "cycle": "Tahunan", "kc_init": 0.80, "kc_mid": 1.00, "kc_end": 0.85}
}

# ==============================================================================
# WORKSPACE SIDEBAR & PRODUCT NAVIGATION
# ==============================================================================
def section_header(number, heading, description, tag):
    st.markdown(f"""
    <div class="section-head">
      <div><div class="index">MODULE {number} / TANICERDAS</div>
        <h2>{html.escape(heading)}</h2><p>{html.escape(description)}</p></div>
      <div class="section-chip">{html.escape(tag)}</div>
    </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div class="brand">
      <span class="brand-symbol" aria-hidden="true">
        <svg xmlns="http://www.w3.org/2000/svg" width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 21v-9"/><path d="M12 12c-4.2 0-6.5-3.5-6.2-8 4.7.1 7.1 2.7 6.2 8Z"/>
          <path d="M12 16c4.2 0 6.5-3.5 6.2-8-4.7.1-7.1 2.7-6.2 8Z"/>
        </svg>
      </span>
      <span><div class="brand-title">TaniCerdas</div><div class="brand-sub">Agriculture Intelligence</div></span>
    </div>
    <div class="sidebar-eyebrow">Workspace settings</div>
    """, unsafe_allow_html=True)
    categories = sorted({item["kategori"] for item in CROPS_DATABASE.values()})
    default_cat_idx = categories.index("Buah-Buahan") if "Buah-Buahan" in categories else 0
    selected_cat = st.selectbox("Kategori komoditas", categories, index=default_cat_idx)
    filtered_crops = sorted([k for k, v in CROPS_DATABASE.items() if v["kategori"] == selected_cat])
    banana_key = "Pisang (Cavendish/Ambon/Raja/Barangan)"
    default_crop_idx = filtered_crops.index(banana_key) if banana_key in filtered_crops else 0
    selected_crop_global = st.selectbox("Komoditas aktif", filtered_crops, index=default_crop_idx)
    crop_data = CROPS_DATABASE[selected_crop_global]
    st.markdown(f"""
      <div class="sidebar-info">
       <div class="tag">Active crop profile</div>
       <div class="crop">{html.escape(selected_crop_global)}</div>
       <div class="sidebar-row"><span>Kategori</span><b>{html.escape(crop_data['kategori'])}</b></div>
       <div class="sidebar-row"><span>Siklus tanam</span><b>{html.escape(crop_data['cycle'])}</b></div>
       <div class="sidebar-row"><span>Potensi hasil</span><b>{html.escape(crop_data['yield_est'])}</b></div>
      </div>
      <div class="sidebar-footer">TANICERDAS / PRECISION SUITE<br>Research and planning workspace<br>Version 4.0 — Dark Emerald</div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="topbar"><span class="crumb">TANICERDAS / <b>INTELLIGENCE WORKSPACE</b></span>
  <span class="topright"><span class="topdot"></span><span class="topbadge">PRECISION SUITE &nbsp; / &nbsp; 2026</span></span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<section class="hero">
 <div>
  <div class="eyebrow">AGRICULTURAL INTELLIGENCE &nbsp; / &nbsp; DATA-DRIVEN OPERATIONS</div>
  <h1>Presisi untuk<br><span>setiap hektare.</span></h1>
  <p>Satu ruang kerja untuk memahami kondisi lahan, mengevaluasi kesehatan tanaman, merencanakan irigasi, dan mengeksplorasi proyeksi hasil panen.</p>
  <div class="hero-tags"><span>Spatial analytics</span><span>Crop diagnostics</span><span>Water planning</span><span>Yield insights</span></div>
 </div>
 <div class="hero-module" aria-label="Ringkasan kapabilitas platform">
   <div class="label">ACTIVE INTELLIGENCE MODULES</div>
   <div class="value">06 <small>/ modules</small></div>
   <div class="mini-wave" aria-hidden="true">
     <i style="height:33%"></i><i style="height:44%"></i><i style="height:38%"></i><i style="height:59%"></i>
     <i style="height:52%"></i><i style="height:67%"></i><i style="height:61%"></i><i style="height:82%"></i>
     <i style="height:73%"></i><i style="height:91%"></i><i style="height:78%"></i><i style="height:95%"></i>
   </div>
   <div class="foot"><span>Spatial / Climate / Crop</span><strong>RESEARCH MODE</strong></div>
 </div>
</section>
<div class="kpi-row">
 <div class="kpi"><small>Current commodity</small><strong>{html.escape(selected_crop_global)}</strong><span class="detail">Profil tanaman terpilih</span></div>
 <div class="kpi"><small>Growth cycle</small><strong>{html.escape(crop_data['cycle'])}</strong><span class="detail">Referensi basis data komoditas</span></div>
 <div class="kpi"><small>Potential yield</small><strong>{html.escape(crop_data['yield_est'])}</strong><span class="detail">Estimasi literatur, bukan pengukuran lapangan</span></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab_yield, tab4, tab5 = st.tabs([
    "Pemetaan lahan", "Diagnosis tanaman", "Perencanaan irigasi",
    "Estimasi panen", "Basis data tanaman", "Referensi dataset"
])

# ------------------------------------------------------------------------------
# TAB 1: KELAYAKAN LAHAN
# ------------------------------------------------------------------------------
with tab1:
    section_header("01", "Pemetaan & kelayakan lahan", "Tandai batas lahan, evaluasi indikator, dan kaji kesesuaiannya terhadap komoditas.", "GEOSPATIAL")
    instruction, basemap_control = st.columns([2.3, 1])
    with instruction:
        st.markdown("**Peta interaktif**")
        st.caption("Gambar poligon atau persegi panjang pada peta. Titik awal peta adalah lokasi contoh di Jawa Timur.")
    with basemap_control:
        basemap_style = st.radio("Lapisan peta", ["Satelit", "Jalan"], horizontal=True)

    lat_center, lon_center = -7.4965, 112.6316

    m = folium.Map(
        location=[lat_center, lon_center],
        zoom_start=17,
        max_zoom=19,
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}" if basemap_style == "Satelit" else "OpenStreetMap"),
        attr=("Esri World Imagery" if basemap_style == "Satelit" else "OpenStreetMap contributors")
    )

    draw = Draw(
        export=True,
        filename='field_polygon.geojson',
        position='topleft',
        draw_options={'polyline': False, 'polygon': True, 'rectangle': True, 'circle': False, 'marker': True, 'circlemarker': False},
        edit_options={'edit': True, 'remove': True}
    )
    draw.add_to(m)

    st_data = st_folium(m, width="100%", height=505, key=f"eos_map_precision_{basemap_style}")

    drawn_features = st_data.get("all_drawings") if st_data else None
    area_ha = 0.0
    current_lat = lat_center
    current_lon = lon_center

    if drawn_features and len(drawn_features) > 0:
        last_draw = drawn_features[-1]
        geom_type = last_draw['geometry']['type']

        if geom_type == 'Point':
            current_lon, current_lat = last_draw['geometry']['coordinates']
            area_ha = 0.0
        else:
            coords = last_draw['geometry']['coordinates'][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            current_lat = sum(lats) / len(lats)
            current_lon = sum(lons) / len(lons)
            # Polygon area on an equirectangular tangent plane (approximation).
            center_lat_rad = np.deg2rad(current_lat)
            pts = [(np.deg2rad(c[0]) * 6371008.8 * np.cos(center_lat_rad),
                    np.deg2rad(c[1]) * 6371008.8) for c in coords]
            signed_area = sum(x1*y2 - x2*y1 for (x1,y1),(x2,y2) in zip(pts, pts[1:]+pts[:1])) / 2.0
            area_ha = round(abs(signed_area) / 10000.0, 3)

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

    st.markdown("""<div class="data-note"><b>Transparansi data.</b> Jika indeks satelit atau pH tidak tersedia, dashboard menggunakan nilai simulasi untuk mendemonstrasikan antarmuka; hasil tersebut tidak boleh ditafsirkan sebagai pengukuran aktual atau dasar keputusan agronomis.</div>""", unsafe_allow_html=True)

    if drawn_features:
        st.download_button("Ekspor batas lahan (GeoJSON)", json.dumps({"type": "FeatureCollection", "features": drawn_features}, indent=2), file_name="tanicerdas_lahan.geojson", mime="application/geo+json")
    st.markdown(f'''
        <div class="custom-card" style="padding:14px 20px;display:flex;flex-wrap:wrap;gap:12px 26px;align-items:center;">
            <span style="color:#90B4A1;font-size:11px;letter-spacing:.1em;font-weight:800;">FIELD OVERVIEW</span>
            <span style="color:#D6F4E2;">Area <b>{area_ha:.3f} ha</b></span>
            <span style="color:#D6F4E2;">Latitude <b>{current_lat:.5f}</b></span>
            <span style="color:#D6F4E2;">Longitude <b>{current_lon:.5f}</b></span>
        </div>
    ''', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDVI (Vegetasi{ " · Simulasi" if real_ndvi is None else " · Data" })</div><div class="metric-value" style="color:#10b981;">{ndvi_val}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDWI (Air · Simulasi)</div><div class="metric-value" style="color:#0284c7;">{ndwi_val}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">NDBI (Bangunan · Simulasi)</div><div class="metric-value" style="color:#ef4444;">{ndbi_val}</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">pH Tanah {"· SoilGrids" if real_ph is not None else "· Simulasi"}</div><div class="metric-value" style="color:#f59e0b;">{ph_lahan}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### Evaluasi kesesuaian untuk {selected_crop_global}")
    if real_ndvi is None or real_ph is None:
        st.caption("Klasifikasi di bawah merupakan simulasi berbasis nilai contoh karena sebagian sumber data belum tersedia.")
    col1, col2, col3 = st.columns([1.2, 1, 1])

    if drawn_features is None or len(drawn_features) == 0:
        with col1:
            st.warning("Tandai area pada peta untuk memulai evaluasi kesesuaian lahan.")
    elif is_settlement:
        with col1:
            st.markdown('<div class="suitability-fail">STATUS LAHAN:<br><span style="font-size:1.1rem;"> INDIKASI AREA TERBANGUN UNTUK PERTANIAN (AREA PEMUKIMAN)</span></div>', unsafe_allow_html=True)
            st.caption("Indikator yang tersedia menunjukkan dominasi permukaan terbangun; periksa dengan survei lapangan.")
        with col2:
            st.markdown("**Parameter Fisik Lahan:**")
            st.write("• **Klasifikasi Lahan:** Pemukiman / Bangunan")
            st.write("• **Kebutuhan Air:** Non-Irigasi Pertanian")
        with col3:
            st.markdown("**Rekomendasi:**")
            st.error("Alihkan ke Urban Farming / Hidroponik")
    elif is_suitable:
        with col1:
            st.markdown(f'<div class="suitability-pass">STATUS LAHAN / {"SIMULASI" if real_ndvi is None or real_ph is None else "ANALISIS"}:<br><span style="font-size:1.1rem;"> SESUAI UNTUK {selected_crop_global.upper()}</span></div>', unsafe_allow_html=True)
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
            st.markdown(f'<div class="suitability-fail">STATUS LAHAN / {"SIMULASI" if real_ndvi is None or real_ph is None else "ANALISIS"}:<br><span style="font-size:1.1rem;"> KURANG OPTIMAL UNTUK {selected_crop_global.upper()}</span></div>', unsafe_allow_html=True)
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
    section_header("02", "Diagnosis kesehatan tanaman", "Analisis visual gejala pada daun dan buah dengan bantuan model multimodal.", "CROP HEALTH")
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
        st.markdown(f"#### Laporan analisis / {selected_crop_global}")
        if image_source:
            image_bytes = image_source.getvalue()
            report_key = hashlib.sha256(
                image_bytes + selected_crop_global.encode("utf-8")
                + str(rotate_angle).encode() + str(flip_mode).encode()
            ).hexdigest()
            if st.button("Mulai analisis", type="primary", use_container_width=True):
                with st.spinner("Memproses gambar dan menganalisis indikasi penyakit tanaman..."):
                    sample = image.copy()
                    sample.thumbnail((800, 800))
                    output = io.BytesIO()
                    sample.convert("RGB").save(output, format="JPEG", quality=80)
                    result = analyze_plant_disease_with_gemini(output.getvalue(), selected_crop_global)
                    st.session_state["report_key"] = report_key
                    st.session_state["report_content"] = result

            if st.session_state.get("report_key") == report_key:
                report = st.session_state["report_content"]
                if "BUKAN_TANAMAN" in report:
                    st.error("Gambar tidak dikenali sebagai sampel tanaman. Unggah foto yang lebih jelas.")
                elif "API key" in report.lower() or "Gagal menghubungkan" in report:
                    st.warning(report)
                else:
                    st.success("Analisis selesai")
                    st.markdown(report)
                    st.download_button("Unduh laporan (TXT)", report, "tanicerdas_diagnosis.txt", mime="text/plain")
            else:
                st.info("Sampel siap dianalisis. Tekan tombol untuk menampilkan hasil.")
        else:
            st.markdown("""<div class="custom-card" style="padding:27px 23px;min-height:205px;display:flex;flex-direction:column;justify-content:center">
              <div style="font-size:10px;color:#7FCFA7;letter-spacing:.14em;font-weight:800;margin-bottom:15px">ANALYSIS CONSOLE</div>
              <h4 style="margin:0 0 10px">Menunggu sampel tanaman</h4>
              <p style="font-size:12px;line-height:1.8;color:#9DB9A7">Pilih kamera atau unggah foto daun dan buah. Laporan akan ditampilkan setelah analisis dijalankan.</p>
            </div>""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 3: KALKULATOR IRIGASI
# ------------------------------------------------------------------------------
with tab3:
    section_header("03", "Perencanaan irigasi", "Simulasikan kebutuhan air berdasarkan parameter cuaca, fase tumbuh, dan luas lahan.", "WATER MANAGEMENT")
    st.write(f"Estimasi kebutuhan air harian dan volume penyiraman untuk **{selected_crop_global}** berdasarkan input simulasi.")

    nasa_data = get_real_weather_nasa(current_lat, current_lon)
    if nasa_data:
        def_temp, def_rh, def_wind, def_rain = nasa_data
    else:
        def_temp, def_rh, def_wind, def_rain = 29.5, 65, 2.1, 1.2

    col_env, col_crop_stage = st.columns([1, 1])

    with col_env:
        st.markdown(f"#### Parameter iklim dan cuaca {'(NASA POWER, historis)' if nasa_data else '(nilai awal simulasi)'}")
        temp = st.slider("Suhu Udara Rata-rata (°C):", min_value=15.0, max_value=42.0, value=float(def_temp), step=0.5)
        humidity = st.slider("Kelembapan Relatif / RH (%):", min_value=20, max_value=95, value=int(def_rh), step=1)
        wind_speed = st.slider("Kecepatan Angin (m/s pada ketinggian 2m):", min_value=0.5, max_value=10.0, value=float(def_wind), step=0.1)
        sunshine_hrs = st.slider("Lama Penyinaran Matahari (Jam/Hari):", min_value=2.0, max_value=12.0, value=8.5, step=0.5)
        rainfall_eff = st.number_input("Curah Hujan Efektif (mm/hari):", min_value=0.0, max_value=50.0, value=float(def_rain), step=0.5)

    with col_crop_stage:
        st.markdown(f"#### Parameter Tanaman & Sistem Irigasi")

        st.success(f"Komoditas Aktif dari Sidebar: **{selected_crop_global}**")

        stage = st.radio("Fase Pertumbuhan Tanaman:", ["Fase Awal", "Fase Vegetatif", "Fase Generatif"], horizontal=False)

        luas_m2 = st.number_input("Luas Lahan (m²):", min_value=10.0, max_value=100000.0, value=1000.0, step=100.0)
        eff_irrigation = st.slider("Efisiensi Sistem Irigasi (%):", min_value=50, max_value=95, value=85)

    c_info = CROPS_DATABASE[selected_crop_global]
    if "Awal" in stage:
        kc = c_info.get("kc_init", 0.50)
    elif "Vegetatif" in stage:
        kc = c_info.get("kc_mid", 1.10)
    else:
        kc = c_info.get("kc_end", 1.00)

    humidity_factor = float(np.clip(1.0 + (60 - humidity) * 0.004, 0.7, 1.3))
    eto = round(0.0023 * (temp + 17.8) * (12.0**0.5) * (0.408 * (sunshine_hrs * 1.35 + 8.0)) * (1 + 0.1 * (wind_speed - 2.0)) * humidity_factor, 2)
    eto = max(0.0, eto)

    etc_gross = round(eto * kc, 2)
    etc_net = max(0.0, round(etc_gross - rainfall_eff, 2))

    vol_liter_m2 = round(etc_net / (eff_irrigation / 100.0), 2)
    vol_total_liter = round(vol_liter_m2 * luas_m2, 1)
    vol_total_m3 = round(vol_total_liter / 1000.0, 2)

    st.markdown("---")
    st.markdown(f"### Hasil simulasi irigasi untuk **{selected_crop_global}**")
    st.caption("Perhitungan menggunakan pendekatan empiris sederhana dan tidak menggantikan perhitungan FAO-56 penuh atau rekomendasi agronom.")

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
            <h4 style="color:#0284c7; margin-top:0;"> Jadwal & Rekomendasi Penyiraman untuk {selected_crop_global}:</h4>
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
            <h5 style="color:#10b981; margin-top:0;"> Catatan Karakteristik {selected_crop_global}:</h5>
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
    section_header("04", "Estimasi hasil panen", "Bandingkan keluaran model dan eksplorasi skenario produktivitas.", "YIELD INTELLIGENCE")
    st.write(f"Simulasi berbobot dari dua input prediksi model untuk memperkirakan bobot panen komoditas **{selected_crop_global}**.")

    col_p1, col_p2 = st.columns([1, 1.2])

    with col_p1:
        st.markdown("#### Parameter simulasi")
        alpha_val = st.slider("Bobot Pembobotan Ensemble Model (α):", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        rf_pred = st.number_input("Prediksi Yield Random Forest (Ton/Ha):", min_value=1.0, max_value=100.0, value=35.0, step=0.5)
        xgb_pred = st.number_input("Prediksi Yield XGBoost (Ton/Ha):", min_value=1.0, max_value=100.0, value=38.5, step=0.5)

        y_ensemble = (alpha_val * rf_pred) + ((1 - alpha_val) * xgb_pred)

    with col_p2:
        st.markdown("#### Hasil kombinasi prediksi")
        y1, y2 = st.columns(2)
        with y1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Estimasi Hasil Ensemble (ŷ)</div><div class="metric-value" style="color:#10b981;">{y_ensemble:.2f} <span style="font-size:0.9rem;">Ton/Ha</span></div></div>', unsafe_allow_html=True)
        with y2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Input XGBoost</div><div class="metric-value" style="color:#38bdf8;">{xgb_pred:.2f} <span style="font-size:0.9rem;">Ton/Ha</span></div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Analisis visual")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.markdown("##### 1. Perbandingan Output Antar Model (Ton/Ha)")

        fig_bar = go.Figure(data=[
            go.Bar(
                x=['Random Forest', 'XGBoost', 'Ensemble AI'],
                y=[rf_pred, xgb_pred, y_ensemble],
                marker_color=['#22D3EE', '#FBBF24', '#34D399'],
                text=[f"{rf_pred:.2f}", f"{xgb_pred:.2f}", f"{y_ensemble:.2f}"],
                textposition='auto'
            )
        ])
        fig_bar.update_layout(
            template="plotly_dark",
            font_color="#C5E3D1",
            font_family="DM Sans",
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
            line=dict(color='#34D399', width=3),
            marker=dict(size=10),
            text=[f"{v:.1f} Ton" for v in yield_trend],
            textposition="top center"
        ))
        fig_line.update_layout(
            template="plotly_dark",
            font_color="#C5E3D1",
            font_family="DM Sans",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Estimasi (Ton/Ha)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("##### 3. Ilustrasi bobot faktor agronomi (bukan feature importance model terlatih)")

    features = ['NDVI (Klorofil)', 'pH Tanah', 'Curah Hujan', 'Suhu Udara', 'Kelembapan', 'NDWI (Air)']
    importance = [35, 22, 18, 12, 8, 5]

    fig_feat = px.bar(
        x=importance,
        y=features,
        orientation='h',
        color=importance,
        color_continuous_scale=['#145544','#19806C','#34D399'],
        labels={'x': 'Kontribusi (%)', 'y': 'Fitur Input'}
    )
    fig_feat.update_layout(
        template="plotly_dark",
            font_color="#C5E3D1",
            font_family="DM Sans",
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
    section_header("05", "Basis data komoditas", "Telusuri parameter tumbuh dan kategori tanaman pertanian Indonesia.", "CROP LIBRARY")
    st.write("Daftar komoditas nasional beserta parameter tumbuh idealnya:")

    df_crops = pd.DataFrame.from_dict(CROPS_DATABASE, orient='index')
    df_crops.index.name = "Nama Komoditas / Tanaman"
    df_crops.reset_index(inplace=True)

    kat_filter = st.multiselect("Filter Kategori Dashboard:", options=list(df_crops["kategori"].unique()), default=list(df_crops["kategori"].unique()))
    query = st.text_input("Cari nama komoditas", placeholder="Contoh: jagung, tomat, pisang")
    df_filtered = df_crops[df_crops["kategori"].isin(kat_filter)]
    if query.strip():
        df_filtered = df_filtered[df_filtered["Nama Komoditas / Tanaman"].str.contains(query.strip(), case=False, regex=False)]
    st.caption(f"Menampilkan {len(df_filtered)} dari {len(df_crops)} komoditas")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)
    st.download_button("Unduh hasil filter (CSV)", df_filtered.to_csv(index=False).encode("utf-8-sig"), "tanicerdas_komoditas.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### Distribusi rentang pH menurut kategori")
    if not df_filtered.empty:
        fig_cat = px.box(df_filtered, x="kategori", y="ph_min", color="kategori", points="all", labels={"ph_min": "pH Minimum Tanah", "kategori": "Kategori Komoditas"})
        fig_cat.update_layout(template="plotly_dark",
            font_color="#C5E3D1",
            font_family="DM Sans", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend_title_text="")
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("Tidak ada komoditas yang cocok dengan filter.")

# ------------------------------------------------------------------------------
# TAB 5: OPEN DATASETS
# ------------------------------------------------------------------------------
with tab5:
    section_header("06", "Referensi dataset", "Sumber data terbuka untuk eksplorasi dan riset visi komputer tanaman.", "RESEARCH LIBRARY")
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

st.markdown("""
<div class="page-footer">
 <span>TaniCerdas / Agriculture Intelligence Platform</span>
 <span>Research workspace &nbsp; · &nbsp; Indonesia &nbsp; · &nbsp; Version 4.0</span>
</div>
""", unsafe_allow_html=True)
