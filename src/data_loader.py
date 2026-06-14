import pandas as pd
import streamlit as st
from pathlib import Path

# Đảm bảo bạn đã import các hằng số này từ file constants.py cùng thư mục
from .constants import SECTOR_VI, REGION_VI, ALPHA, BETA, GAMMA, DELTA, THETA

# Tự động nhận diện thư mục gốc và thư mục data
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def load_macro():
    file_path = DATA_DIR / "vietnam_macro_2020_2025.csv"
    df = pd.read_csv(file_path)
    return df.sort_values("year").reset_index(drop=True)

@st.cache_data
def load_sectors():
    file_path = DATA_DIR / "vietnam_sectors_2024.csv"
    df = pd.read_csv(file_path)
    df["sector_name_vi"] = df["sector_name_en"].map(SECTOR_VI)
    return df

@st.cache_data
def load_regions():
    file_path = DATA_DIR / "vietnam_regions_2024.csv"
    df = pd.read_csv(file_path)
    df["region_name_vi"] = df["region_name_en"].map(REGION_VI)
    return df

# ĐÂY LÀ HÀM BỊ THIẾU GÂY RA LỖI (Đã được bổ sung)
def cobb_core(K, L, D, AI, H):
    return (K ** ALPHA) * (L ** BETA) * (D ** GAMMA) * (AI ** DELTA) * (H ** THETA)