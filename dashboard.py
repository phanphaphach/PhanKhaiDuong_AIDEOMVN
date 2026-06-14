import streamlit as st

# CẤU HÌNH TRANG (Bắt buộc phải khai báo đầu tiên)
st.set_page_config(page_title="AIDEOM-VN", page_icon="🇻🇳", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      .stMetric {background: rgba(180,180,180,0.06); border-radius: 12px;
                 padding: 12px 16px;}
      div[data-testid="stSidebar"] {min-width: 270px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Nạp dữ liệu và cấu hình tiện ích từ utils/
from src.data_loader import load_macro, load_sectors, load_regions

# Nạp các hàm render giao diện từ modules/
from modules import (home, lesson_01, lesson_02, lesson_03, lesson_04, 
                     lesson_05, lesson_06, lesson_07, lesson_08, 
                     lesson_09, lesson_10, lesson_11, lesson_12)

# Khởi tạo dữ liệu hệ thống tổng thể
macro = load_macro()
sectors = load_sectors()
regions = load_regions()

# Cài đặt Sidebar định tuyến
st.sidebar.title("🇻🇳 AIDEOM-VN")
st.sidebar.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")

PAGES = [
    "🏠 Trang chủ Tổng quan Hệ thống",
    "🔹 Bài 1 — Cobb-Douglas + AI",
    "🔹 Bài 2 — LP ngân sách số",
    "🔹 Bài 3 — Priority 10 ngành",
    "🔹 Bài 4 — LP ngành-vùng",
    "🔹 Bài 5 — MIP 15 dự án",
    "🔹 Bài 6 — TOPSIS 6 vùng",
    "🔹 Bài 7 — NSGA-II Pareto",
    "🔹 Bài 8 — Tối ưu động 2026-2035",
    "🔹 Bài 9 — Lao động & AI",
    "🔹 Bài 10 — Stochastic 2 giai đoạn",
    "🔹 Bài 11 — Q-learning RL",
    "🚀 Bài 12 — Đồ án AIDEOM tích hợp",
]

choice = st.sidebar.radio("Điều hướng", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Sinh viên thực hiện** \n"
    "Phan Khải Dương  \n"
    "MSV: 23051209 \n"
    "Viện Quản trị Kinh doanh  \n"
    "Đại học Kinh tế, ĐHQGHN - 2025-2026"
)
st.sidebar.caption("Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")

# Điều hướng chính xác luồng chạy ứng dụng
if choice == PAGES[0]:
    home.render(macro, sectors, regions)
elif choice == PAGES[1]:
    lesson_01.render(macro)
elif choice == PAGES[2]:
    lesson_02.render()
elif choice == PAGES[3]:
    lesson_03.render(sectors)
elif choice == PAGES[4]:
    lesson_04.render()
elif choice == PAGES[5]:
    lesson_05.render()
elif choice == PAGES[6]:
    lesson_06.render(regions)
elif choice == PAGES[7]:
    lesson_07.render()
elif choice == PAGES[8]:
    lesson_08.render()
elif choice == PAGES[9]:
    lesson_09.render()
elif choice == PAGES[10]:
    lesson_10.render()
elif choice == PAGES[11]:
    lesson_11.render()
elif choice == PAGES[12]:
    lesson_12.render(macro, regions)