import streamlit as st
import pandas as pd

def render(macro, sectors, regions):
    st.title("🇻🇳 AIDEOM-VN")
    st.subheader("AI-Driven Decision Optimization Model for Vietnam")
    st.markdown(
        "Web app giải **12 bài toán mô hình ra quyết định** phát triển kinh tế "
        "Việt Nam trong kỉ nguyên AI — dữ liệu thực 2020–2025. Hệ thống AIDEOM-VN "
        "chuyển bài toán chính sách kinh tế thành mô hình toán có ràng buộc và giải "
        "bằng các kỹ thuật tối ưu LP, MIP, đa mục tiêu, động, ngẫu nhiên và học tăng cường."
    )

    # --- Số liệu vĩ mô tham chiếu nhanh 2025 ---
    last = macro.iloc[-1]
    prev = macro.iloc[-2]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP 2025", f"{last['GDP_billion_USD']:.1f} tỷ USD",
              f"{last['GDP_growth_pct']:.2f}%")
    c2.metric("Kinh tế số / GDP", f"≈{last['digital_economy_share_GDP_pct']:.1f}%",
              f"+{last['digital_economy_share_GDP_pct']-prev['digital_economy_share_GDP_pct']:.1f} đpt")
    c3.metric("FDI giải ngân 2025", f"{last['FDI_disbursed_billion_USD']:.1f} tỷ USD",
              f"+{(last['FDI_disbursed_billion_USD']/prev['FDI_disbursed_billion_USD']-1)*100:.1f}%")
    c4.metric("GDP/người 2025", f"{int(last['GDP_per_capita_USD']):,} USD",
              f"+{(last['GDP_per_capita_USD']/prev['GDP_per_capita_USD']-1)*100:.1f}%")

    st.markdown("---")
    st.subheader("📦 Xác nhận nạp dữ liệu (3 tệp CSV gốc)")
    st.caption("Kiểm tra Shape / số dòng / số cột & xem trước 5 dòng đầu của mỗi tệp.")

    for name, df in [
        ("vietnam_macro_2020_2025.csv", macro),
        ("vietnam_sectors_2024.csv", sectors),
        ("vietnam_regions_2024.csv", regions),
    ]:
        st.markdown(f"**`{name}`** — Shape `{df.shape}` "
                    f"({df.shape[0]} dòng × {df.shape[1]} cột)")
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")
    st.subheader("🗺️ 12 bài toán theo 4 cấp độ")
    levels = pd.DataFrame({
        "Cấp độ": ["DỄ", "DỄ", "DỄ", "TRUNG BÌNH", "TRUNG BÌNH", "TRUNG BÌNH",
                    "KHÁ KHÓ", "KHÁ KHÓ", "KHÁ KHÓ", "KHÓ", "KHÓ", "KHÓ"],
        "Bài": [f"Bài {i}" for i in range(1, 13)],
        "Nội dung": [
            "Hàm sản xuất Cobb-Douglas mở rộng + AI — growth accounting, dự báo GDP 2030",
            "LP phân bổ ngân sách 4 hạng mục — scipy/pulp, shadow price",
            "Chỉ số ưu tiên 10 ngành — min-max norm, weighted scoring, sensitivity",
            "LP phân bổ ngân sách số ngành-vùng (PuLP + CVXPY), công bằng vùng",
            "MIP lựa chọn 15 dự án chuyển đổi số — knapsack + ràng buộc logic",
            "TOPSIS xếp hạng 6 vùng — entropy weight, AHP",
            "NSGA-II tối ưu đa mục tiêu — Pareto 4 mục tiêu xung đột",
            "Tối ưu động liên thời gian 2026-2035 — CVXPY, welfare chiết khấu",
            "Tác động AI tới lao động — NetJob ròng theo ngành",
            "Quy hoạch ngẫu nhiên 2 giai đoạn — Pyomo, VSS & EVPI",
            "Q-learning chính sách kinh tế thích nghi — MDP 81 trạng thái",
            "Đồ án tích hợp AIDEOM-VN — dashboard 5 kịch bản chính sách",
        ],
    })
    st.dataframe(levels, use_container_width=True, hide_index=True)