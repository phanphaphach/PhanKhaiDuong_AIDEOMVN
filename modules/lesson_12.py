import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.constants import K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES, ALPHA, BETA, GAMMA, DELTA, THETA, SEQ
from src.data_loader import cobb_core
from src.components import policy_box

def render(macro, regions):
    st.title("🚀 Bài 12 — Đồ án tích hợp mô hình tổng thể AIDEOM-VN")
    st.markdown(
        "> **Mục tiêu:** tích hợp các kỹ thuật Bài 1–11 thành hệ thống 6 module (M1–M6) "
        "với dashboard thử nghiệm 5 kịch bản chính sách kinh tế (S1–S5)."
    )
    st.markdown("---")

    SCENARIOS = {
        "S1. Truyền thống": np.array([0.70, 0.10, 0.10, 0.10]),
        "S2. Số hóa nhanh": np.array([0.25, 0.45, 0.15, 0.15]),
        "S3. AI dẫn dắt": np.array([0.20, 0.20, 0.45, 0.15]),
        "S4. Bao trùm số": np.array([0.30, 0.20, 0.10, 0.40]),
        "S5. Tối ưu cân bằng": np.array([0.40, 0.25, 0.20, 0.15]),
    }
    SCN_DESC = {
        "S1. Truyền thống": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống, xuất khẩu.",
        "S2. Số hóa nhanh": "Tăng đầu tư chính phủ số, doanh nghiệp số, thanh toán số.",
        "S3. AI dẫn dắt": "Ưu tiên AI, dữ liệu lớn, bán dẫn, trung tâm dữ liệu.",
        "S4. Bao trùm số": "Ưu tiên vùng yếu, SME, giáo dục số, nông nghiệp số.",
        "S5. Tối ưu cân bằng": "Cân bằng 4 trụ cột theo trọng số chính sách AIDEOM-VN.",
    }

    scn = st.selectbox("🎛️ Chọn kịch bản chính sách để giả lập", list(SCENARIOS.keys()), index=4)
    alloc = SCENARIOS[scn]
    st.caption(SCN_DESC[scn])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Tổng quan (M1-M2)", "💰 Phân bổ (M3)", "🎯 So sánh 5 kịch bản (M6)", "⚠️ Cảnh báo rủi ro (M4-M5)"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.subheader("M1 — Dự báo kinh tế (Cobb-Douglas, kế thừa Bài 1)")
        year = macro["year"].values
        Y = macro["GDP_trillion_VND"].values.astype(float)
        core = cobb_core(K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES)
        A = Y / core
        A_bar = A.mean()
        Y_hat = A_bar * core
        mape = (np.abs((Y - Y_hat) / Y) * 100).mean()
        
        n_fwd = 5
        K30 = K_SERIES[-1] * (1 + 0.04 + 0.05 * alloc[0]) ** n_fwd
        D30 = D_SERIES[-1] * (1 + 0.06 + 0.30 * alloc[1]) ** n_fwd
        AI30 = AI_SERIES[-1] * (1 + 0.05 + 0.25 * alloc[2]) ** n_fwd
        H30 = H_SERIES[-1] * (1 + 0.02 + 0.20 * alloc[3]) ** n_fwd
        A30 = A[-1] * 1.012 ** n_fwd
        Y30 = A30 * cobb_core(K30, L_SERIES[-1] * 1.04 ** n_fwd, D30, AI30, H30)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("MAPE (Cobb-Douglas)", f"{mape:.2f}%")
        c2.metric("Ā (TFP trung bình)", f"{A_bar:.4f}")
        c3.metric(f"Y 2030 dự báo ({scn[:2]})", f"{Y30:,.0f} ngh.tỷ")
        
        n = len(year) - 1
        avg_dln = lambda x: (np.log(x[-1]) - np.log(x[0])) / n
        contrib = {"TFP (A)": avg_dln(Y) - (ALPHA * avg_dln(K_SERIES) + BETA * avg_dln(L_SERIES) + GAMMA * avg_dln(D_SERIES) + DELTA * avg_dln(AI_SERIES) + THETA * avg_dln(H_SERIES)),
                   "Vốn (K)": ALPHA * avg_dln(K_SERIES), "Lao động (L)": BETA * avg_dln(L_SERIES), "Số hóa (D)": GAMMA * avg_dln(D_SERIES), "AI": DELTA * avg_dln(AI_SERIES), "Nhân lực số (H)": THETA * avg_dln(H_SERIES)}
        cdf = pd.DataFrame({"Yếu tố": list(contrib.keys()), "Đóng góp %/năm": [v * 100 for v in contrib.values()]})
        fig = px.bar(cdf, x="Yếu tố", y="Đóng góp %/năm", color="Yếu tố", color_discrete_sequence=SEQ, title="Phân rã đóng góp tăng trưởng 2020-2025")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("M2 — Đánh giá sẵn sàng số (TOPSIS, kế thừa Bài 6)")
        rdf = regions.copy()
        crit = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD", "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct", "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
        isb = np.array([True] * 7 + [False])
        w = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
        Xr = rdf[crit].values.astype(float)
        Rn = Xr / np.sqrt((Xr ** 2).sum(0))
        V = Rn * w
        As = np.where(isb, V.max(0), V.min(0))
        An = np.where(isb, V.min(0), V.max(0))
        Cstar = np.sqrt(((V - An) ** 2).sum(1)) / (np.sqrt(((V - As) ** 2).sum(1)) + np.sqrt(((V - An) ** 2).sum(1)))
        rdf["TOPSIS"] = Cstar
        fig = px.bar(rdf.sort_values("TOPSIS"), x="TOPSIS", y="region_name_vi", orientation="h", color="TOPSIS", color_continuous_scale="Tealgrn", title="Xếp hạng sẵn sàng số 6 vùng", labels={"region_name_vi": ""})
        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TAB 2 ----------------
    with tab2:
        st.subheader("M3 — Tối ưu phân bổ ngân sách")
        budget = st.number_input("Tổng ngân sách giả lập (nghìn tỷ VND)", 50, 500, 100, 10)
        labels = ["Vốn vật chất (K)", "Hạ tầng số (D)", "Công nghệ AI", "Nhân lực số (H)"]
        amounts = alloc * budget
        c1, c2 = st.columns([1, 1])
        with c1:
            fig = px.pie(names=labels, values=amounts, color_discrete_sequence=SEQ, title=f"Cơ cấu vốn — {scn}", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            adf = pd.DataFrame({"Trụ cột": labels, "Tỷ lệ %": (alloc * 100).round(1), "Ngân sách (ngh.tỷ)": amounts.round(1)})
            st.dataframe(adf, use_container_width=True, hide_index=True)
            st.metric("Tổng ngân sách phân bổ", f"{amounts.sum():,.0f} ngh.tỷ")

    # ---------------- TAB 3 ----------------
    with tab3:
        st.subheader("M6 — So sánh đa mục tiêu 5 kịch bản")
        def kpi_scn(alloc_arr):
            K, D, AI, H, U = 27500.0, 20.3, 86.0, 30.0, 0.38
            lastY = (K ** 0.33) * (54.0 ** 0.42) * (D ** 0.10) * (AI ** 0.08) * (H ** 0.07)
            tot_growth, tot_emis, tot_cyber = 0.0, 0.0, 0.0
            for _ in range(10):
                a = alloc_arr; b = 1000.0
                K += a[0] * b
                D = 0.88 * D + a[1] * b / 100.0
                AI = 0.85 * AI + a[2] * b / 20.0
                H += a[3] * b / 200.0
                Y = (K ** 0.33) * (54.0 ** 0.42) * (D ** 0.10) * (AI ** 0.08) * (H ** 0.07)
                tot_growth += (Y - lastY) / lastY
                U = float(np.clip(U + 0.06 * a[2] - 0.10 * a[3], 0.05, 0.95))
                tot_cyber += a[2] - 0.5 * a[3]
                tot_emis += a[0]
                lastY = Y
            return dict(growth=tot_growth, inclusion=alloc_arr[3] * 10 + (1 - U), green=10 - tot_emis, security=5 - tot_cyber, U_final=U)
            
        kpis = {name: kpi_scn(a) for name, a in SCENARIOS.items()}
        dims = ["growth", "inclusion", "green", "security"]
        dim_lbl = ["Tăng trưởng", "Bao trùm", "Xanh", "An ninh"]
        mat = np.array([[kpis[n][d] for d in dims] for n in SCENARIOS])
        matn = (mat - mat.min(0)) / (mat.max(0) - mat.min(0) + 1e-9)
        figr = go.Figure()
        for i, name in enumerate(SCENARIOS):
            highlight = (name == scn)
            figr.add_scatterpolar(r=matn[i].tolist() + [matn[i][0]], theta=dim_lbl + [dim_lbl[0]], fill="toself", name=name, opacity=1.0 if highlight else 0.3, line=dict(width=4 if highlight else 1.5))
        figr.update_layout(title="Radar đánh đổi đa mục tiêu (kịch bản chọn được tô đậm)", polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(figr, use_container_width=True)

        kdf = pd.DataFrame({"Kịch bản": list(SCENARIOS.keys()), "Σ tăng trưởng 10 năm": [kpis[n]["growth"] for n in SCENARIOS], "Thất nghiệp cuối kỳ": [kpis[n]["U_final"] for n in SCENARIOS]})
        figb = px.bar(kdf, x="Kịch bản", y="Σ tăng trưởng 10 năm", color="Kịch bản", color_discrete_sequence=SEQ, title="Tổng tăng trưởng tích lũy 10 năm")
        figb.update_layout(showlegend=False)
        st.plotly_chart(figb, use_container_width=True)
        st.dataframe(kdf.round(3), use_container_width=True, hide_index=True)

    # ---------------- TAB 4 ----------------
    with tab4:
        st.subheader("M4 — Mô phỏng việc làm ròng NetJob (kế thừa Bài 9)")
        sector_vi = ["Nông-Lâm-TS", "CN chế biến", "Xây dựng", "Bán buôn-lẻ", "Tài chính-NH", "Logistics", "CNTT-TT", "Giáo dục"]
        risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
        a1_arr = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
        b1_arr = np.array([45, 28, 35, 32, 22, 30, 20, 55])
        c1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
        budget_job = 30000
        x_AI_share = alloc[2] / (alloc[2] + alloc[3]) if (alloc[2] + alloc[3]) > 0 else 0.5
        x_AI = np.full(8, budget_job * x_AI_share / 8)
        x_H = np.full(8, budget_job * (1 - x_AI_share) / 8)
        netjob = a1_arr * x_AI + b1_arr * x_H - c1c * risk * x_AI
        ndf = pd.DataFrame({"Ngành": sector_vi, "NetJob": netjob})
        fig = px.bar(ndf, x="Ngành", y="NetJob", color="NetJob", color_continuous_scale="RdYlGn", title=f"NetJob ròng theo ngành — {scn}")
        fig.update_layout(xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
        neg = ndf[ndf["NetJob"] < 0]
        if len(neg): st.error(f"⚠️ Cảnh báo: {len(neg)} ngành có NetJob ÂM — {', '.join(neg['Ngành'])}. Cần tăng đầu tư đào tạo lại (H).")
        else: st.success("✅ Tất cả ngành đều có NetJob ròng dương dưới kịch bản này.")

        st.subheader("M5 — Đánh giá rủi ro (cyber / môi trường / phụ thuộc)")
        cyber = alloc[2] * 100
        emission = alloc[0] * 100
        depend = (1 - alloc[3]) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Rủi ro an ninh dữ liệu", f"{cyber:.0f}/100", "Cao" if cyber > 35 else "Vừa")
        c2.metric("Cường độ phát thải", f"{emission:.0f}/100", "Cao" if emission > 50 else "Thấp")
        c3.metric("Phụ thuộc (thiếu nhân lực)", f"{depend:.0f}/100", "Cao" if depend > 70 else "Vừa")

    policy_box(12)