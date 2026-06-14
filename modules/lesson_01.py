import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.constants import K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES, ALPHA, BETA, GAMMA, DELTA, THETA, SEQ
from src.data_loader import cobb_core
from src.components import header, policy_box

def render(macro):
    header("🔹 Bài 1 — Hàm sản xuất Cobb-Douglas mở rộng + AI",
           "Ước lượng TFP, phân rã tăng trưởng (growth accounting) và dự báo GDP 2030 "
           "cho nền kinh tế Việt Nam với các yếu tố mới: số hóa D, năng lực AI, nhân lực số H.")

    year = macro["year"].values
    K, L, D, AI, H = K_SERIES, L_SERIES, D_SERIES, AI_SERIES, H_SERIES
    Y = macro["GDP_trillion_VND"].values.astype(float)

    # --- Câu 1.4.1: TFP A_t (giải ngược) ---
    core = cobb_core(K, L, D, AI, H)
    A = Y / core
    tfp_cagr = (A[-1] / A[0]) ** (1 / (len(A) - 1)) - 1

    st.subheader("Câu 1.4.1 — Ước lượng năng suất nhân tố tổng hợp (TFP) $A_t$")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.line(x=year, y=A, markers=True,
                      labels={"x": "Năm", "y": "TFP $A_t$"},
                      title="TFP $A_t$ giải ngược từ hàm sản xuất, 2020–2025")
        fig.update_traces(line_color="#c0392b")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("TFP CAGR 2020-2025", f"{tfp_cagr*100:.2f}%/năm")
        st.metric("A trung bình", f"{A.mean():.4f}")
        st.caption("TFP đi lên ⇒ tăng trưởng dựa nhiều hơn vào hiệu quả/công nghệ "
                   "(tăng trưởng theo chiều sâu).")
    t141 = pd.DataFrame({"Năm": year, "Y (GDP)": Y.round(1),
                         "Core CD": core.round(2), "TFP A_t": A.round(4)})
    st.dataframe(t141, use_container_width=True, hide_index=True)

    # --- Câu 1.4.2: Dự báo & MAPE ---
    st.subheader("Câu 1.4.2 — Dự báo $\\hat{Y}_t$ với A = trung bình & MAPE")
    A_bar = A.mean()
    Y_hat = A_bar * core
    ape = np.abs((Y - Y_hat) / Y) * 100
    mape = ape.mean()
    st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")
    dfp = pd.DataFrame({"Năm": year, "Y thực tế": Y, "Ŷ dự báo": Y_hat.round(1),
                        "Sai số %": ape.round(2)})
    fig = go.Figure()
    fig.add_scatter(x=year, y=Y, mode="lines+markers", name="Y thực tế",
                    line_color="#2c3e50")
    fig.add_scatter(x=year, y=Y_hat, mode="lines+markers", name="Ŷ dự báo (A=const)",
                    line=dict(color="#e67e22", dash="dash"))
    fig.update_layout(title=f"Y thực tế vs Ŷ dự báo (MAPE = {mape:.2f}%)",
                      xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dfp, use_container_width=True, hide_index=True)

    # --- Câu 1.4.3: Phân rã tăng trưởng ---
    st.subheader("Câu 1.4.3 — Phân rã tăng trưởng GDP bình quân năm (growth accounting)")
    n = len(year) - 1
    avg_dln = lambda x: (np.log(x[-1]) - np.log(x[0])) / n
    dlnY = avg_dln(Y)
    contrib = {
        "K (vốn vật chất)": ALPHA * avg_dln(K),
        "L (lao động)": BETA * avg_dln(L),
        "D (số hóa)": GAMMA * avg_dln(D),
        "AI (DN số)": DELTA * avg_dln(AI),
        "H (nhân lực số)": THETA * avg_dln(H),
    }
    contrib["TFP (phần dư)"] = dlnY - sum(contrib.values())
    tot = sum(contrib.values())
    t143 = pd.DataFrame({
        "Yếu tố": list(contrib.keys()),
        "Đóng góp (điểm %/năm)": [v * 100 for v in contrib.values()],
        "Tỷ trọng (%)": [v / tot * 100 for v in contrib.values()],
    })
    fig = px.bar(t143, x="Yếu tố", y="Đóng góp (điểm %/năm)",
                 color="Yếu tố", color_discrete_sequence=SEQ,
                 title="Đóng góp từng yếu tố vào tăng trưởng GDP (điểm %/năm)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t143.round(3), use_container_width=True, hide_index=True)
    new_factors = {k: v for k, v in contrib.items()
                   if k in ["D (số hóa)", "AI (DN số)", "H (nhân lực số)"]}
    st.info(f"Trong các yếu tố MỚI, đóng góp lớn nhất: **{max(new_factors, key=new_factors.get)}** "
            f"— tăng trưởng GDP log bình quân ≈ {dlnY*100:.2f}%/năm.")

    # --- Câu 1.4.4: Dự báo 2030 (tương tác) ---
    st.subheader("Câu 1.4.4 — Mô phỏng & dự báo GDP 2030")
    cc = st.columns(3)
    D_2030 = cc[0].slider("D (KTS/GDP %) 2030", 19.5, 40.0, 30.0, 0.5)
    AI_2030 = cc[1].slider("AI (nghìn DN số) 2030", 80.0, 150.0, 100.0, 1.0)
    H_2030 = cc[2].slider("H (LĐ qua ĐT %) 2030", 29.0, 50.0, 35.0, 0.5)
    cc2 = st.columns(3)
    g_KL = cc2[0].slider("Tăng trưởng K, L (%/năm)", 3.0, 10.0, 6.0, 0.5) / 100
    g_TFP = cc2[1].slider("Tăng trưởng TFP (%/năm)", 0.0, 3.0, 1.2, 0.1) / 100
    n_fwd = 2030 - 2025
    K30 = K[-1] * (1 + g_KL) ** n_fwd
    L30 = L[-1] * (1 + g_KL) ** n_fwd
    A30 = A[-1] * (1 + g_TFP) ** n_fwd
    Y30 = A30 * cobb_core(K30, L30, D_2030, AI_2030, H_2030)
    growth30 = (Y30 / Y[-1]) ** (1 / n_fwd) - 1
    m1, m2 = st.columns(2)
    m1.metric("GDP dự báo 2030", f"{Y30:,.0f} nghìn tỷ VND")
    m2.metric("Tăng trưởng BQ 2025→2030", f"{growth30*100:.2f}%/năm")
    fig = go.Figure()
    fig.add_scatter(x=year, y=Y, mode="lines+markers", name="GDP thực tế 2020-2025",
                    line_color="#2c3e50")
    fig.add_scatter(x=[2025, 2030], y=[Y[-1], Y30], mode="lines+markers",
                    name="Dự báo tới 2030", line=dict(color="#27ae60", dash="dash"))
    fig.update_layout(title="GDP thực tế 2020-2025 và dự báo 2030",
                      xaxis_title="Năm", yaxis_title="GDP (nghìn tỷ VND)")
    st.plotly_chart(fig, use_container_width=True)

    policy_box(1)