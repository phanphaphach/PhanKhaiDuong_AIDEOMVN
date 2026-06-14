import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go

from src.components import header, policy_box

def render():
    header("🔹 Bài 9 — Tác động AI tới thị trường lao động Việt Nam",
           "Phân bổ 30 nghìn tỷ cho 8 ngành (x_AI, x_H) tối đa hóa tổng NetJob ròng, "
           "bảo đảm không ngành nào mất việc làm ròng.")

    N = 8
    sector_vi = ["Nông-Lâm-Thủy sản", "CN chế biến chế tạo", "Xây dựng", "Bán buôn-bán lẻ",
                 "Tài chính-Ngân hàng", "Logistics-Vận tải", "CNTT-Truyền thông",
                 "Giáo dục-Đào tạo"]
    L = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
    risk = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
    a1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
    b1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
    c1c = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
    d1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])
    BUDGET = 30000
    disp_rate = c1c * risk
    net_ai = a1 - disp_rate
    PER_CAP = BUDGET / N * 1.5

    def solve_model(per_cap=None, cap5=False):
        xA = cp.Variable(N, nonneg=True)
        xH = cp.Variable(N, nonneg=True)
        Displaced = cp.multiply(disp_rate, xA)
        RetrainCap = cp.multiply(d1, xH)
        NetJob = cp.multiply(a1, xA) + cp.multiply(b1, xH) - Displaced
        cons = [cp.sum(xA + xH) <= BUDGET, NetJob >= 0, Displaced <= RetrainCap]
        if per_cap is not None:
            cons += [xA <= per_cap, xH <= per_cap]
        if cap5:
            cons += [Displaced <= 0.05 * (L * 1000)]
        prob = cp.Problem(cp.Maximize(cp.sum(NetJob)), cons)
        prob.solve()
        return dict(status=prob.status, Z=prob.value, xA=xA.value, xH=xH.value,
                    net=NetJob.value, disp=Displaced.value)

    # --- Câu 9.4.1 ---
    st.subheader("Câu 9.4.1 — Phân bổ tối ưu (x_AI, x_H) & NetJob ròng")
    sol = solve_model(per_cap=PER_CAP)
    st.metric("Tổng NetJob Z*", f"{sol['Z']:,.0f} job-units")
    rows = []
    for i in range(N):
        rows.append({"Ngành": sector_vi[i], "x_AI": round(sol["xA"][i]),
                     "x_H": round(sol["xH"][i]),
                     "NewJob": round(a1[i] * sol["xA"][i]),
                     "Upgrade": round(b1[i] * sol["xH"][i]),
                     "Displaced": round(disp_rate[i] * sol["xA"][i]),
                     "NetJob": round(sol["net"][i])})
    rdf = pd.DataFrame(rows)
    fig = px.bar(rdf, x="Ngành", y="NetJob", color="NetJob",
                 color_continuous_scale="Greens", title="NetJob ròng theo ngành")
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rdf, use_container_width=True, hide_index=True)

    # --- Câu 9.4.2 ---
    st.subheader("Câu 9.4.2 — Ngưỡng đào tạo tối thiểu x_H ngành 2 (CN chế biến chế tạo)")
    i = 1
    st.latex(rf"NetJob_2 = ({a1[i]} - {c1c[i]}\times{risk[i]:.2f})\cdot x_{{AI}} + "
             rf"{b1[i]}\cdot x_H = ({net_ai[i]:+.3f})\cdot x_{{AI}} + {b1[i]}\cdot x_H")
    xAI_max = PER_CAP
    if net_ai[i] >= 0:
        ratio = disp_rate[i] / d1[i]
        st.success(f"Hệ số ròng AI = {net_ai[i]:+.3f} > 0 ⇒ NetJob₂ ≥ 0 với mọi x_AI "
                   f"(kể cả x_H=0). Nhưng ràng buộc năng lực đào tạo lại buộc "
                   f"x_H ≥ {ratio:.4f}·x_AI; khi x_AI tối đa = {xAI_max:,.0f} thì "
                   f"x_H tối thiểu = {ratio*xAI_max:,.1f}.")
        xH_grid = np.linspace(0, PER_CAP, 60)
        netjob2 = net_ai[i] * xAI_max + b1[i] * xH_grid
    else:
        xH_th = -net_ai[i] * xAI_max / b1[i]
        st.warning(f"Hệ số ròng AI = {net_ai[i]:.3f} < 0 ⇒ cần x_H ≥ {xH_th:,.1f}.")
        xH_grid = np.linspace(0, PER_CAP, 60)
        netjob2 = net_ai[i] * xAI_max + b1[i] * xH_grid
    fig = px.area(x=xH_grid, y=netjob2, labels={"x": "x_H ngành 2", "y": "NetJob₂"},
                  title=f"NetJob₂ theo x_H (x_AI = {xAI_max:,.0f})")
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 9.4.3: Sankey ---
    st.subheader("Câu 9.4.3 — Luồng lao động nhóm dễ tổn thương (ngành 1, 3, 4)")
    vuln = [0, 2, 3]
    src_labels = [sector_vi[i] for i in vuln]
    dst_labels = ["Việc làm mới (AI)", "Nâng cấp kỹ năng", "Bị thay thế (AI)"]
    labels = src_labels + dst_labels
    ns = len(vuln)
    s_idx, t_idx, vals, lcolors = [], [], [], []
    palette = {"new": "rgba(41,128,185,0.55)", "up": "rgba(22,160,133,0.6)",
               "disp": "rgba(192,57,43,0.6)"}
    for k, i in enumerate(vuln):
        for off, val, c in [(0, a1[i] * sol["xA"][i], palette["new"]),
                            (1, b1[i] * sol["xH"][i], palette["up"]),
                            (2, disp_rate[i] * sol["xA"][i], palette["disp"])]:
            if val > 1e-9:
                s_idx.append(k); t_idx.append(ns + off); vals.append(float(val)); lcolors.append(c)
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=24, thickness=20,
                  color=["#2980b9"] * ns + ["#3498db", "#16a085", "#c0392b"]),
        link=dict(source=s_idx, target=t_idx, value=vals, color=lcolors)))
    fig.update_layout(title_text="Luồng dịch chuyển lao động (job-units)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 9.4.4 ---
    st.subheader("Câu 9.4.4 — Ràng buộc 'không ngành nào mất quá 5% lao động'")
    sol5 = solve_model(per_cap=PER_CAP, cap5=True)
    if sol5["status"].startswith("optimal"):
        c1, c2 = st.columns(2)
        c1.metric("Z* có C4 (5%)", f"{sol5['Z']:,.0f}", f"{sol5['Z']-sol['Z']:,.0f}")
        c2.metric("Chi phí bảo vệ XH", f"{sol['Z']-sol5['Z']:,.0f} job-units")
        cmp = pd.DataFrame({"Ngành": sector_vi, "Không C4": sol["net"].round(0),
                            "Có C4 (5%)": sol5["net"].round(0)})
        fig = go.Figure()
        fig.add_bar(x=sector_vi, y=sol["net"], name="Không C4", marker_color="#3498db")
        fig.add_bar(x=sector_vi, y=sol5["net"], name="Có C4 (5%)", marker_color="#e67e22")
        fig.update_layout(barmode="group", xaxis_tickangle=-25,
                          title="NetJob theo ngành: tác động ràng buộc bảo vệ 5%")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)
    else:
        st.error("Bài toán bất khả thi với ràng buộc 5%.")

    policy_box(9)