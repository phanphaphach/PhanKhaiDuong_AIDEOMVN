import streamlit as st
import numpy as np
import pandas as pd
import cvxpy as cp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.constants import ALPHA, BETA, GAMMA, DELTA, THETA, SEQ
from src.components import header, policy_box

def render():
    header("🔹 Bài 8 — Tối ưu động phân bổ liên thời gian 2026-2035",
           "Tối đa hóa welfare chiết khấu Σ ρᵗ·ln(Cₜ) với hàm sản xuất Cobb-Douglas và "
           "động học tích lũy vốn K, D, AI, H.")

    T = 10
    years = np.arange(2026, 2026 + T)
    A_TFP, L_LAB = 1.0, 54.0
    EXPO = np.array([ALPHA, GAMMA, DELTA, THETA, BETA])
    dep_K, dep_D, dep_AI, dep_H = 0.05, 0.12, 0.15, 0.02
    eff_H = 0.8
    K0, D0, AI0, H0 = 27500.0, 20.3, 86.0, 30.0

    @st.cache_data
    def build_solve(rho=0.97, shock_idx=None, shock_pct=0.08, fixed_invest=None):
        K = cp.Variable(T + 1, nonneg=True); D = cp.Variable(T + 1, nonneg=True)
        AI = cp.Variable(T + 1, nonneg=True); H = cp.Variable(T + 1, nonneg=True)
        Y = cp.Variable(T, nonneg=True)
        IK = cp.Variable(T, nonneg=True); ID = cp.Variable(T, nonneg=True)
        IAI = cp.Variable(T, nonneg=True); IH = cp.Variable(T, nonneg=True)
        C = cp.Variable(T, nonneg=True)
        cons = [K[0] == K0, D[0] == D0, AI[0] == AI0, H[0] == H0]
        for t in range(T):
            stack = cp.vstack([K[t], D[t], AI[t], H[t], cp.Constant(L_LAB)])
            prod = A_TFP * cp.geo_mean(stack, EXPO)
            if shock_idx is not None and t == shock_idx:
                cons += [Y[t] <= (1 - shock_pct) * prod]
            else:
                cons += [Y[t] <= prod]
            cons += [C[t] + IK[t] + ID[t] + IAI[t] + IH[t] <= Y[t]]
            if fixed_invest is not None:
                cons += [IK[t] + ID[t] + IAI[t] + IH[t] == fixed_invest[t]]
            cons += [K[t + 1] == (1 - dep_K) * K[t] + IK[t],
                     D[t + 1] == (1 - dep_D) * D[t] + ID[t],
                     AI[t + 1] == (1 - dep_AI) * AI[t] + IAI[t],
                     H[t + 1] == (1 - dep_H) * H[t] + eff_H * IH[t]]
        obj = cp.Maximize(cp.sum([rho ** t * cp.log(C[t]) for t in range(T)]))
        prob = cp.Problem(obj, cons)
        prob.solve(solver=cp.CLARABEL)
        return dict(status=prob.status, welfare=prob.value, K=K.value, D=D.value,
                    AI=AI.value, H=H.value, Y=Y.value, C=C.value,
                    I_total=(IK.value + ID.value + IAI.value + IH.value))

    rho = st.slider("Hệ số chiết khấu ρ", 0.85, 0.99, 0.97, 0.01)
    base = build_solve(rho=rho)

    st.subheader("Câu 8.3.1 — Giải bài toán tối ưu động (CVXPY/CLARABEL)")
    st.metric("Welfare tối ưu W* = Σ ρᵗ·ln(Cₜ)", f"{base['welfare']:.4f}")
    traj = pd.DataFrame({"Năm": years, "Y": base["Y"].round(1), "C": base["C"].round(1),
                         "Đầu tư": base["I_total"].round(1), "K": base["K"][:T].round(0),
                         "D": base["D"][:T].round(1), "AI": base["AI"][:T].round(1),
                         "H": base["H"][:T].round(1)})
    st.dataframe(traj, use_container_width=True, hide_index=True)

    st.subheader("Câu 8.3.2 — Quỹ đạo tối ưu K, D, AI, H, Y, C")
    series = [("K (vốn vật chất)", base["K"][:T]), ("D (số hóa)", base["D"][:T]),
              ("AI (DN số)", base["AI"][:T]), ("H (nhân lực số)", base["H"][:T]),
              ("Y (sản lượng)", base["Y"]), ("C (tiêu dùng)", base["C"])]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[s[0] for s in series])
    for i, (name, val) in enumerate(series):
        fig.add_scatter(x=years, y=val, mode="lines+markers", name=name,
                        row=i // 3 + 1, col=i % 3 + 1, line_color=SEQ[i % len(SEQ)])
    fig.update_layout(height=560, showlegend=False, title_text="Quỹ đạo tối ưu 2026-2035")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Câu 8.3.3 — Cú sốc 2028: Y giảm 8% (như bão Yagi)")
    shock_pct = st.slider("Mức sốc giảm Y năm 2028 (%)", 0, 20, 8, 1) / 100
    shock_idx = int(np.where(years == 2028)[0][0])
    shock = build_solve(rho=rho, shock_idx=shock_idx, shock_pct=shock_pct)
    c1, c2 = st.columns(2)
    c1.metric("Welfare sau sốc", f"{shock['welfare']:.4f}",
              f"{shock['welfare']-base['welfare']:.4f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Sản lượng Y", "Tổng đầu tư"])
    fig.add_scatter(x=years, y=base["Y"], mode="lines+markers", name="Kế hoạch gốc",
                    line_color="#27ae60", row=1, col=1)
    fig.add_scatter(x=years, y=shock["Y"], mode="lines+markers", name="Sau sốc 2028",
                    line=dict(color="#c0392b", dash="dash"), row=1, col=1)
    fig.add_scatter(x=years, y=base["I_total"], mode="lines+markers", name="Đầu tư gốc",
                    line_color="#27ae60", row=1, col=2, showlegend=False)
    fig.add_scatter(x=years, y=shock["I_total"], mode="lines+markers", name="Đầu tư sau sốc",
                    line=dict(color="#c0392b", dash="dash"), row=1, col=2, showlegend=False)
    fig.add_vline(x=2028, line_dash="dot", line_color="gray")
    fig.update_layout(height=400, title_text="Phản ứng của mô hình trước cú sốc 2028")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Câu 8.3.4 — Trải đều vs Front-load (cùng tổng ngân sách)")
    TOTAL = base["I_total"].sum()
    even = np.full(T, TOTAL / T)
    fw = np.array([3, 3, 3, 1, 1, 1, 0.5, 0.5, 0.5, 0.5])
    front = TOTAL * fw / fw.sum()
    sol_e = build_solve(rho=rho, fixed_invest=even)
    sol_f = build_solve(rho=rho, fixed_invest=front)
    winner = "FRONT-LOAD" if sol_f["welfare"] > sol_e["welfare"] else "TRẢI ĐỀU"
    c1, c2, c3 = st.columns(3)
    c1.metric("Welfare trải đều", f"{sol_e['welfare']:.4f}")
    c2.metric("Welfare front-load", f"{sol_f['welfare']:.4f}")
    c3.metric("Thắng", winner, f"{abs(sol_f['welfare']-sol_e['welfare']):.4f}")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Lịch đầu tư", "Quỹ đạo tiêu dùng C"])
    fig.add_bar(x=years, y=even, name="Trải đều", marker_color="#3498db", row=1, col=1)
    fig.add_bar(x=years, y=front, name="Front-load", marker_color="#e67e22", row=1, col=1)
    fig.add_scatter(x=years, y=sol_e["C"], mode="lines+markers", name="C trải đều",
                    line_color="#3498db", row=1, col=2)
    fig.add_scatter(x=years, y=sol_f["C"], mode="lines+markers", name="C front-load",
                    line_color="#e67e22", row=1, col=2)
    fig.update_layout(height=400, title_text=f"{winner} cho welfare cao hơn")
    st.plotly_chart(fig, use_container_width=True)

    policy_box(8)