import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import linprog
import pulp

from src.constants import SEQ
from src.components import header, policy_box

def render():
    header("🔹 Bài 2 — LP phân bổ ngân sách đầu tư số",
           "Phân bổ 100 nghìn tỷ VND cho 4 hạng mục (hạ tầng số, R&D/AI, nhân lực số, "
           "doanh nghiệp AI) tối đa hóa tăng GDP kỳ vọng; phân tích giá đối ngẫu.")

    profit = np.array([0.85, 1.20, 0.95, 1.35])
    names = ["x1: Hạ tầng số", "x2: R&D công nghệ", "x3: Nhân lực số", "x4: Doanh nghiệp AI"]
    BUDGET = 100

    def build_constraints(B, x3_min=20):
        A_ub = [[1, 1, 1, 1], [-1, 0, 0, 0], [0, -1, 0, 0],
                [0, 0, -1, 0], [0, 0, 0, -1], [0.35, -0.65, 0.35, -0.65]]
        b_ub = [B, -25, -15, -x3_min, -10, 0]
        return A_ub, b_ub

    # --- Câu 2.4.1: scipy.linprog ---
    st.subheader("Câu 2.4.1 — Giải bằng `scipy.optimize.linprog` (HiGHS)")
    c = (-profit).tolist()
    A_ub, b_ub = build_constraints(BUDGET)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    Z_opt, x_opt = -res.fun, res.x
    m1, m2 = st.columns(2)
    m1.metric("Z* tối ưu (lợi ích kỳ vọng)", f"{Z_opt:.3f} nghìn tỷ")
    m2.metric("Vốn sử dụng", f"{x_opt.sum():.1f} / {BUDGET}")
    t241 = pd.DataFrame({"Hạng mục": names, "Phân bổ x* (ngh.tỷ)": x_opt.round(3),
                         "Tỷ trọng %": (x_opt / x_opt.sum() * 100).round(1),
                         "Hệ số lợi ích": profit})
    fig = px.pie(t241, names="Hạng mục", values="Phân bổ x* (ngh.tỷ)",
                 color_discrete_sequence=SEQ, title="Cơ cấu phân bổ tối ưu")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t241, use_container_width=True, hide_index=True)

    # --- Câu 2.4.2: PuLP + dual ---
    st.subheader("Câu 2.4.2 — Giải bằng PuLP & giá đối ngẫu (shadow price)")
    prob = pulp.LpProblem("Budget_Allocation", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{i+1}", lowBound=0) for i in range(4)]
    prob += pulp.lpSum(profit[i] * x[i] for i in range(4))
    prob += (pulp.lpSum(x) <= BUDGET, "NganSachTong")
    prob += (x[0] >= 25, "San_x1")
    prob += (x[1] >= 15, "San_x2")
    prob += (x[2] >= 20, "San_x3")
    prob += (x[3] >= 10, "San_x4")
    prob += (0.35 * (x[0] + x[2]) <= 0.65 * (x[1] + x[3]), "CanDoiCoCau")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    duals = pd.DataFrame([
        {"Ràng buộc": cn, "Shadow price": round(co.pi, 4), "Slack": round(co.slack, 4)}
        for cn, co in prob.constraints.items()])
    sp_budget = prob.constraints["NganSachTong"].pi
    st.dataframe(duals, use_container_width=True, hide_index=True)
    st.success(f"**Shadow price ngân sách tổng = {sp_budget:.4f}** — tăng 1 nghìn tỷ "
               f"ngân sách ⇒ Z* tăng thêm ≈ {sp_budget:.2f} nghìn tỷ. Ràng buộc ngân "
               f"sách đang *chặt* (binding): ngân sách là nguồn lực khan hiếm quyết định Z*.")

    # --- Câu 2.4.3: Độ nhạy Z*(B) ---
    st.subheader("Câu 2.4.3 — Độ nhạy: đường cong $Z^*(B)$")
    B_grid = np.linspace(90, 150, 61)
    Z_grid = []
    for B in B_grid:
        Ab, bb = build_constraints(B)
        r = linprog(c, A_ub=Ab, b_ub=bb, bounds=[(0, None)] * 4, method="highs")
        Z_grid.append(-r.fun)
    fig = px.line(x=B_grid, y=Z_grid, labels={"x": "Ngân sách tổng B (ngh.tỷ)", "y": "Z*"},
                  title="Đường cong giá trị tối ưu Z*(B)")
    fig.update_traces(line_color="#2980b9")
    for B in [100, 120, 140]:
        Ab, bb = build_constraints(B)
        zz = -linprog(c, A_ub=Ab, b_ub=bb, bounds=[(0, None)] * 4, method="highs").fun
        fig.add_scatter(x=[B], y=[zz], mode="markers+text", text=[f"{zz:.1f}"],
                        textposition="top center", marker=dict(color="#e74c3c", size=11),
                        showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 2.4.4: x3>=30 ---
    st.subheader("Câu 2.4.4 — Ưu tiên nhân lực số: ràng buộc $x_3 \\geq 30$")
    x3_min = st.slider("Sàn nhân lực số x3", 20, 45, 30, 1)
    An, bn = build_constraints(BUDGET, x3_min=x3_min)
    res2 = linprog(c, A_ub=An, b_ub=bn, bounds=[(0, None)] * 4, method="highs")
    if res2.status == 0:
        Z_new = -res2.fun
        dZ = Z_opt - Z_new
        m1, m2 = st.columns(2)
        m1.metric("Z* mới", f"{Z_new:.3f}", f"{-dZ:.3f}")
        m2.metric("Chi phí cơ hội", f"{dZ:.2f} nghìn tỷ ({dZ/Z_opt*100:.2f}%)")
        cmp = pd.DataFrame({"Hạng mục": names, "x* gốc (x3≥20)": x_opt.round(2),
                            f"x* mới (x3≥{x3_min})": res2.x.round(2)})
        figb = go.Figure()
        figb.add_bar(x=names, y=x_opt, name="Gốc (x3≥20)", marker_color="#3498db")
        figb.add_bar(x=names, y=res2.x, name=f"x3≥{x3_min}", marker_color="#e67e22")
        figb.update_layout(barmode="group", yaxis_title="Ngân sách (ngh.tỷ)",
                           title="Phân bổ: trước vs sau khi ưu tiên nhân lực số")
        st.plotly_chart(figb, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)
        st.info(f"Bài toán **vẫn khả thi** (tổng sàn = 25+15+{x3_min}+10 = {80+x3_min-30} ≤ 100). "
                f"Buộc rót thêm vào x3 (lợi ích biên 0.95) thay vì x4 (1.35) làm Z* giảm — "
                f"đây là chi phí cơ hội của mục tiêu chính sách.")
    else:
        st.error(f"Bài toán KHÔNG khả thi với x3 ≥ {x3_min}.")

    policy_box(2)