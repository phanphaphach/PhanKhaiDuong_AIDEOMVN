import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pulp import (LpProblem, LpMaximize, LpVariable, lpSum,
                  PULP_CBC_CMD, LpStatus, value)

from src.components import header, policy_box

def render():
    header("🔹 Bài 5 — MIP lựa chọn danh mục dự án chuyển đổi số",
           "Chọn tập dự án tối ưu trong 15 dự án ứng cử (ngân sách 80 nghìn tỷ) với "
           "ràng buộc loại trừ, tiên quyết, bắt buộc và đếm số lượng.")

    P = list(range(1, 16))
    proj_names = {1: "TT dữ liệu Hòa Lạc", 2: "TT dữ liệu phía Nam", 3: "5G toàn quốc",
                  4: "VNeID 2.0", 5: "Cổng DVC v3", 6: "Y tế số", 7: "Giáo dục số K-12",
                  8: "TT AI + supercomputing", 9: "Sandbox fintech", 10: "Logistics thông minh",
                  11: "Nông nghiệp số ĐBSCL", 12: "Đào tạo 50k kỹ sư AI", 13: "Bán dẫn Bắc Ninh",
                  14: "An ninh mạng SOC", 15: "Open Data quốc gia"}
    C = {1: 12000, 2: 11500, 3: 18000, 4: 4500, 5: 3200, 6: 5800, 7: 6500, 8: 15000,
         9: 2500, 10: 7200, 11: 4800, 12: 8500, 13: 20000, 14: 3800, 15: 1500}
    C1 = {1: 8500, 2: 7500, 3: 12000, 4: 3500, 5: 2500, 6: 4000, 7: 4500, 8: 9000,
          9: 1800, 10: 5000, 11: 3500, 12: 5500, 13: 13000, 14: 2800, 15: 1200}
    B = {1: 21500, 2: 20800, 3: 32500, 4: 9200, 5: 6800, 6: 11400, 7: 12200, 8: 28500,
         9: 5800, 10: 13800, 11: 8500, 12: 16200, 13: 35000, 14: 7500, 15: 3800}
    N_MIN, N_MAX = 7, 11

    def build_solve(cap_budget=80000, op_budget=40000, obj_coeff=None, p1p2="exclusive"):
        coeff = obj_coeff if obj_coeff is not None else B
        m = LpProblem("VN_Project", LpMaximize)
        y = LpVariable.dicts("y", P, cat="Binary")
        m += lpSum(coeff[i] * y[i] for i in P)
        m += lpSum(C[i] * y[i] for i in P) <= cap_budget
        m += lpSum(C1[i] * y[i] for i in P) <= op_budget
        if p1p2 == "exclusive":
            m += y[1] + y[2] <= 1
        elif p1p2 == "both":
            m += y[1] == 1; m += y[2] == 1
        m += y[8] <= y[12]
        m += y[13] <= y[12]
        m += y[4] + y[5] >= 1
        m += y[14] >= 1
        m += lpSum(y[i] for i in P) >= N_MIN
        m += lpSum(y[i] for i in P) <= N_MAX
        m.solve(PULP_CBC_CMD(msg=False))
        status = LpStatus[m.status]
        if status != "Optimal":
            return status, None, []
        sel = [i for i in P if y[i].value() > 0.5]
        return status, value(m.objective), sel

    # --- Câu 5.4.1 ---
    st.subheader("Câu 5.4.1 — Giải bằng PuLP/CBC")
    cap_b = st.slider("Ngân sách đầu tư (nghìn tỷ)", 80, 120, 80, 5) * 1000
    st1, Z1, sel1 = build_solve(cap_budget=cap_b)
    tot_c = sum(C[i] for i in sel1)
    tot_c1 = sum(C1[i] for i in sel1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Z* tổng lợi ích", f"{Z1:,.0f} tỷ")
    c2.metric("Vốn đầu tư dùng", f"{tot_c:,.0f} / {cap_b:,}")
    c3.metric("Vận hành dùng", f"{tot_c1:,.0f} / 40,000")
    det = pd.DataFrame({"Dự án": [f"P{i} — {proj_names[i]}" for i in sel1],
                        "Vốn ĐT": [C[i] for i in sel1], "Vận hành": [C1[i] for i in sel1],
                        "Lợi ích B": [B[i] for i in sel1],
                        "B/C": [round(B[i] / C[i], 2) for i in sel1]})
    fig = px.bar(det.sort_values("Lợi ích B"), x="Lợi ích B", y="Dự án", orientation="h",
                 color="B/C", color_continuous_scale="Viridis",
                 title="Danh mục dự án được chọn")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(det, use_container_width=True, hide_index=True)
    npv_cap = Z1 / tot_c
    st.info(f"NPV biên (Z*/vốn đầu tư) = **{npv_cap:.2f}** — mỗi 1 tỷ vốn đầu tư tạo "
            f"{npv_cap:.2f} tỷ lợi ích. Nút thắt thực sự là **ngân sách vận hành** "
            f"(dùng {tot_c1:,.0f}/40,000), không phải vốn đầu tư.")

    # --- Câu 5.4.2: nới ngân sách vận hành ---
    st.subheader("Câu 5.4.2 — Z* theo ngân sách vận hành (nút thắt thực sự)")
    op_grid = list(range(40000, 70001, 5000))
    Z_byop = [build_solve(cap_budget=100000, op_budget=ob)[1] for ob in op_grid]
    fig = px.line(x=op_grid, y=Z_byop, markers=True,
                  labels={"x": "Ngân sách vận hành C1 (tỷ VND)", "y": "Z* (tỷ VND)"},
                  title="Z* tăng khi nới ngân sách vận hành")
    fig.update_traces(line_color="#8e44ad")
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 5.4.3: redundancy P1 & P2 ---
    st.subheader("Câu 5.4.3 — Quốc hội yêu cầu cả P1 và P2 (redundancy)")
    st3, Z3, sel3 = build_solve(p1p2="both")
    if st3 == "Optimal":
        st.metric("Z* (buộc P1 & P2)", f"{Z3:,.0f} tỷ", f"{Z3-Z1:,.0f}")
        st.warning(f"If GIỮ ràng buộc loại trừ gốc (y1+y2≤1) thì **bất khả thi**. "
                   f"Hiểu redundancy là THAY ràng buộc loại trừ ⇒ khả thi, nhưng Z* giảm "
                   f"{Z1-Z3:,.0f} tỷ ({(Z1-Z3)/Z1*100:.2f}%) — chi phí của yêu cầu dự phòng.")
        st.caption("Dự án chọn: " + ", ".join(f"P{i}" for i in sel3))

    # --- Câu 5.4.4: rủi ro tiến độ ---
    st.subheader("Câu 5.4.4 — Tối đa hóa lợi ích KỲ VỌNG (rủi ro tiến độ)")
    infra, ai_semi, digi_gov = [1, 2, 3], [8, 12, 13], [6, 7, 10, 11]
    p = {i: 0.80 for i in P}
    for i in infra: p[i] = 0.85
    for i in ai_semi: p[i] = 0.65
    for i in digi_gov: p[i] = 0.75
    coeff_exp = {i: p[i] * B[i] for i in P}
    st4, EZ, sel4 = build_solve(obj_coeff=coeff_exp)
    st.metric("E[Z] tối ưu (kỳ vọng)", f"{EZ:,.0f} tỷ")
    added = sorted(set(sel4) - set(sel1))
    removed = sorted(set(sel1) - set(sel4))
    dfp = pd.DataFrame({"Dự án": [f"P{i}" for i in P],
                        "Lợi ích B": [B[i] for i in P],
                        "Kỳ vọng p·B": [round(p[i] * B[i]) for i in P],
                        "Xác suất p": [p[i] for i in P]})
    fig = go.Figure()
    fig.add_bar(x=dfp["Dự án"], y=dfp["Lợi ích B"], name="Lợi ích B", marker_color="#bdc3c7")
    fig.add_bar(x=dfp["Dự án"], y=dfp["Kỳ vọng p·B"], name="Kỳ vọng p·B", marker_color="#2980b9")
    fig.update_layout(barmode="group", title="Lợi ích danh nghĩa vs kỳ vọng (có rủi ro)",
                      yaxis_title="Tỷ VND")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"So với 5.4.1: thêm {['P'+str(i) for i in added] or 'không'}, "
            f"bớt {['P'+str(i) for i in removed] or 'không'}. Các dự án AI/bán dẫn "
            f"(p=0.65) bị 'phạt' mạnh ⇒ danh mục dịch chuyển sang dự án chắc chắn hơn.")

    policy_box(5)
