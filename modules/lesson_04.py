import streamlit as st
import numpy as np
import pandas as pd
import pulp
import plotly.express as px
import plotly.graph_objects as go

from src.components import header, policy_box

def render():
    header("🔹 Bài 4 — LP phân bổ ngân sách số theo ngành-vùng",
           "Phân bổ 50 nghìn tỷ VND cho 6 vùng × 4 hạng mục (I, D, AI, H) tối đa hóa "
           "GDP gain với ràng buộc công bằng vùng miền.")

    reg_codes = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items = ["I", "D", "AI", "H"]
    region_vi = {"NMM": "Trung du-MN Bắc", "RRD": "ĐB sông Hồng", "NCC": "BTB-DH Trung",
                 "CH": "Tây Nguyên", "SE": "Đông Nam Bộ", "MD": "ĐB sông Cửu Long"}
    item_vi = {"I": "Hạ tầng (I)", "D": "Số hóa (D)", "AI": "Trí tuệ NT (AI)", "H": "Nhân lực (H)"}
    beta = {("NMM", "I"): 1.15, ("NMM", "D"): 0.85, ("NMM", "AI"): 0.55, ("NMM", "H"): 1.30,
            ("RRD", "I"): 0.95, ("RRD", "D"): 1.25, ("RRD", "AI"): 1.40, ("RRD", "H"): 1.05,
            ("NCC", "I"): 1.05, ("NCC", "D"): 0.95, ("NCC", "AI"): 0.85, ("NCC", "H"): 1.15,
            ("CH", "I"): 1.20, ("CH", "D"): 0.75, ("CH", "AI"): 0.45, ("CH", "H"): 1.35,
            ("SE", "I"): 0.90, ("SE", "D"): 1.30, ("SE", "AI"): 1.55, ("SE", "H"): 1.00,
            ("MD", "I"): 1.10, ("MD", "D"): 0.85, ("MD", "AI"): 0.65, ("MD", "H"): 1.25}
    D0 = {"NMM": 38, "RRD": 78, "NCC": 55, "CH": 32, "SE": 82, "MD": 48}
    gamma_c, BUDGET, R_MIN, R_MAX, H_MIN = 0.002, 50000, 5000, 12000, 12000

    D_floor_max = max(D0.values())
    worst = min(D0, key=D0.get)
    D_ceil_worst = D0[worst] + gamma_c * R_MAX
    lam_max = D_ceil_worst / D_floor_max

    st.subheader("⚠️ Chẩn đoán ràng buộc công bằng C5")
    st.warning(f"Với λ=0.70 như đề, mô hình **BẤT KHẢ THI**: cần Dmax ≥ {D_floor_max} "
               f"(vùng SE) nhưng vùng yếu nhất ({worst}) dồn hết {R_MAX} chỉ đạt "
               f"D = {D_ceil_worst:.1f}. **λ khả thi lớn nhất ≈ {lam_max:.4f}**. "
               f"App dùng λ = 0.68 cho mô hình 'có công bằng'.")
    LAM = st.slider("Chọn λ (hệ số công bằng vùng)", 0.30, float(round(lam_max, 2)), 0.68, 0.01)

    @st.cache_data
    def solve_pulp(lam, fair=True):
        m = pulp.LpProblem("VN_Digital_Budget", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", (reg_codes, items), lowBound=0)
        m += pulp.lpSum(beta[(r, j)] * x[r][j] for r in reg_codes for j in items)
        m += pulp.lpSum(x[r][j] for r in reg_codes for j in items) <= BUDGET
        for r in reg_codes:
            m += pulp.lpSum(x[r][j] for j in items) >= R_MIN
            m += pulp.lpSum(x[r][j] for j in items) <= R_MAX
        m += pulp.lpSum(x[r]["H"] for r in reg_codes) >= H_MIN
        if fair:
            Mv = pulp.LpVariable("Dmax", lowBound=0)
            for r in reg_codes:
                m += D0[r] + gamma_c * x[r]["D"] <= Mv
                m += D0[r] + gamma_c * x[r]["D"] >= lam * Mv
        m.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[m.status]
        if status != "Optimal":
            return status, None, None
        X = np.array([[x[r][j].value() for j in items] for r in reg_codes])
        return status, pulp.value(m.objective), X

    st.subheader(f"Câu 4.4.1 — Giải bằng PuLP/CBC (có C5, λ={LAM})")
    status_p, Z_pulp, X_pulp = solve_pulp(LAM, True)
    if status_p != "Optimal":
        st.error(f"Trạng thái: {status_p} — thử giảm λ.")
    else:
        st.metric("Z* (PuLP) — GDP gain", f"{Z_pulp:,.1f} tỷ VND")
        X_df = pd.DataFrame(X_pulp, index=[region_vi[r] for r in reg_codes],
                            columns=[item_vi[j] for j in items])
        X_df["TỔNG vùng"] = X_df.sum(axis=1)
        st.dataframe(X_df.round(1), use_container_width=True)

        # --- Câu 4.4.2: CVXPY ---
        st.subheader(f"Câu 4.4.2 — Giải bằng CVXPY & so sánh PuLP")
        try:
            import cvxpy as cp
            B_mat = np.array([[beta[(r, j)] for j in items] for r in reg_codes])
            X = cp.Variable((6, 4), nonneg=True)
            cons = [cp.sum(X) <= BUDGET, cp.sum(X[:, items.index("H")]) >= H_MIN]
            for i in range(6):
                cons += [cp.sum(X[i, :]) >= R_MIN, cp.sum(X[i, :]) <= R_MAX]
            Mv = cp.Variable(nonneg=True)
            d0 = np.array([D0[r] for r in reg_codes])
            for i in range(6):
                cons += [d0[i] + gamma_c * X[i, items.index("D")] <= Mv,
                         d0[i] + gamma_c * X[i, items.index("D")] >= LAM * Mv]
            probc = cp.Problem(cp.Maximize(cp.sum(cp.multiply(B_mat, X))), cons)
            probc.solve(solver=cp.HIGHS)
            same = abs(probc.value - Z_pulp) < 1e-2
            c1, c2 = st.columns(2)
            c1.metric("Z* (CVXPY)", f"{probc.value:,.1f}")
            c2.metric("Z* (PuLP)", f"{Z_pulp:,.1f}")
            st.success(f"Hai phương pháp cho kết quả **{'GIỐNG NHAU' if same else 'KHÁC NHAU'}** "
                       "(cùng LP lồi ⇒ optimum toàn cục duy nhất về Z*).")
        except Exception as e:
            st.info(f"CVXPY không khả dụng ({e}); kết quả PuLP vẫn hợp lệ.")

        # --- Câu 4.4.3: heatmap ---
        st.subheader("Câu 4.4.3 — Heatmap phân bổ tối ưu")
        fig = px.imshow(X_pulp, color_continuous_scale="YlOrRd", aspect="auto",
                        text_auto=".0f",
                        labels=dict(x="Hạng mục", y="Vùng", color="Tỷ VND"),
                        x=[item_vi[j] for j in items], y=[region_vi[r] for r in reg_codes],
                        title=f"Phân bổ tối ưu x[vùng, hạng mục] (λ={LAM})")
        st.plotly_chart(fig, use_container_width=True)
        reg_tot = X_pulp.sum(axis=1)
        top_r = reg_codes[int(reg_tot.argmax())]
        st.info(f"Vùng nhận nhiều ngân sách nhất: **{region_vi[top_r]}** ({reg_tot.max():,.0f} tỷ).")

        # --- Câu 4.4.4: chi phí công bằng ---
        st.subheader("Câu 4.4.4 — Chi phí kinh tế của công bằng vùng miền (bỏ C5)")
        st_nf, Z_nf, X_nf = solve_pulp(LAM, False)
        cost_fair = Z_nf - Z_pulp
        c1, c2, c3 = st.columns(3)
        c1.metric("Z* bỏ C5", f"{Z_nf:,.0f}")
        c2.metric("Z* có C5", f"{Z_pulp:,.0f}")
        c3.metric("Chi phí công bằng", f"{cost_fair:,.0f} tỷ", f"{cost_fair/Z_nf*100:.2f}%")
        cmp = pd.DataFrame({"Vùng": [region_vi[r] for r in reg_codes],
                            "Tổng (có C5)": X_pulp.sum(axis=1).round(0),
                            "Tổng (bỏ C5)": X_nf.sum(axis=1).round(0)})
        fig = go.Figure()
        fig.add_bar(y=cmp["Vùng"], x=cmp["Tổng (có C5)"], orientation="h",
                    name=f"Có công bằng (λ={LAM})", marker_color="#16a085")
                    
        fig.add_bar(y=cmp["Vùng"], x=cmp["Tổng (bỏ C5)"], orientation="h",
                    name="Bỏ công bằng (max hiệu quả)", marker_color="#e74c3c")
        fig.update_layout(barmode="group", title="Phân bổ vùng: có vs bỏ ràng buộc công bằng",
                          xaxis_title="Tổng ngân sách vùng (tỷ VND)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

    policy_box(4)