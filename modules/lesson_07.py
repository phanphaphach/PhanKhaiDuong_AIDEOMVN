import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.constants import SEQ
from src.components import header, policy_box

def render():
    header("🔹 Bài 7 — NSGA-II tối ưu đa mục tiêu Pareto",
           "24 biến (6 vùng × 4 hạng mục), 4 mục tiêu xung đột: tăng trưởng (max), "
           "bao trùm/bất bình đẳng (min), phát thải (min), rủi ro ròng (min).")

    reg_codes = ["NMM", "RRD", "NCC", "CH", "SE", "MD"]
    items = ["I", "D", "AI", "H"]
    region_vi = {"NMM": "Trung du-MN Bắc", "RRD": "ĐB sông Hồng", "NCC": "BTB-DH Trung",
                 "CH": "Tây Nguyên", "SE": "Đông Nam Bộ", "MD": "ĐB sông Cửu Long"}
    BETA = np.array([[1.15, 0.85, 0.55, 1.30], [0.95, 1.25, 1.40, 1.05],
                     [1.05, 0.95, 0.85, 1.15], [1.20, 0.75, 0.45, 1.35],
                     [0.90, 1.30, 1.55, 1.00], [1.10, 0.85, 0.65, 1.25]])
    E_FACTOR = np.array([0.42, 0.55, 0.48, 0.32, 0.62, 0.38])
    RHO_RISK = np.array([0.18, 0.45, 0.28, 0.12, 0.52, 0.22])
    SIG_MITIG = np.array([0.32, 0.28, 0.30, 0.35, 0.25, 0.30])
    BUDGET, R_MIN, R_MAX, H_MIN = 50000, 5000, 12000, 12000

    @st.cache_data(show_spinner="Đang chạy NSGA-II (pop=100, n_gen=200)...")
    def run_nsga():
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize

        class VNProblem(ElementwiseProblem):
            def __init__(self):
                super().__init__(n_var=24, n_obj=4, n_ieq_constr=14,
                                 xl=np.zeros(24), xu=np.ones(24) * R_MAX)

            def _evaluate(self, x, out, *a, **k):
                X = x.reshape(6, 4)
                f1 = -(BETA * X).sum()
                sums = X.sum(axis=1)
                f2 = np.abs(sums - sums.mean()).mean()
                f3 = (E_FACTOR * (X[:, 0] + X[:, 2])).sum()
                f4 = (RHO_RISK * X[:, 2]).sum() - (SIG_MITIG * X[:, 3]).sum()
                out["F"] = [f1, f2, f3, f4]
                g = [X.sum() - BUDGET]
                for r in range(6):
                    g.append(R_MIN - X[r].sum())
                    g.append(X[r].sum() - R_MAX)
                g.append(H_MIN - X[:, 3].sum())
                out["G"] = g

        res = minimize(VNProblem(), NSGA2(pop_size=100), ("n_gen", 200),
                       seed=42, verbose=False)
        return res.F, res.X, float(res.CV.max())

    try:
        F, Xsol, cv = run_nsga()
        GDP, INEQ, EMIS, RISK = -F[:, 0], F[:, 1], F[:, 2], F[:, 3]

        st.subheader("Câu 7.4.1 — Chạy NSGA-II")
        c1, c2 = st.columns(2)
        c1.metric("Số nghiệm Pareto", f"{len(F)}")
        c2.metric("Vi phạm ràng buộc lớn nhất (CV)", f"{cv:.4f}",
                  "Toàn bộ khả thi" if cv < 1e-6 else "")

        # --- Câu 7.4.2: scatter 3D + parallel ---
        st.subheader("Câu 7.4.2 — Trực quan hóa mặt Pareto")
        fig = px.scatter_3d(x=GDP, y=INEQ, z=EMIS, color=RISK,
                            color_continuous_scale="Viridis",
                            labels={"x": "GDP gain (f1)", "y": "Bất BĐ (f2)",
                                    "z": "Phát thải (f3)", "color": "Rủi ro (f4)"},
                            title="Mặt Pareto 3D (màu = rủi ro ròng f4)")
        st.plotly_chart(fig, use_container_width=True)
        Fp = np.column_stack([GDP, INEQ, EMIS, RISK])
        figp = px.parallel_coordinates(
            pd.DataFrame(Fp, columns=["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro"]),
            color=GDP, color_continuous_scale="Tealrose",
            title="Parallel coordinates — 4 mục tiêu trên tập Pareto")
        st.plotly_chart(figp, use_container_width=True)

        # --- Câu 7.4.3: TOPSIS chọn nghiệm thỏa hiệp ---
        st.subheader("Câu 7.4.3 — TOPSIS chọn nghiệm thỏa hiệp")
        cc = st.columns(4)
        w1 = cc[0].slider("Tăng trưởng", 0.0, 1.0, 0.40, 0.05)
        w2 = cc[1].slider("Bao trùm", 0.0, 1.0, 0.25, 0.05)
        w3 = cc[2].slider("Môi trường", 0.0, 1.0, 0.20, 0.05)
        w4 = cc[3].slider("An ninh", 0.0, 1.0, 0.15, 0.05)
        w_policy = np.array([w1, w2, w3, w4])
        w_policy = w_policy / w_policy.sum()

        def topsis_pareto(F, w):
            rng = F.max(0) - F.min(0)
            rng[rng == 0] = 1e-12
            norm = (F.max(0) - F) / rng
            V = norm * w
            ideal, anti = V.max(0), V.min(0)
            Sp = np.sqrt(((V - ideal) ** 2).sum(1))
            Sn = np.sqrt(((V - anti) ** 2).sum(1))
            return Sn / (Sp + Sn)

        Cc = topsis_pareto(F, w_policy)
        idx_comp = int(Cc.argmax())
        m = st.columns(4)
        m[0].metric("GDP gain", f"{GDP[idx_comp]:,.0f}")
        m[1].metric("Bất BĐ", f"{INEQ[idx_comp]:.1f}")
        m[2].metric("Phát thải", f"{EMIS[idx_comp]:,.0f}")
        m[3].metric("Rủi ro", f"{RISK[idx_comp]:,.0f}")
        X_comp = Xsol[idx_comp].reshape(6, 4)
        Xc_df = pd.DataFrame(X_comp, index=[region_vi[r] for r in reg_codes], columns=items)
        Xc_df["Tổng"] = Xc_df.sum(axis=1)
        st.dataframe(Xc_df.round(0), use_container_width=True)

        # --- Câu 7.4.4: chi phí cơ hội + radar ---
        st.subheader("Câu 7.4.4 — Chi phí cơ hội: tăng trưởng-max vs thỏa hiệp")
        idx_g = int(GDP.argmax())
        def pct(new, base):
            return (new - base) / abs(base) * 100 if base != 0 else 0
        rows = pd.DataFrame({
            "Mục tiêu": ["GDP gain", "Bất bình đẳng", "Phát thải", "Rủi ro ròng"],
            "Tăng trưởng-max": [GDP[idx_g], INEQ[idx_g], EMIS[idx_g], RISK[idx_g]],
            "Thỏa hiệp": [GDP[idx_comp], INEQ[idx_comp], EMIS[idx_comp], RISK[idx_comp]],
        })
        rows["Δ% so thỏa hiệp"] = [pct(rows.iloc[i, 1], rows.iloc[i, 2]) for i in range(4)]

        def to_good(F):
            rng = F.max(0) - F.min(0); rng[rng == 0] = 1e-12
            return (F.max(0) - F) / rng
        G = to_good(F)
        labels_r = ["Tăng trưởng", "Bao trùm", "Môi trường", "An ninh"]
        figr = go.Figure()
        for idx, lab, col in [(idx_comp, "Thỏa hiệp", "#16a085"),
                              (idx_g, "Tăng trưởng-max", "#e74c3c")]:
            figr.add_scatterpolar(r=G[idx].tolist() + [G[idx][0]],
                                  theta=labels_r + [labels_r[0]], fill="toself",
                                  name=lab, line_color=col)
        figr.update_layout(title="Radar: Thỏa hiệp vs Tăng trưởng-max (xa tâm = tốt hơn)",
                           polar=dict(radialaxis=dict(range=[0, 1])))
        st.plotly_chart(figr, use_container_width=True)
        st.dataframe(rows.round(1), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Lỗi khi chạy NSGA-II: {e}")

    policy_box(7)