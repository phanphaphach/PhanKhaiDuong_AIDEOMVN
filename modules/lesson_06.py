import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.constants import SEQ
from src.components import header, policy_box

def render(regions):
    header("🔹 Bài 6 — TOPSIS xếp hạng 6 vùng theo mức độ sẵn sàng AI",
           "Áp dụng MCDM-TOPSIS để chọn vùng triển khai trung tâm AI & sandbox dữ liệu "
           "trước; so sánh trọng số chuyên gia, Entropy và AHP.")
    df = regions.copy()
    criteria = ["grdp_per_capita_million_VND", "fdi_registered_billion_USD",
                "digital_index_0_100", "ai_readiness_0_100", "trained_labor_pct",
                "rd_intensity_pct", "internet_penetration_pct", "gini_coef"]
    crit_short = ["GRDP/người", "FDI", "Chỉ số số", "AI readiness", "LĐ đào tạo",
                  "R&D", "Internet", "Gini"]
    is_benefit = np.array([True, True, True, True, True, True, True, False])
    X = df[criteria].values.astype(float)
    names_vi = df["region_name_vi"].values
    n_alt, n_crit = X.shape

    def topsis(X, w, is_benefit):
        R = X / np.sqrt((X ** 2).sum(axis=0))
        V = R * w
        A_star = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
        A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))
        S_star = np.sqrt(((V - A_star) ** 2).sum(axis=1))
        S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
        return S_neg / (S_star + S_neg)

    # --- Câu 6.4.1 ---
    st.subheader("Câu 6.4.1 — TOPSIS với trọng số chuyên gia")
    w_expert = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])
    C_exp = topsis(X, w_expert, is_benefit)
    t = pd.DataFrame({"Vùng": names_vi, "C* (chuyên gia)": C_exp}).sort_values(
        "C* (chuyên gia)", ascending=False).reset_index(drop=True)
    t.index += 1
    fig = px.bar(t.sort_values("C* (chuyên gia)"), x="C* (chuyên gia)", y="Vùng",
                 orientation="h", color="C* (chuyên gia)", color_continuous_scale="Tealgrn",
                 title="Điểm TOPSIS C* (trọng số chuyên gia)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(t.round(4), use_container_width=True)
    st.success(f"**TOP-3:** {', '.join(t['Vùng'].head(3).tolist())}")

    # --- Câu 6.4.2: Entropy ---
    st.subheader("Câu 6.4.2 — Trọng số khách quan bằng Entropy")
    def entropy_weights(X, is_benefit):
        Xc = X.copy().astype(float)
        for j in range(Xc.shape[1]):
            if not is_benefit[j]:
                Xc[:, j] = Xc[:, j].max() + Xc[:, j].min() - Xc[:, j]
        Pm = Xc / Xc.sum(axis=0)
        k = 1.0 / np.log(len(Xc))
        E = -k * np.nansum(Pm * np.log(Pm + 1e-12), axis=0)
        d = 1 - E
        return d / d.sum()
    w_ent = entropy_weights(X, is_benefit)
    C_ent = topsis(X, w_ent, is_benefit)
    wdf = pd.DataFrame({"Tiêu chí": crit_short, "Chuyên gia": w_expert.round(4),
                        "Entropy": w_ent.round(4)})
    fig = go.Figure()
    fig.add_bar(x=crit_short, y=w_expert, name="Chuyên gia", marker_color="#2980b9")
    fig.add_bar(x=crit_short, y=w_ent, name="Entropy", marker_color="#e67e22")
    fig.update_layout(barmode="group", title="So sánh trọng số chuyên gia vs Entropy")
    st.plotly_chart(fig, use_container_width=True)
    cmp = pd.DataFrame({"Vùng": names_vi,
                        "Hạng chuyên gia": pd.Series(C_exp).rank(ascending=False).astype(int).values,
                        "Hạng entropy": pd.Series(C_ent).rank(ascending=False).astype(int).values})
    d2 = (cmp["Hạng chuyên gia"] - cmp["Hạng entropy"]) ** 2
    rho = 1 - 6 * d2.sum() / (n_alt * (n_alt ** 2 - 1))
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.info(f"Spearman ρ = **{rho:.4f}** — Entropy dồn trọng số vào tiêu chí phân hóa "
            f"mạnh (FDI, R&D); nhóm dẫn đầu nhìn chung ổn định.")

    # --- Câu 6.4.3: độ nhạy w_AI ---
    st.subheader("Câu 6.4.3 — Độ nhạy theo trọng số $w_{AI}$")
    idx_ai = criteria.index("ai_readiness_0_100")
    base_other = np.delete(w_expert, idx_ai)
    base_other = base_other / base_other.sum()
    w_ai_grid = np.arange(0.10, 0.40 + 1e-9, 0.05)
    track = np.zeros((len(w_ai_grid), n_alt))
    top3 = []
    for r, wai in enumerate(w_ai_grid):
        w = np.zeros(n_crit)
        w[idx_ai] = wai
        j = 0
        for k in range(n_crit):
            if k != idx_ai:
                w[k] = base_other[j] * (1 - wai); j += 1
        Cc = topsis(X, w, is_benefit)
        track[r] = Cc
        top3.append([names_vi[i] for i in np.argsort(-Cc)[:3]])
    stable = len({tuple(x) for x in top3}) == 1
    fig = go.Figure()
    for i in range(n_alt):
        fig.add_scatter(x=w_ai_grid, y=track[:, i], mode="lines+markers", name=names_vi[i])
    fig.update_layout(title="Điểm TOPSIS C* theo trọng số w_AI",
                      xaxis_title="w_AI", yaxis_title="C*")
    st.plotly_chart(fig, use_container_width=True)

    # --- Câu 6.4.4: AHP ---
    st.subheader("Câu 6.4.4 — AHP đơn giản (eigenvector + CR) so với TOPSIS")
    A_pair = np.outer(w_expert, 1.0 / w_expert)
    eigvals, eigvecs = np.linalg.eig(A_pair)
    imax = np.argmax(eigvals.real)
    lam_max = eigvals.real[imax]
    pv = np.abs(eigvecs[:, imax].real)
    w_ahp = pv / pv.sum()
    CI = (lam_max - n_crit) / (n_crit - 1)
    RI = {6: 1.24, 7: 1.32, 8: 1.41}[n_crit]
    CR = CI / RI
    C_ahp = topsis(X, w_ahp, is_benefit)
    final = pd.DataFrame({"Vùng": names_vi, "C* chuyên gia": C_exp.round(4),
                          "C* entropy": C_ent.round(4), "C* AHP": C_ahp.round(4)})
    c1, c2, c3 = st.columns(3)
    c1.metric("λ_max", f"{lam_max:.3f}")
    c2.metric("CI", f"{CI:.4f}")
    c3.metric("CR", f"{CR:.4f}", "Nhất quán" if CR < 0.1 else "Không nhất quán")
    fig = go.Figure()
    order = final.sort_values("C* chuyên gia", ascending=False)["Vùng"]
    for col, color in [("C* chuyên gia", "#2980b9"), ("C* entropy", "#e67e22"),
                       ("C* AHP", "#16a085")]:
        fig.add_bar(x=order, y=[final.set_index("Vùng").loc[v, col] for v in order],
                    name=col, marker_color=color)
    fig.update_layout(barmode="group", title="So sánh điểm TOPSIS theo 3 bộ trọng số")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(final, use_container_width=True, hide_index=True)

    policy_box(6)