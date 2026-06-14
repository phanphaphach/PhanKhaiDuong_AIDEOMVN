import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.constants import SEQ
from src.components import header, policy_box

def render(sectors):
    header("🔹 Bài 3 — Chỉ số ưu tiên ngành (Priority) cho 10 ngành",
           "Xây dựng chỉ số ưu tiên định lượng để xác định ngành nào nên đẩy mạnh "
           "chuyển đổi số & AI trước, tạo hiệu ứng lan tỏa tối đa.")
    df = sectors.copy()
    cols_good = ["growth_rate_2024_pct", "gdp_share_2024_pct", "spillover_coef_0_1",
                 "export_billion_USD", "labor_million", "ai_readiness_0_100"]
    col_bad = "automation_risk_pct"
    crit_labels = ["Tăng trưởng", "GDP-share", "Lan tỏa", "Xuất khẩu",
                   "Việc làm", "AI readiness", "Rủi ro (đảo)"]

    norm_good = lambda x: (x - x.min()) / (x.max() - x.min())
    norm_bad = lambda x: (x.max() - x) / (x.max() - x.min())
    Xg = df[cols_good].apply(norm_good)
    Xb = norm_bad(df[col_bad])
    Xnorm = Xg.copy()
    Xnorm["risk_inv"] = Xb.values
    Xnorm.columns = crit_labels
    X7 = Xnorm.values

    # --- Câu 3.4.1 ---
    st.subheader("Câu 3.4.1 — Ma trận chuẩn hóa min-max 7 tiêu chí (Risk đảo dấu)")
    nv = Xnorm.copy()
    nv.insert(0, "Ngành", df["sector_name_vi"].values)
    st.dataframe(nv.round(3), use_container_width=True, hide_index=True)

    # --- Câu 3.4.2 ---
    st.subheader("Câu 3.4.2 — Priority (trọng số mặc định) & xếp hạng")
    w_raw = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20, 0.15])
    w_default = w_raw / w_raw.sum()
    df["Priority"] = X7 @ w_default
    rank = df[["sector_name_vi", "Priority"]].sort_values("Priority", ascending=False)
    rank = rank.reset_index(drop=True)
    rank.index += 1
    fig = px.bar(rank.sort_values("Priority"), x="Priority", y="sector_name_vi",
                 orientation="h", color="Priority", color_continuous_scale="RdYlGn",
                 title="Điểm Priority theo trọng số mặc định", labels={"sector_name_vi": ""})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rank.rename(columns={"sector_name_vi": "Ngành"}), use_container_width=True)
    st.success(f"**TOP-3 ưu tiên:** {', '.join(rank['sector_name_vi'].head(3).tolist())}")

    # --- Câu 3.4.3: độ nhạy a6 ---
    st.subheader("Câu 3.4.3 — Độ nhạy theo trọng số $a_6$ (AI readiness)")
    a6_values = np.arange(0.05, 0.40 + 1e-9, 0.05)
    idx_a6 = 5
    base_other = np.delete(w_raw, idx_a6)
    base_other = base_other / base_other.sum()
    sectors_vi = df["sector_name_vi"].values
    rank_matrix = np.zeros((len(a6_values), len(sectors_vi)), dtype=int)
    top3_track = []
    for r, a6 in enumerate(a6_values):
        w = np.zeros(7)
        w[idx_a6] = a6
        w_rest = base_other * (1 - a6)
        j = 0
        for k in range(7):
            if k != idx_a6:
                w[k] = w_rest[j]; j += 1
        scores = X7 @ w
        rank_matrix[r] = (-scores).argsort().argsort() + 1
        top3_track.append([sectors_vi[i] for i in np.argsort(-scores)[:3]])
    changed = len({tuple(t) for t in top3_track}) > 1
    fig = px.imshow(rank_matrix.T, aspect="auto", color_continuous_scale="RdYlGn_r",
                    labels=dict(x="Trọng số a6", y="", color="Hạng"),
                    x=[f"{v:.2f}" for v in a6_values], y=list(sectors_vi),
                    text_auto=True, title="Heatmap thứ hạng khi thay đổi a6")
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"TOP-3 **{'CÓ' if changed else 'KHÔNG'} thay đổi** khi a6 biến thiên 0.05→0.40.")

    # --- Câu 3.4.4 ---
    st.subheader("Câu 3.4.4 — So sánh 'Định hướng tăng trưởng' vs 'Bao trùm'")
    w_growth = np.array([0.25, 0.20, 0.05, 0.20, 0.05, 0.20, 0.05])
    w_incl = np.array([0.10, 0.05, 0.25, 0.05, 0.25, 0.05, 0.25])
    w_growth /= w_growth.sum(); w_incl /= w_incl.sum()
    df["P_growth"] = X7 @ w_growth
    df["P_incl"] = X7 @ w_incl
    rg = df.sort_values("P_growth", ascending=False)["sector_name_vi"].head(3).tolist()
    ri = df.sort_values("P_incl", ascending=False)["sector_name_vi"].head(3).tolist()
    comp = pd.DataFrame({"Hạng": [1, 2, 3], "Tăng trưởng": rg, "Bao trùm": ri})
    cf = df[["sector_name_vi", "P_growth", "P_incl"]].sort_values("P_growth")
    fig = go.Figure()
    fig.add_bar(y=cf["sector_name_vi"], x=cf["P_growth"], orientation="h",
                name="Định hướng tăng trưởng", marker_color="#e67e22")
    fig.add_bar(y=cf["sector_name_vi"], x=cf["P_incl"], orientation="h",
                name="Định hướng bao trùm", marker_color="#16a085")
    fig.update_layout(barmode="group", title="Priority theo 2 định hướng chính sách",
                      xaxis_title="Priority")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.caption(f"Chung cả 2 top-3: {sorted(set(rg) & set(ri)) or 'không có'}")

    policy_box(3)