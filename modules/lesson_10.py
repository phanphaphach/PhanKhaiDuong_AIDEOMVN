import streamlit as st
import numpy as np
import pandas as pd
import pyomo.environ as pyo
import plotly.express as px

from src.constants import SEQ
from src.components import header, policy_box

def render():
    header("🔹 Bài 10 — Quy hoạch ngẫu nhiên hai giai đoạn dưới bất định",
           "Quyết định ngân sách first-stage (here-and-now) trước khi biết kịch bản, "
           "second-stage recourse sau khi kịch bản hiện thực hóa; tính VSS & EVPI.")

    JS = ["I", "D", "AI", "H"]
    SS = ["s1", "s2", "s3", "s4"]
    SCN_NAME = {"s1": "Lạc quan", "s2": "Cơ sở", "s3": "Bi quan", "s4": "Khủng hoảng"}
    PROB = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
    BETA_B = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
    BETA_S = {("s1", "I"): 1.25, ("s1", "D"): 1.35, ("s1", "AI"): 1.55, ("s1", "H"): 1.05,
              ("s2", "I"): 1.00, ("s2", "D"): 1.10, ("s2", "AI"): 1.25, ("s2", "H"): 0.95,
              ("s3", "I"): 0.75, ("s3", "D"): 0.85, ("s3", "AI"): 0.90, ("s3", "H"): 1.00,
              ("s4", "I"): 0.40, ("s4", "D"): 0.50, ("s4", "AI"): 0.55, ("s4", "H"): 1.10}
    BUDGET1, RESERVE, COUPLE, SOFT_CAP, PEN1, PEN2 = 65000.0, 15000.0, 0.5, 9000.0, 0.05, 0.85

    # Định vị solver phù hợp tự động
    def get_solver():
        for name in ["glpk", "cbc", "appsi_highs"]:
            try:
                s = pyo.SolverFactory(name)
                if s.available(exception_flag=False):
                    return s, name
            except Exception:
                continue
        return None, None

    SOLVER, sname = get_solver()
    if SOLVER is None:
        st.error("Không tìm thấy solver (GLPK/CBC/HiGHS) cho Pyomo.")
        st.stop()
    st.caption(f"Solver Pyomo đang dùng: `{sname}`")

    def build_sp(fix_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=JS); m.S = pyo.Set(initialize=SS)
        m.p = pyo.Param(m.S, initialize=PROB)
        m.beta = pyo.Param(m.J, initialize=BETA_B)
        m.beta_s = pyo.Param(m.S, m.J, initialize=BETA_S)
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
        m.u_lo = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.u_hi = pyo.Var(m.S, within=pyo.NonNegativeReals)
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
        if fix_x is not None:
            m.fixx = pyo.ConstraintList()
            for j in JS:
                m.fixx.add(m.x[j] == float(fix_x[j]))
        m.budget2 = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) <= RESERVE)
        m.couple = pyo.Constraint(m.S, rule=lambda m, s: m.y[s, "AI"] <= COUPLE * m.x["H"])
        m.split = pyo.Constraint(m.S, rule=lambda m, s: sum(m.y[s, j] for j in m.J) == m.u_lo[s] + m.u_hi[s])
        m.lo_cap = pyo.Constraint(m.S, rule=lambda m, s: m.u_lo[s] <= SOFT_CAP)
        m.obj = pyo.Objective(rule=lambda m: (
            sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.p[s] * (sum(m.beta_s[s, j] * m.y[s, j] for j in m.J)
                            - PEN1 * m.u_lo[s] - PEN2 * m.u_hi[s]) for s in m.S)),
            sense=pyo.maximize)
        return m

    def build_det(beta_scn, fix_x=None):
        m = pyo.ConcreteModel()
        m.J = pyo.Set(initialize=JS)
        m.beta = pyo.Param(m.J, initialize=BETA_B)
        m.beta_s = pyo.Param(m.J, initialize=beta_scn)
        m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.y = pyo.Var(m.J, within=pyo.NonNegativeReals)
        m.u_lo = pyo.Var(within=pyo.NonNegativeReals)
        m.u_hi = pyo.Var(within=pyo.NonNegativeReals)
        m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= BUDGET1)
        if fix_x is not None:
            m.fixx = pyo.ConstraintList()
            for j in JS:
                m.fixx.add(m.x[j] == float(fix_x[j]))
        m.budget2 = pyo.Constraint(expr=sum(m.y[j] for j in m.J) <= RESERVE)
        m.couple = pyo.Constraint(expr=m.y["AI"] <= COUPLE * m.x["H"])
        m.split = pyo.Constraint(expr=sum(m.y[j] for j in m.J) == m.u_lo + m.u_hi)
        m.lo_cap = pyo.Constraint(expr=m.u_lo <= SOFT_CAP)
        m.obj = pyo.Objective(rule=lambda m: (
            sum(m.beta[j] * m.x[j] for j in m.J)
            + sum(m.beta_s[j] * m.y[j] for j in m.J) - PEN1 * m.u_lo - PEN2 * m.u_hi),
            sense=pyo.maximize)
        return m

    def solve(m):
        SOLVER.solve(m)
        return pyo.value(m.obj)

    def get_x(m):
        return {j: round(pyo.value(m.x[j]), 2) for j in JS}

    @st.cache_data
    def run_all():
        m_sp = build_sp()
        Z_SP = solve(m_sp)
        x_SP = get_x(m_sp)
        y_tab = pd.DataFrame({s: {j: round(pyo.value(m_sp.y[s, j]), 1) for j in JS}
                              for s in SS}).T
        det_x, det_Z = {}, {}
        for s in SS:
            md = build_det({j: BETA_S[(s, j)] for j in JS})
            det_Z[s] = solve(md); det_x[s] = get_x(md)
        beta_bar = {j: sum(PROB[s] * BETA_S[(s, j)] for s in SS) for j in JS}
        m_ev = build_det(beta_bar)
        Z_EV = solve(m_ev); x_EV = get_x(m_ev)
        EEV = solve(build_sp(fix_x=x_EV))
        WS = sum(PROB[s] * det_Z[s] for s in SS)
        return dict(Z_SP=Z_SP, x_SP=x_SP, y_tab=y_tab, det_x=det_x, det_Z=det_Z,
                    x_EV=x_EV, EEV=EEV, WS=WS, RP=Z_SP, VSS=Z_SP - EEV, EVPI=WS - Z_SP)

    R = run_all()

    st.subheader("Câu 10.5.1 — Lời giải Stochastic Programming (SP)")
    st.metric("Giá trị kỳ vọng RP", f"{R['RP']:,.1f} tỷ VND")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Quyết định first-stage x\\* (tỷ VND):**")
        st.dataframe(pd.Series(R["x_SP"], name="x*_SP").to_frame(), use_container_width=True)
    with c2:
        st.markdown("**Second-stage yˢ theo kịch bản:**")
        yt = R["y_tab"].copy()
        yt.index = [SCN_NAME[s] for s in yt.index]
        st.dataframe(yt, use_container_width=True)

    st.subheader("Câu 10.5.2 — Deterministic từng kịch bản & so sánh EV vs SP")
    cmp = pd.DataFrame({"x*_SP": R["x_SP"], "x_EV": R["x_EV"]})
    cmp["Chênh lệch"] = cmp["x*_SP"] - cmp["x_EV"]
    st.dataframe(cmp, use_container_width=True)
    figd = px.bar(pd.DataFrame(R["det_x"]).T.assign(
        scn=[SCN_NAME[s] for s in SS]).set_index("scn"),
        barmode="group", title="First-stage tối ưu theo từng kịch bản (giải riêng)")
    st.plotly_chart(figd, use_container_width=True)

    st.subheader("Câu 10.5.3 — VSS & EVPI")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EEV", f"{R['EEV']:,.0f}")
    c2.metric("RP (SP)", f"{R['RP']:,.0f}")
    c3.metric("WS", f"{R['WS']:,.0f}")
    c4.metric("VSS = RP-EEV", f"{R['VSS']:,.1f}")
    c5.metric("EVPI = WS-RP", f"{R['EVPI']:,.1f}")
    fig = px.bar(x=["EEV", "RP (SP)", "WS"], y=[R["EEV"], R["RP"], R["WS"]],
                 color=["EEV", "RP (SP)", "WS"], color_discrete_sequence=SEQ,
                 labels={"x": "", "y": "Lợi ích kỳ vọng (tỷ VND)"},
                 title=f"EEV ≤ RP ≤ WS — VSS={R['VSS']:,.0f}, EVPI={R['EVPI']:,.0f}")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    policy_box(10)