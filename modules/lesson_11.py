import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.constants import SEQ
from src.components import header, policy_box

def render():
    header("🔹 Bài 11 — Q-learning cho chính sách kinh tế thích nghi",
           "Mô hình hóa nền kinh tế VN như MDP 81 trạng thái × 5 hành động; huấn luyện "
           "π* qua Q-learning tabular và so sánh với chính sách rule-based.")
    st.caption("Lưu ý: AI hỗ trợ ra quyết định, KHÔNG thay thế trách nhiệm chính trị-xã hội.")

    ALLOC = {
        0: np.array([0.70, 0.10, 0.10, 0.10]),
        1: np.array([0.40, 0.25, 0.15, 0.20]),
        2: np.array([0.25, 0.45, 0.15, 0.15]),
        3: np.array([0.20, 0.20, 0.45, 0.15]),
        4: np.array([0.30, 0.20, 0.10, 0.40]),
    }
    W = np.array([0.40, 0.25, 0.20, 0.15])
    BINS = {"growth": [0.04, 0.07], "D": [22.0, 28.0], "AI": [110.0, 160.0], "U": [0.30, 0.45]}
    action_names = {0: "a0 Truyền thống (70%K)", 1: "a1 Cân bằng", 2: "a2 Số hóa nhanh (45%D)",
                    3: "a3 AI dẫn dắt (45%AI)", 4: "a4 Bao trùm (40%H)"}
    level = {0: "low", 1: "medium", 2: "high"}

    def _bin(v, edges):
        return 0 if v < edges[0] else (1 if v < edges[1] else 2)

    class Env:
        T = 10
        def reset(self, init_state=None):
            self.K, self.D, self.AI, self.H = 27500.0, 20.3, 86.0, 30.0
            self.U, self.last_growth = 0.38, 0.06
            self.last_Y = self._prod()
            self.t = 0
            self.state = (np.array(init_state, dtype=int) if init_state is not None
                          else self._disc())
            return self.state
        def _prod(self):
            return (self.K ** 0.33) * (54.0 ** 0.42) * (self.D ** 0.10) * \
                   (self.AI ** 0.08) * (self.H ** 0.07)
        def _disc(self):
            return np.array([_bin(self.last_growth, BINS["growth"]), _bin(self.D, BINS["D"]),
                             _bin(self.AI, BINS["AI"]), _bin(self.U, BINS["U"])], dtype=int)
        def step(self, action):
            a = ALLOC[action]; budget = 1000.0
            self.K += a[0] * budget
            self.D = (1 - 0.12) * self.D + a[1] * budget / 100.0
            self.AI = (1 - 0.15) * self.AI + a[2] * budget / 20.0
            self.H += a[3] * budget / 200.0
            Y = self._prod()
            growth = (Y - self.last_Y) / self.last_Y
            dU = 0.06 * a[2] - 0.10 * a[3]
            self.U = float(np.clip(self.U + dU, 0.05, 0.95))
            reward = W[0] * growth - W[1] * dU - W[2] * (a[2] - 0.5 * a[3]) - W[3] * a[0]
            self.last_growth, self.last_Y = growth, Y
            self.t += 1
            self.state = self._disc()
            return self.state, float(reward), self.t >= self.T

    @st.cache_data(show_spinner="Đang huấn luyện Q-learning (10,000 episodes)...")
    def train_q(alpha=0.1, gamma=0.95, n_eps=10000):
        rng = np.random.default_rng(42)
        env = Env()
        Q = np.zeros((3, 3, 3, 3, 5))
        hist = []
        for ep in range(n_eps):
            s = env.reset()
            eps = max(0.05, 1.0 - ep / 5000.0)
            tot = 0.0
            while True:
                a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
                s2, r, done = env.step(a)
                Q[tuple(s) + (a,)] += alpha * (r + gamma * Q[tuple(s2)].max() - Q[tuple(s) + (a,)])
                s = s2; tot += r
                if done:
                    break
            hist.append(tot)
        return Q, np.array(hist)

    def run_policy(Q, policy, n=2000):
        rng = np.random.default_rng(0)
        env = Env()
        out = []
        for _ in range(n):
            s = env.reset(); tot = 0.0
            while True:
                if policy == "pi":
                    a = int(np.argmax(Q[tuple(s)]))
                elif policy == "random":
                    a = int(rng.integers(5))
                else:
                    a = policy
                s, r, done = env.step(a)
                tot += r
                if done:
                    break
            out.append(tot)
        return np.array(out)

    Q, hist = train_q()

    st.subheader("Câu 11.3.1-2 — Môi trường & huấn luyện Q-learning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Không gian trạng thái", "3⁴ = 81")
    c2.metric("Reward 100 ep đầu", f"{hist[:100].mean():.4f}")
    c3.metric("Reward 100 ep cuối", f"{hist[-100:].mean():.4f}")
    window = 100
    ma = np.convolve(hist, np.ones(window) / window, mode="valid")
    fig = px.line(x=np.arange(len(ma)), y=ma, labels={"x": "Episode", "y": "Tổng reward"},
                  title="Learning curve Q-learning (moving avg 100 ep)")
    fig.update_traces(line_color="#1f77b4")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Câu 11.3.3 — Chính sách tối ưu π*(s) tại các trạng thái")
    test_states = {
        "VN 2026 thực tế (med, med, low, med)": (1, 1, 0, 1),
        "Suy thoái (low, low, low, high U)": (0, 0, 0, 2),
        "Bứt phá (high, high, high, low U)": (2, 2, 2, 0),
        "Số hóa cao nhưng thất nghiệp": (1, 2, 1, 2),
        "Tăng trưởng nóng thiếu AI": (2, 0, 0, 1),
    }
    rows = []
    for name, s in test_states.items():
        qv = Q[s]
        best = int(np.argmax(qv))
        rows.append({"Trạng thái": name,
                     "[g, D, AI, U]": f"[{level[s[0]]}, {level[s[1]]}, {level[s[2]]}, {level[s[3]]}]",
                     "π*(s)": action_names[best],
                     "Q max": round(float(qv[best]), 3)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Câu 11.3.4 — So sánh π* với chính sách rule-based")
    pi_star = run_policy(Q, "pi")
    a1r = run_policy(Q, 1)
    a3r = run_policy(Q, 3)
    randr = run_policy(Q, "random")
    cmp = pd.DataFrame({
        "Chính sách": ["π* (Q-learning)", "a1 Cân bằng", "a3 AI dẫn dắt", "Random"],
        "Reward TB": [pi_star.mean(), a1r.mean(), a3r.mean(), randr.mean()],
        "Độ lệch chuẩn": [pi_star.std(), a1r.std(), a3r.std(), randr.std()],
    })
    fig = go.Figure()
    for name, data, c in [("π* (Q-learn)", pi_star, "#2ca02c"), ("a1 Cân bằng", a1r, "#ff7f0e"),
                          ("a3 AI dẫn dắt", a3r, "#d62728"), ("Random", randr, "#7f7f7f")]:
        fig.add_box(y=data, name=name, marker_color=c)
    fig.update_layout(title="Phân phối phần thưởng tích lũy (2000 episodes)",
                      yaxis_title="Tổng reward / episode", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cmp.round(4), use_container_width=True, hide_index=True)

    policy_box(11)