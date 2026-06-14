# 🇻🇳 AIDEOM-VN — Mô hình Ra Quyết định Phát triển Kinh tế Việt Nam trong Kỉ nguyên AI

> **Bộ bài tập thực hành** | Môn: Các Mô hình Ra Quyết định
> Sinh viên thực hiện: Phan Khải Dương - MSV: 23051209  
> Trường Đại học Kinh tế — Viện Quản trị Kinh doanh  
> Dữ liệu thực tế Việt Nam 2020–2025

---

## 📋 Mô tả dự án

Notebook này triển khai **11/12 bài tập** của bộ đề AIDEOM-VN, bao phủ toàn bộ 4 cấp độ từ **Dễ → Khó**, sử dụng dữ liệu kinh tế thực tế của Việt Nam giai đoạn 2020–2025. Mỗi bài bao gồm:

- **Code Python** giải trọn vẹn các câu hỏi lập trình
- **Biểu đồ trực quan hóa** (matplotlib, plotly, seaborn)
- **Câu hỏi thảo luận chính sách** với phân tích liên hệ thực tiễn Việt Nam

| Cấp độ | Bài | Chủ đề | Công cụ chính |
|--------|-----|--------|---------------|
| **Dễ** | 1 | Hàm sản xuất Cobb-Douglas mở rộng (AI + Số hóa) | numpy, pandas, matplotlib |
| | 2 | Phân bổ ngân sách LP đơn giản (4 hạng mục) | scipy.optimize, PuLP |
| | 3 | Chỉ số ưu tiên ngành (Weighted Scoring, 10 ngành) | pandas, numpy |
| **Trung bình** | 4 | LP phân bổ ngân sách số theo vùng-ngành (24 biến) | PuLP, CVXPY |
| | 5 | MIP lựa chọn dự án chuyển đổi số (15 dự án) | PuLP (CBC) |
| | 6 | TOPSIS xếp hạng 6 vùng kinh tế + Entropy + AHP | numpy |
| **Khá khó** | 7 | Tối ưu đa mục tiêu Pareto — NSGA-II (4 objectives) | pymoo |
| | 8 | Tối ưu động liên thời gian 2026–2035 | CVXPY (geo_mean) |
| | 9 | Tác động AI tới thị trường lao động (8 ngành) | CVXPY, plotly (Sankey) |
| **Khó** | 10 | Quy hoạch ngẫu nhiên hai giai đoạn (Stochastic LP) | Pyomo, GLPK |
| | 11 | Học tăng cường Q-learning + DQN | gymnasium, stable-baselines3 |
| *(chưa có)* | 12 | Đồ án tích hợp AIDEOM-VN (6 module + Dashboard) | Streamlit/Dash |

## 📁 Cấu trúc file

```
.
├── data                                  # Chứa 3 file .csv
├── README.md                             # File này
├── dashboard.py                          # Dashboard Streamlit (Bài 12)
├── requirements.txt                      # Danh sách thư viện Python
├── modules                               # Chứa cách giải của 12 bài trong dashboard
├── winglpk-4.65                          # Solver GLPK cho bài 10
├── src                                   # Chứa các file dùng chung của dashboard
├── notebooks                             # Chứa các model huấn luyện của 11 bài toán
└── report                                # Chứa báo cáo bài tập cuối kì dạng .pdf của học phần
```

---

## 📦 Thư viện sử dụng

Cài các thư viện cần sử dụng, đã được thiết lập trong file requirements.txt:
```python
pip install -r requirements.txt
pip install torch torchvision torchaudio        #PyTorch với máy không có GPU rời
```
Kiểm tra bằng terminal:
```PowerShell
python -c "import numpy, pandas, scipy, pulp, cvxpy, pymoo, pyomo; print('OK')"
python -c "import gymnasium, stable_baselines3, torch; print('RL OK')"
python -c "import streamlit; print('Dashboard OK')"
```
Solver GLPK (Windows 64-bit):
Folder chứa solver đã được tích hợp sẵn trong folder code. Sau khi tải về, hãy cài đặt như sau:
**Thêm GLPK vào biến môi trường (Environment Variables):**
> Mở Start Menu, gõ Environment Variables và chọn Edit the system environment variables.
> Trong cửa sổ System Properties, nhấn nút Environment Variables....
> Ở phần System variables, tìm biến Path và nhấn Edit....
> Nhấn New và thêm đường dẫn đến thư mục chứa file thực thi (chọn thư mục 64-bit). Ví dụ: C:\winglpk-4.65\glpk-4.65\w64.
> Nhấn OK để lưu tất cả các cửa sổ.

**Kiểm tra cài đặt:**
```PowerShell
glpsol --version
```
> Nếu màn hình hiển thị thông tin phiên bản của GLPK thì bạn đã cài đặt và cấu hình PATH thành công.

**Tổng hợp thư viện cần dùng:**

| Thư viện | Phiên bản khuyến nghị | Dùng trong bài |
|----------|----------------------|----------------|
| numpy, pandas | ≥ 1.24 / ≥ 2.0 | Tất cả |
| scipy | ≥ 1.10 | Bài 2 |
| matplotlib, seaborn | ≥ 3.7 / ≥ 0.12 | Tất cả |
| PuLP | ≥ 2.7 | Bài 2, 4, 5 |
| CVXPY | ≥ 1.4 | Bài 4, 8, 9 |
| Pyomo + GLPK | ≥ 6.6 | Bài 10 |
| pymoo | ≥ 0.6.1 | Bài 7 |
| gymnasium | ≥ 0.29 | Bài 11 |
| stable-baselines3 | ≥ 2.1 | Bài 11 (DQN) |
| plotly | ≥ 5.17 | Bài 9 (Sankey) |

---

## 📝 Ghi chú kỹ thuật

### Bài 4 — Ràng buộc công bằng vùng miền (C5)

Mô hình gốc với λ = 0.70 là **bất khả thi** do chênh lệch quá lớn giữa vùng SE (D₀=82) và CH (D₀=32). Notebook đã chẩn đoán chính xác vấn đề này và điều chỉnh λ = 0.68 để mô hình có nghiệm. Chi tiết giải thích toán học nằm trong cell code Bài 4.

### Bài 1 — Dữ liệu bổ sung

Một số chuỗi dữ liệu (K, L, AI, H) không có trong file CSV mà được lấy trực tiếp từ Bảng 1.3 của đề bài, vì file `vietnam_macro_2020_2025.csv` không chứa các cột này. Đây là thiết kế có chủ đích của đề bài.

### Bài 3 — Tiêu chí "Năng suất"

Đề bài sử dụng "Năng suất (tr.VND/LĐ)" trong bảng dữ liệu nhưng gợi ý code dùng `gdp_share_2024_pct` làm proxy cho productivity/quy mô ngành. Notebook tuân theo gợi ý code của đề.

---

## 📚 Nguồn dữ liệu

- **Tổng cục Thống kê (GSO/NSO)** — GDP, cơ cấu ngành, lao động, xuất nhập khẩu
- **Bộ Khoa học và Công nghệ (MoST)** — Doanh nghiệp công nghệ số, R&D
- **Bộ Thông tin và Truyền thông (MIC)** — Chỉ số kinh tế số, AI readiness
- **World Bank** — Vietnam Digital Economy Report 2024
- **WIPO** — Global Innovation Index 2025

*Lưu ý: Các con số trong file CSV được làm tròn để thuận tiện cho giảng dạy và lập trình. Khi viết luận văn/bài báo, cần truy xuất số liệu gốc từ các nguồn chính thức.*

---

## ⚖️ Liêm chính học thuật

Notebook sử dụng AI hỗ trợ (Anthropic Claude Code, Google Gemini) trong quá trình phát triển code và phân tích. Việc sử dụng này được khai báo rõ ràng theo đúng quy định của đề bài (Phụ lục F2, trang 30).
