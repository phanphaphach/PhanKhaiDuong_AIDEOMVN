# 🇻🇳 AIDEOM-VN — Mô hình Ra Quyết định Phát triển Kinh tế Việt Nam trong Kỉ nguyên AI

> **Bộ bài tập thực hành** | Môn: Các Mô hình Ra Quyết định  
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

---

## 🚀 Hướng dẫn chạy trên Google Colab

### Bước 1 — Mở notebook

Nhấn badge **"Open in Colab"** ở đầu notebook, hoặc:

1. Truy cập [Google Colab](https://colab.research.google.com/)
2. Chọn **File → Upload notebook** → tải file `AIDEOM_VN_Models_1.ipynb`

### Bước 2 — Upload dữ liệu CSV

Notebook cần 3 file CSV đặt **cùng thư mục** với notebook (hoặc upload lên Colab):

```
vietnam_macro_2020_2025.csv    # GDP, cơ cấu ngành, FDI, xuất nhập khẩu 2020-2025
vietnam_sectors_2024.csv       # 10 ngành kinh tế: tăng trưởng, AI readiness, rủi ro
vietnam_regions_2024.csv       # 6 vùng KT-XH: GRDP, FDI, chỉ số số hóa, Gini
```

**Cách upload trên Colab:**

```python
# Cách 1: Upload thủ công (sẽ hiện hộp thoại chọn file)
from google.colab import files
uploaded = files.upload()

# Cách 2: Mount Google Drive (nếu file nằm trong Drive)
from google.colab import drive
drive.mount('/content/drive')
# Sau đó copy file vào thư mục làm việc:
# !cp /content/drive/MyDrive/path/to/*.csv .
```

### Bước 3 — Chạy từng cell

- Nhấn **Shift + Enter** để chạy từng cell theo thứ tự
- Hoặc **Runtime → Run all** để chạy toàn bộ (mất khoảng 5–10 phút)
- Các thư viện thiếu sẽ được **tự động cài đặt** (pulp, cvxpy, pymoo, pyomo, gymnasium, stable-baselines3...)

### Lưu ý quan trọng

- **Bài 10** cần solver GLPK, notebook tự cài bằng `apt-get install glpk-utils` (chỉ hoạt động trên Colab/Linux)
- **Bài 11** (DQN) cần `stable-baselines3` và `torch`, Colab đã cài sẵn PyTorch
- **Không cần GPU** cho bất kỳ bài nào (Bài 11 DQN chạy CPU đủ nhanh với env đơn giản)
- Nếu gặp lỗi `FileNotFoundError` khi đọc CSV → kiểm tra lại đã upload đúng 3 file chưa

---

## 📁 Cấu trúc file

```
.
├── AIDEOM_VN_Models_1.ipynb              # Notebook chính (11 bài, 23 cells)
├── README.md                             # File này
├── app.py                                # Dashboard Streamlit (Bài 12)
├── requirements.txt                      # Danh sách thư viện Python
├── vietnam_macro_2020_2025.csv           # Dữ liệu vĩ mô 2020-2025
├── vietnam_regions_2024.csv              # Dữ liệu 6 vùng KT-XH 2024
└── vietnam_sectors_2024.csv              # Dữ liệu 10 ngành 2024
```

---

## 📦 Thư viện sử dụng

Notebook tự cài các thư viện thiếu, nhưng nếu muốn cài trước:

```python
!pip install -q numpy pandas scipy matplotlib seaborn \
    pulp cvxpy pyomo pymoo gymnasium stable-baselines3 plotly
!apt-get -qq install -y glpk-utils   # solver cho Pyomo (Bài 10)
```

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

Notebook sử dụng AI hỗ trợ (Claude) trong quá trình phát triển code và phân tích. Việc sử dụng này được khai báo rõ ràng theo đúng quy định của đề bài (Phụ lục F2, trang 30).
