# 🤖 Dự Án Học Máy: Xây Dựng Và Đánh Giá 5 Mô Hình Học Máy
## Online Shoppers Purchasing Intention Prediction

Dự án triển khai quy trình Học máy (Machine Learning Pipeline) hoàn chỉnh từ cơ bản đến nâng cao trên tập dữ liệu thương mại điện tử **Online Shoppers Purchasing Intention**, nhằm dự đoán xem phiên truy cập của khách hàng có dẫn đến việc mua hàng (`Revenue = True / False`) hay không.

---

## 📌 5 Mô Hình Được Triển Khai
1. **Logistic Regression (Hồi quy Logistic):** Mô hình tuyến tính xác suất (Baseline model).
2. **K-Nearest Neighbors - KNN (K láng giềng gần nhất):** Mô hình học dựa trên khoảng cách Euclidean.
3. **Decision Tree (Cây quyết định):** Cấu trúc cây CART, phân nhánh theo Gini Impurity.
4. **Random Forest (Rừng ngẫu nhiên):** Ensemble Bagging kết hợp 100 cây quyết định, giảm phương sai và chống overfitting.
5. **Support Vector Machine - SVM (Máy vector hỗ trợ):** Tối đa hóa lề phân tách với hàm nhân RBF phi tuyến (Radial Basis Function Kernel).

---

## 🛠️ Quy Trình Thực Hiện (End-to-End Pipeline)

Mỗi mô hình đều được xây dựng qua đầy đủ các bước chuẩn mực:
* **Bước 1: Khám phá & Phân tích dữ liệu (EDA):** Kiểm tra dữ liệu thiếu, xử lý 125 dòng trùng lặp, phân tích phân phối mất cân bằng nhãn (`84.4% False` vs `15.6% True`), vẽ ma trận tương quan giữa các biến số.
* **Bước 2: Tiền xử lý dữ liệu (Data Preprocessing):**
  * One-Hot Encoding cho các biến phân loại (`Month`, `VisitorType`, `Weekend`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`).
  * Phân chia tập Train/Test theo tỷ lệ **80/20** với `stratify=y` để bảo toàn tỷ lệ nhãn.
  * Chuẩn hóa thang đo đặc trưng bằng `StandardScaler` (chỉ `fit_transform` trên Train để tránh Data Leakage).
* **Bước 3: Huấn luyện & Đánh giá từng mô hình:**
  * Giải thích cơ sở lý thuyết, công thức toán và ưu nhược điểm.
  * Thiết lập và giải thích ý nghĩa các siêu tham số (`Hyperparameters`).
  * Dự đoán nhãn lớp và xác suất.
  * Đánh giá chi tiết bằng Ma trận nhầm lẫn (`Confusion Matrix`), `Classification Report`, và đường cong `ROC-AUC`.
  * Trực quan hóa đặc thù: Sơ đồ nhánh cây Decision Tree, biểu đồ Feature Importance của Random Forest.
* **Bước 4: So sánh tổng hợp & Khuyến nghị kinh doanh:**
  * Tổng hợp bảng số liệu so sánh.
  * Vẽ biểu đồ cột nhóm và biểu đồ gộp Combined ROC Curves.

---

## 📊 Kết Quả So Sánh Thực Nghiệm

| Thứ Hạng | Mô Hình (Model) | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Thời Gian Train |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Decision Tree** | **90.21%** | 70.85% | **63.61%** | **67.03%** | 0.9256 | ~0.03s |
| 🥈 | **Random Forest** | **89.92%** | **80.09%** | 47.38% | **59.54%** | **0.9292** | ~0.15s |
| 🥉 | **Logistic Regression** | 88.78% | 75.47% | 41.88% | 53.87% | 0.8981 | ~0.02s |
| 4 | **Support Vector Machine (SVM)** | 88.90% | 77.61% | 40.84% | 53.52% | 0.8781 | ~6.46s |
| 5 | **K-Nearest Neighbors (KNN)** | 85.54% | 62.61% | 18.85% | 28.97% | 0.7165 | ~0.002s |

> **Nhận xét chính:**
> * Do dữ liệu bị mất cân bằng, **F1-Score** và **ROC-AUC** là các chỉ số trung thực nhất để đánh giá mô hình.
> * **Random Forest** và **Decision Tree** đạt hiệu năng vượt trội nhất với ROC-AUC xấp xỉ **0.93**, nhờ khai thác hiệu quả đặc trưng quan trọng nhất là `PageValues`.

---

## 📁 Cấu Trúc Thư Mục

```text
Machine-Learning/
├── online_shoppers_intention.csv     # Tập dữ liệu gốc
├── online_shoppers_ml_5_models.ipynb # File Jupyter Notebook hoàn chỉnh có chú thích và đồ thị
├── train_5_models.py                 # File Python chạy tự động toàn bộ quy trình
├── create_ml_notebook.py             # Script xây dựng và tự động sinh Notebook
├── models_comparison.csv             # Bảng kết quả so sánh dạng CSV
├── best_model_random_forest.joblib   # Model Random Forest đã huấn luyện tối ưu
├── scaler.joblib                     # Bộ chuẩn hóa dữ liệu StandardScaler đã fit
├── output_plots/                     # Thư mục chứa hình ảnh biểu đồ phân tích
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_decision_tree_structure.png
│   ├── 04_rf_feature_importance.png
│   ├── 05_all_confusion_matrices.png
│   ├── 06_combined_roc_curves.png
│   └── 07_models_comparison_bar.png
├── .gitignore
└── README.md
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Cài đặt các thư viện cần thiết
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
```

### 2. Xem và chạy Jupyter Notebook
Mở tệp `online_shoppers_ml_5_models.ipynb` bằng VS Code hoặc Jupyter Lab:
```bash
jupyter notebook online_shoppers_ml_5_models.ipynb
```

### 3. Chạy trực tiếp file Python
```bash
python train_5_models.py
```
Toàn bộ biểu đồ sẽ tự động được lưu vào thư mục `output_plots/`.
