# -*- coding: utf-8 -*-
"""
DỰ ÁN HỌC MÁY: XÂY DỰNG VÀ ĐÁNH GIÁ 5 MÔ HÌNH HỌC MÁY
Dự đoán ý định mua hàng của khách truy cập (Online Shoppers Purchasing Intention)

Quy trình chuẩn bao gồm:
1. Đọc và khám phá dữ liệu (EDA)
2. Tiền xử lý dữ liệu (Xử lý trùng lặp, One-Hot Encoding, Train/Test Split, Feature Scaling)
3. Huấn luyện và đánh giá 5 mô hình máy học:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Support Vector Machine (SVM)
4. So sánh tổng hợp hiệu năng và lưu kết quả/biểu đồ
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Cấu hình UTF-8 cho console Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Tạo thư mục lưu biểu đồ
OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def main():
    print("=" * 75)
    print(" BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN VÀ ĐÁNH GIÁ 5 MÔ HÌNH HỌC MÁY ")
    print("=" * 75)

    # -------------------------------------------------------------
    # BƯỚC 1: ĐỌC VÀ KHÁM PHÁ DỮ LIỆU (EDA)
    # -------------------------------------------------------------
    csv_file = 'online_shoppers_intention.csv'
    if not os.path.exists(csv_file):
        print(f"Lỗi: Không tìm thấy file dữ liệu '{csv_file}'!")
        return

    print(f"\n[BƯỚC 1] Đọc dữ liệu từ '{csv_file}'...")
    df = pd.read_csv(csv_file)
    print(f"- Kích thước ban đầu: {df.shape[0]} dòng, {df.shape[1]} cột.")

    # Kiểm tra và xử lý trùng lặp
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"- Đã phát hiện và loại bỏ {n_duplicates} dòng trùng lặp. Kích thước mới: {df.shape}")

    # Kiểm tra Missing values
    n_null = df.isnull().sum().sum()
    print(f"- Tổng số giá trị thiếu (Null/NaN): {n_null}")

    # Phân tích biến mục tiêu Revenue
    revenue_counts = df['Revenue'].value_counts()
    print(f"- Phân phối biến mục tiêu 'Revenue':")
    print(f"  + False (Không mua): {revenue_counts[False]} ({revenue_counts[False]/len(df)*100:.2f}%)")
    print(f"  + True  (Có mua):    {revenue_counts[True]} ({revenue_counts[True]/len(df)*100:.2f}%)")

    # Lưu biểu đồ phân phối biến mục tiêu
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.countplot(x='Revenue', data=df, ax=axes[0], palette=['#3498DB', '#2ECC71'])
    axes[0].set_title('Số lượng phiên theo Revenue', fontweight='bold')
    axes[1].pie(revenue_counts, labels=['False (Không mua)', 'True (Có mua)'],
                autopct='%1.1f%%', colors=['#3498DB', '#2ECC71'], startangle=140, explode=[0, 0.1])
    axes[1].set_title('Tỷ lệ ý định mua hàng', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_target_distribution.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/01_target_distribution.png")

    # Lưu biểu đồ tương quan
    num_cols = [
        'Administrative', 'Administrative_Duration', 
        'Informational', 'Informational_Duration', 
        'ProductRelated', 'ProductRelated_Duration', 
        'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
    ]
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='Blues', linewidths=0.5)
    plt.title('Ma Trận Tương Quan Các Biến Số (Correlation Heatmap)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_correlation_heatmap.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/02_correlation_heatmap.png")

    # -------------------------------------------------------------
    # BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU & PHÂN CHIA TẬP TRAIN/TEST
    # -------------------------------------------------------------
    print("\n[BƯỚC 2] Tiền xử lý dữ liệu & Chuẩn hóa...")
    X = df.drop(columns=['Revenue'])
    y = df['Revenue'].astype(int)

    categorical_features = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True, dtype=int)
    print(f"- Số đặc trưng sau One-Hot Encoding: {X_encoded.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"- Tập Train: {X_train.shape[0]} mẫu | Tập Test: {X_test.shape[0]} mẫu (Tỷ lệ 80/20)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("- Đã chuẩn hóa dữ liệu bằng StandardScaler (fit trên Train, transform trên Test).")

    # -------------------------------------------------------------
    # BƯỚC 3: XÂY DỰNG & HUẤN LUYỆN 5 MÔ HÌNH HỌC MÁY
    # -------------------------------------------------------------
    print("\n[BƯỚC 3] Khởi tạo và huấn luyện 5 mô hình học máy...")

    models = {
        'Logistic Regression': {
            'instance': LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42),
            'use_scaled': True,
            'desc': 'Mô hình hồi quy Logistic tuyến tính'
        },
        'K-Nearest Neighbors': {
            'instance': KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='minkowski', p=2),
            'use_scaled': True,
            'desc': 'Mô hình học dựa trên khoảng cách Euclidean với K=7'
        },
        'Decision Tree': {
            'instance': DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42),
            'use_scaled': False,
            'desc': 'Cây quyết định với giới hạn độ sâu max_depth=5'
        },
        'Random Forest': {
            'instance': RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1),
            'use_scaled': False,
            'desc': 'Rừng ngẫu nhiên Ensemble 100 cây'
        },
        'Support Vector Machine': {
            'instance': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42),
            'use_scaled': True,
            'desc': 'Máy vector hỗ trợ với RBF Kernel phi tuyến'
        }
    }

    results = {}

    for name, config in models.items():
        print(f"\n>>> Đang huấn luyện: {name} ({config['desc']})...")
        clf = config['instance']
        
        # Chọn dữ liệu phù hợp (Scaled cho LR, KNN, SVM; Raw cho Trees)
        X_tr = X_train_scaled if config['use_scaled'] else X_train
        X_te = X_test_scaled if config['use_scaled'] else X_test

        start_t = time.time()
        clf.fit(X_tr, y_train)
        elapsed_t = time.time() - start_t

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0

        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'Time (s)': elapsed_t,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'model': clf
        }

        print(f"  + Hoàn thành trong {elapsed_t:.4f}s")
        print(f"  + Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | AUC: {auc:.4f}")

    # -------------------------------------------------------------
    # BƯỚC 4: VẼ BIỂU ĐỒ ĐẶC THÙ CỦA TỪNG MÔ HÌNH
    # -------------------------------------------------------------
    print("\n[BƯỚC 4] Vẽ biểu đồ phân tích chuyên sâu cho từng mô hình...")

    # Cây quyết định
    plt.figure(figsize=(18, 8))
    plot_tree(results['Decision Tree']['model'], max_depth=2, feature_names=X_encoded.columns.tolist(),
              class_names=['Không mua', 'Có mua'], filled=True, rounded=True, fontsize=10)
    plt.title('Cấu Trúc 2 Tầng Của Cây Quyết Định (Decision Tree)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_decision_tree_structure.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/03_decision_tree_structure.png")

    # Feature Importance của Random Forest
    rf_model = results['Random Forest']['model']
    importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns).nlargest(10).sort_values()
    plt.figure(figsize=(10, 6))
    importances.plot(kind='barh', color='#2ECC71', edgecolor='black')
    plt.title('Top 10 Đặc Trưng Quan Trọng Nhất (Random Forest Feature Importance)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_rf_feature_importance.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/04_rf_feature_importance.png")

    # Ma trận nhầm lẫn tổng hợp cho cả 5 mô hình
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for idx, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
    plt.suptitle('So Sánh Ma Trận Nhầm Lẫn (Confusion Matrix) Của 5 Mô Hình', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_all_confusion_matrices.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/05_all_confusion_matrices.png")

    # Đường cong ROC gộp
    plt.figure(figsize=(10, 7))
    colors = ['#3498DB', '#9B59B6', '#E67E22', '#2ECC71', '#E74C3C']
    for (name, res), color in zip(results.items(), colors):
        if res['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
            plt.plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC = {res['ROC-AUC']:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Phân loại ngẫu nhiên (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR / Recall)')
    plt.title('Đường Cong ROC Gộp Cả 5 Mô Hình Học Máy', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '06_combined_roc_curves.png'), dpi=200)
    plt.close()
    print(f"  -> Đã lưu biểu đồ: {OUTPUT_DIR}/06_combined_roc_curves.png")

    # -------------------------------------------------------------
    # BƯỚC 5: TỔNG HỢP VÀ KẾT LUẬN
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print(" BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG 5 MÔ HÌNH HỌC MÁY ")
    print("=" * 75)

    summary_list = []
    for name, res in results.items():
        summary_list.append({
            'Model': name,
            'Accuracy (%)': round(res['Accuracy'] * 100, 2),
            'Precision (%)': round(res['Precision'] * 100, 2),
            'Recall (%)': round(res['Recall'] * 100, 2),
            'F1-Score (%)': round(res['F1-Score'] * 100, 2),
            'ROC-AUC': round(res['ROC-AUC'], 4),
            'Train Time (s)': round(res['Time (s)'], 4)
        })

    summary_df = pd.DataFrame(summary_list).sort_values(by='F1-Score (%)', ascending=False).reset_index(drop=True)
    print(summary_df.to_string(index=False))

    # Lưu bảng so sánh ra CSV
    summary_csv_path = 'models_comparison.csv'
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n-> Đã lưu bảng kết quả vào: {summary_csv_path}")

    # Vẽ biểu đồ so sánh các chỉ số
    plot_df = summary_df.melt(id_vars='Model', value_vars=['Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)'],
                              var_name='Metric', value_name='Score (%)')
    plt.figure(figsize=(14, 6))
    chart = sns.barplot(x='Model', y='Score (%)', hue='Metric', data=plot_df, palette='Spectral')
    plt.title('So Sánh Các Chỉ Số Hiệu Năng Giữa 5 Mô Hình Máy Học', fontsize=14, fontweight='bold')
    plt.ylabel('Điểm số (%)')
    plt.ylim(0, 105)
    plt.legend(title='Chỉ số', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '07_models_comparison_bar.png'), dpi=200)
    plt.close()
    print(f"-> Đã lưu biểu đồ so sánh: {OUTPUT_DIR}/07_models_comparison_bar.png")

    # Lưu mô hình tốt nhất (Random Forest)
    best_model = results['Random Forest']['model']
    joblib.dump(best_model, 'best_model_random_forest.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("-> Đã lưu mô hình tốt nhất vào 'best_model_random_forest.joblib' và scaler vào 'scaler.joblib'")

    print("\n" + "=" * 75)
    print(" QUY TRÌNH ĐÃ HOÀN TẤT THÀNH CÔNG! ")
    print(" Bạn có thể mở file Jupyter Notebook: 'online_shoppers_ml_5_models.ipynb'")
    print(" hoặc xem các biểu đồ đã được lưu trong thư mục: 'output_plots/'")
    print("=" * 75)

if __name__ == '__main__':
    main()
