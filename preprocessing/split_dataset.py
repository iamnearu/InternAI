"""
File: split_dataset.py
Mục đích:
- Chia file donut_labels.csv thành 2 file:
    + donut_labels_train.csv (80%)
    + donut_labels_val.csv   (20%)
- Phục vụ huấn luyện mô hình Donut với Early Stopping.

Yêu cầu:
- File đầu vào: donut_labels.csv (phải có 2 cột: image, target)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Đường dẫn đến file dữ liệu gốc
INPUT_CSV = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels.csv"

# Đường dẫn file output
TRAIN_CSV = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels_train.csv"
VAL_CSV   = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\donut_labels_val.csv"

# Đọc dữ liệu gốc
df = pd.read_csv(INPUT_CSV)

# Chia 80% train, 20% val (có xáo trộn dữ liệu, cố định seed để tái lập)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Ghi ra file mới
train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

print(f"✅ Đã chia dữ liệu: {len(train_df)} train | {len(val_df)} val")
print(f"→ Train saved to: {TRAIN_CSV}")
print(f"→ Val   saved to: {VAL_CSV}")
