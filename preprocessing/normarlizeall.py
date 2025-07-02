import pandas as pd
import os
from datetime import datetime

# ==== 1. Đọc file CSV gốc ====
df = pd.read_csv(r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\label_cleaned.csv")

# ==== 2. Đổi tên cột về chuẩn ====
df = df.rename(columns={
    "image_name": "img",
    "ID": "id_number",
    "Name": "full_name",
    "DOB": "dob"
})

# ==== 3. Chuẩn hóa ngày sinh ====
def normalize_date(date_str):
    try:
        return pd.to_datetime(date_str, dayfirst=True).strftime("%Y-%m-%d")
    except Exception:
        return None

df["dob"] = df["dob"].apply(normalize_date)

# ==== 4. Kiểm tra id_number hợp lệ ====
df["id_number"] = df["id_number"].astype(str).str.strip()
valid_mask = df["id_number"].str.match(r"^\d{9,12}$")
invalid_df = df[~valid_mask]
df_cleaned = df[valid_mask]

# ==== 5. Xoá ảnh tương ứng với dòng sai ====
image_folder = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\sample_images"  # <-- Thư mục chứa ảnh
deleted_count = 0

for img_name in invalid_df["img"]:
    img_path = os.path.join(image_folder, img_name)
    if os.path.exists(img_path):
        os.remove(img_path)
        deleted_count += 1

# ==== 6. Lưu kết quả ====
df_cleaned.to_csv("label_cleaned.csv", index=False)
invalid_df.to_csv("invalid_ids_removed.csv", index=False)

# ==== 7. Thông báo ====
print("✅ Đã loại dòng sai định dạng và ảnh tương ứng")
print(f"🗑️ Số dòng bị xóa: {len(invalid_df)}")
print(f"🗑️ Số ảnh đã xóa: {deleted_count}")
