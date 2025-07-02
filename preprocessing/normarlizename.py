import pandas as pd

# Đọc file có header sẵn
df = pd.read_csv(r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\gt_2s.csv")

# Trích tên ảnh từ đường dẫn
df["image_name"] = df["img_path"].apply(lambda x: x.split("/")[-1])

# Chọn và sắp xếp lại các cột
df_cleaned = df[["image_name", "ID", "Name", "DOB"]]

# Lưu ra file mới
df_cleaned.to_csv(r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\label_cleaned.csv", index=False)

print("✅ Xử lý hoàn tất, đã lưu file label_cleaned.csv")
