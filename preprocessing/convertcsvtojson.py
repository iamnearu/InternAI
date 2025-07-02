import pandas as pd
import os
import json

csv_path = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\label_cleaned.csv'
output_dir = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\labels'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    label = {
        "so_cccd": str(row['id_number']),
        "ho_va_ten": row['full_name'],
        "ngay_sinh": row['dob'],
    }

    json_name = os.path.splitext(row['img'])[0] + ".json"
    json_path = os.path.join(output_dir, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(label, f, ensure_ascii=False, indent=4)

print(f"✅ Converted {len(df)} rows to JSON.")
