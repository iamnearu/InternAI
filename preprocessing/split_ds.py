import pandas as pd
import os
from sklearn.model_selection import train_test_split

# -------- Paths --------
input_csv = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\cropped_labels.csv"
output_dir = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\splits"
os.makedirs(output_dir, exist_ok=True)

# -------- Load and group --------
df = pd.read_csv(input_csv)
df["group"] = df["img"].apply(lambda x: x.split("_")[0])  # group by base image

unique_groups = df["group"].unique()
print(f"📦 Total ID groups: {len(unique_groups)}")

# -------- Stratified split by group (no leakage) --------
train_ids, temp_ids = train_test_split(unique_groups, test_size=0.30, random_state=42)
val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

def assign_split(group):
    if group in train_ids:
        return "train"
    elif group in val_ids:
        return "val"
    elif group in test_ids:
        return "test"

df["split"] = df["group"].apply(assign_split)

# -------- Save split CSVs --------
for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split]
    out_path = os.path.join(output_dir, f"{split}_labels.csv")
    split_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(split_df)} rows to: {out_path}")
