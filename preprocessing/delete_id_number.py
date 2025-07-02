import pandas as pd
import os

# Paths
csv_path = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\label_cleaned.csv'
image_folder = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\testset'

# Load CSV
df = pd.read_csv(csv_path)

# Find duplicates
duplicates = df[df.duplicated(subset='id_number', keep='first')]
print(f" Found {len(duplicates)} duplicate rows to delete.")

# Delete associated images
for filename in duplicates['img']:
    img_path = os.path.join(image_folder, filename)
    if os.path.exists(img_path):
        os.remove(img_path)
        print(f" Deleted image: {img_path}")

# Keep only the first occurrence of each ID
df_cleaned = df.drop_duplicates(subset='id_number', keep='first')

# Save cleaned CSV
df_cleaned.to_csv(csv_path, index=False, encoding='utf-8')
print(f" Saved cleaned CSV to: {csv_path}")
