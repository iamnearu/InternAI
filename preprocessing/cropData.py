import os
import cv2
import pandas as pd

# Paths
input_csv = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\augmented_labels.csv'
image_dir = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\augmented_images'
output_dir = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\cropped_images'
output_csv = r'C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\cropped_labels.csv'

os.makedirs(output_dir, exist_ok=True)

# Crop coordinates (adjust based on your CCCD layout)
CROPS = {
    'id_number': (50, 120, 500, 40),
    'full_name': (50, 180, 500, 40),
    'dob': (50, 240, 300, 40),
}

# Load dataset
df = pd.read_csv(input_csv)
cropped_data = []

for idx, row in df.iterrows():
    image_path = os.path.join(image_dir, row['img'])
    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Cannot read image: {image_path}")
        continue

    for field, (x, y, w, h) in CROPS.items():
        # Clip to image boundaries
        H, W = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)

        cropped = image[y1:y2, x1:x2]
        if cropped is None or cropped.size == 0:
            print(f"⚠️ Skipped empty crop: {row['img']} - {field}")
            continue

        # Save cropped image
        cropped_name = f"{row['img'][:-4]}_{field}.jpg"
        cropped_path = os.path.join(output_dir, cropped_name)
        cv2.imwrite(cropped_path, cropped)

        # Add row to output CSV
        cropped_data.append({
            'img': cropped_name,
            'field': field,
            'text': str(row[field]).strip()
        })

# Save new CSV file
pd.DataFrame(cropped_data).to_csv(output_csv, index=False, encoding='utf-8')
print("✅ All crops saved to:", output_dir)
print("📝 Cropped labels saved to:", output_csv)
