import os
import cv2
import albumentations as A
import pandas as pd
from tqdm import tqdm

image_dir = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\sample_images"
csv_path = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\label_cleaned.csv"
augmented_dir = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\augmented_images"
os.makedirs(augmented_dir, exist_ok=True)

df = pd.read_csv(csv_path)

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.RandomShadow(p=0.3),
    A.Perspective(scale=(0.02, 0.05), p=0.3)
])

augmented_records = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_name = row['img']
    img_path = os.path.join(image_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Skipping missing image: {img_path}")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {img_path}")
        continue

    base_name = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]

    # Save original image with _orig suffix
    orig_out = f"{base_name}_orig{ext}"
    orig_path = os.path.join(augmented_dir, orig_out)
    cv2.imwrite(orig_path, image)
    if os.path.exists(orig_path):
        augmented_records.append({
            "img": orig_out,
            "id_number": row["id_number"],
            "full_name": row["full_name"],
            "dob": row["dob"]
        })
    else:
        print(f"Failed to save original image: {orig_out}")

    # Generate and save 5 augmented images
    for i in range(5):
        aug_img = augment(image=image)["image"]
        aug_name = f"{base_name}_aug{i+1}{ext}"
        aug_path = os.path.join(augmented_dir, aug_name)
        cv2.imwrite(aug_path, aug_img)
        if os.path.exists(aug_path):
            augmented_records.append({
                "img": aug_name,
                "id_number": row["id_number"],
                "full_name": row["full_name"],
                "dob": row["dob"]
            })
        else:
            print(f"Failed to save augmented image: {aug_name}")

# Save the new label file
pd.DataFrame(augmented_records).to_csv(os.path.join(augmented_dir, "augmented_labels.csv"), index=False)
print("Augmentation complete. Saved augmented_labels.csv")