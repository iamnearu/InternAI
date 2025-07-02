import os
import cv2

# -------- CONFIG --------
input_folder = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\testset"
output_folder = input_folder
target_size = (960, 1280)  # (width, height)

# ------------------------
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"🔍 Found {len(image_files)} images to resize.")

for file in image_files:
    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, file)

    image = cv2.imread(input_path)
    if image is None:
        print(f"❌ Failed to read: {input_path}")
        continue

    resized = cv2.resize(image, target_size)
    cv2.imwrite(output_path, resized)
    print(f"✅ Resized and saved: {output_path}")

print("🎉 Done resizing all images.")
