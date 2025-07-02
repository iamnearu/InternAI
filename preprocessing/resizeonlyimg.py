import cv2

# -------- Configuration --------
input_path = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\sample_images\img1342.jpg"
output_path = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\sample_images\img1342.jpg"
resize_to = (2000, 1000)  # (width, height)

# -------- Load and Resize --------
image = cv2.imread(input_path)
if image is None:
    print(f"❌ Failed to read image: {input_path}")
else:
    resized = cv2.resize(image, resize_to)
    cv2.imwrite(output_path, resized)
    print(f"✅ Resized image saved to: {output_path}")
