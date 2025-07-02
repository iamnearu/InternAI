import cv2
import os
import json

# -------- CONFIG --------
input_folder = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\testset"
output_json_dir = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\data\box_coords"
os.makedirs(output_json_dir, exist_ok=True)


# -------- Helper --------
def draw_boxes_on_image(image_path, json_output_path):
    boxes = []
    drawing = False
    ix, iy = -1, -1

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image: {image_path}")
        return

    # Resize ảnh để vừa với màn hình (ví dụ: giảm 50% kích thước)
    scale_percent = 50
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img_copy = img_resized.copy()

    # Tạo cửa sổ và giới hạn kích thước
    window_name = os.path.basename(image_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            fx, fy = x, y
            x1, y1 = min(ix, fx), min(iy, fy)
            w, h = abs(fx - ix), abs(fy - iy)
            cv2.rectangle(img_copy, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            boxes.append({"x": x1, "y": y1, "w": w, "h": h})
            print(f"📐 Box: (x={x1}, y={y1}, w={w}, h={h})")

    cv2.setMouseCallback(window_name, draw_rectangle)

    print(f"\n🖱️ Drawing boxes for: {image_path}")
    print("🔑 S to save boxes, N to move to next image, ESC to finish and save")

    while True:
        cv2.imshow(window_name, img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Phím S để lưu
            if boxes:
                scale_factor_x = img.shape[1] / img_resized.shape[1]
                scale_factor_y = img.shape[0] / img_resized.shape[0]
                boxes_original = [
                    {
                        "x": int(box["x"] * scale_factor_x),
                        "y": int(box["y"] * scale_factor_y),
                        "w": int(box["w"] * scale_factor_x),
                        "h": int(box["h"] * scale_factor_y)
                    }
                    for box in boxes
                ]
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(boxes_original, f, indent=4)
                print(f"💾 Saved {len(boxes)} boxes to: {json_output_path}")
                # Làm mới img_copy để tiếp tục vẽ
                img_copy = img_resized.copy()
            else:
                print("⚠️ No boxes to save!")
        elif key == ord('n'):  # Phím N để chuyển ảnh tiếp theo
            break
        elif key == 27:  # ESC để kết thúc và lưu
            if boxes:
                scale_factor_x = img.shape[1] / img_resized.shape[1]
                scale_factor_y = img.shape[0] / img_resized.shape[0]
                boxes_original = [
                    {
                        "x": int(box["x"] * scale_factor_x),
                        "y": int(box["y"] * scale_factor_y),
                        "w": int(box["w"] * scale_factor_x),
                        "h": int(box["h"] * scale_factor_y)
                    }
                    for box in boxes
                ]
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(boxes_original, f, indent=4)
                print(f"✅ Saved {len(boxes)} boxes to: {json_output_path}")
            else:
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(boxes, f, indent=4)
                print(f"✅ No boxes saved to: {json_output_path}")
            break

    cv2.destroyWindow(window_name)


# -------- Loop All Images --------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
current_image = 0

while current_image < len(image_files):
    img_name = image_files[current_image]
    image_path = os.path.join(input_folder, img_name)
    json_path = os.path.join(output_json_dir, f"{os.path.splitext(img_name)[0]}.json")

    draw_boxes_on_image(image_path, json_path)
    current_image += 1  # Chuyển sang ảnh tiếp theo khi nhấn N hoặc ESC