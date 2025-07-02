"""
File: recognizer.py
Thư mục: ocr/

Mục đích:
- Load mô hình Donut đã fine-tune để trích xuất thông tin từ ảnh CCCD.
- Chuyển kết quả dạng thẻ HTML-like (<tag>...</tag>) thành dict JSON Python.
"""

from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image
import torch
import re

# --------- Load mô hình và processor từ thư mục output ----------
model_path = r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\donut_output"  # ⚠️ Đường dẫn model đã fine-tune

# Load model và processor
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = DonutProcessor.from_pretrained(model_path)

# Đưa mô hình lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# --------- Hàm tiền xử lý lỗi định dạng thẻ HTML-like ----------
def fix_unclosed_tags(text):
    """
    Tự động sửa các thẻ mở chưa được đóng.
    Ví dụ: <so_cccd>123456789 → <so_cccd>123456789</so_cccd>
    """
    opens = re.findall(r"<([a-zA-Z0-9_]+)>", text)
    closes = re.findall(r"</([a-zA-Z0-9_]+)>", text)
    fixed_text = text

    for tag in opens:
        if opens.count(tag) > closes.count(tag):
            fixed_text += f"</{tag}>"  # tự động thêm thẻ đóng vào cuối
    return fixed_text


# --------- Hàm chính: xử lý ảnh → dict JSON ----------
def recognize_with_donut_json(image_path):
    """
    Thực hiện inference ảnh CCCD bằng mô hình Donut → trả về JSON dict.

    Tham số:
        image_path (str): Đường dẫn ảnh CCCD

    Trả về:
        dict: {tag: value}, ví dụ:
            {
                "so_cccd": "123456789",
                "ho_va_ten": "NGUYEN VAN A",
                "ngay_sinh": "01/01/2000"
            }
    """
    # Mở ảnh và convert sang RGB
    image = Image.open(image_path).convert("RGB")

    # Tạo tensor ảnh
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Tạo token đầu vào cho decoder (prompt nếu có)
    decoder_input_ids = processor.tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    # Sinh output từ mô hình
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=512
    )

    # Decode ra chuỗi kết quả dạng thẻ
    result_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Sửa lỗi định dạng thẻ nếu có
    result_text = fix_unclosed_tags(result_text)

    # Tách các thẻ thành dict
    matches = re.findall(r"<(.*?)>(.*?)</\1>", result_text)
    result_dict = {k: v.strip() for k, v in matches}

    return result_dict
result = recognize_with_donut_json(r"C:\Users\Iamnearu\Documents\ThucTapAI\InternAI\dataset_donut\testsets\img932.jpg")
print(result)