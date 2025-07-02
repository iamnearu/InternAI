"""
File: ocr_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Trích xuất thông tin CCCD từ ảnh bằng Donut (OCR).
- Sau khi OCR thành công, tự động lưu thông tin đó vào Qdrant.
"""

from ocr.recognizer import recognize_with_donut_json
from insert_to_qdrant import insert_cccd_info  # ✅ Gắn hàm insert
def json_to_text(data: dict) -> str:
    """
    Chuyển dict {'so_cccd': '...', 'ho_va_ten': '...'} thành đoạn text mô tả.
    """
    lines = []
    if "so_cccd" in data:
        lines.append(f"Số CCCD: {data['so_cccd']}")
    if "ho_va_ten" in data:
        lines.append(f"Họ tên: {data['ho_va_ten']}")
    if "ngay_sinh" in data:
        lines.append(f"Ngày sinh: {data['ngay_sinh']}")
    return "\n".join(lines)

def ocr_node(image_path: str) -> dict:
    """
    Node OCR trích thông tin từ ảnh CCCD và lưu kết quả vào Qdrant nếu thành công.

    Trả về:
        dict: {
            "text": ...,  # Chuỗi ghép từ JSON
            "success": True/False,
            "error": ...
        }
    """
    try:
        # ✅ 1. OCR ảnh
        result_dict = recognize_with_donut_json(image_path)

        # ✅ 2. Ghép lại thành chuỗi để đưa vào LangGraph
        text = " | ".join([f"{k}: {v}" for k, v in result_dict.items()])

        # ✅ 3. Lưu vào Qdrant nếu có dữ liệu
        if result_dict:
            insert_cccd_info(result_dict)
        print(result_dict)
        text = json_to_text(text)  # output là dict
        return {"text": text, "success": True}

    except Exception as e:
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }
