"""
File: donut_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Đây là node OCR chuyên biệt gọi mô hình Donut đã fine-tune để trích xuất thông tin từ ảnh CCCD.
- Output là JSON chứa các trường như: số CCCD, họ tên, ngày sinh.
"""

from ocr.recognizer import recognize_with_donut_json  # Hàm xử lý ảnh → JSON

def donut_node(image_path):
    """
    Node LangGraph gọi mô hình Donut để trích xuất thông tin từ ảnh CCCD.

    Tham số:
        image_path (str): Đường dẫn tới ảnh CCCD.

    Trả về:
        dict: {
            "data": dict kết quả JSON (các trường đã trích),
            "success": True nếu thành công, False nếu lỗi,
            "error": (nếu có lỗi)
        }
    """
    try:
        # Gọi inference bằng mô hình Donut
        extracted = recognize_with_donut_json(image_path)

        return {
            "data": extracted,
            "success": True
        }

    except Exception as e:
        return {
            "data": {},
            "success": False,
            "error": str(e)
        }
