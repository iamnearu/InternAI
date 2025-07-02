"""
File: caption_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Tạo caption mô tả nội dung ảnh CCCD bằng Gemini Pro Vision.
- Dùng làm fallback nếu OCR thất bại.
"""

import google.generativeai as genai
from PIL import Image

# ✅ Cấu hình API key Gemini (bạn đã lấy trên Google AI Studio)
genai.configure(api_key="AIzaSyC6fS5gEB4WIuiXOMY-RRSDJOus8OnbjyU")  # 👉 NHỚ thay bằng API key thật

# ✅ Tạo đối tượng model Gemini hỗ trợ ảnh
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_caption(image_path: str) -> str:
    """
    Gửi ảnh và prompt yêu cầu mô tả ảnh CCCD bằng Gemini.

    Tham số:
        image_path (str): Đường dẫn ảnh CCCD.

    Trả về:
        str: Mô tả ảnh (caption) do Gemini sinh ra.
    """
    image = Image.open(image_path).convert("RGB")
    prompt = "Hãy mô tả nội dung ảnh CCCD này bằng tiếng Việt, chỉ miêu tả 3 trường là số căn cước công dân, họ và tên và ngày sinh."

    response = model.generate_content([prompt, image])
    return response.text.strip()

def caption_node(image_path: str) -> dict:
    """
    Node LangGraph: sinh caption từ ảnh CCCD.

    Trả về:
        dict: {
            "text": caption sinh ra,
            "success": True nếu có kết quả,
            "error": nếu lỗi
        }
    """
    try:
        caption = generate_caption(image_path)
        return {
            "text": caption,
            "success": True
        }

    except Exception as e:
        return {
            "text": "",
            "success": False,
            "error": str(e)
        }
