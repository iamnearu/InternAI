"""
File: merge_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Hợp nhất văn bản trích xuất từ OCR (Donut) và Caption (MiniGPT-4).
- Ưu tiên kết quả từ OCR nếu thành công, fallback sang caption nếu thất bại.
- Trả ra dict {"text": "..."} để tương thích với LangGraph.
"""

def merge_node(ocr_output: dict, caption_output: dict = None) -> dict:
    """
    Kết hợp văn bản từ OCR node và Caption node.

    Tham số:
        ocr_output (dict): Kết quả từ OCR node. {"text": ..., "success": True/False}
        caption_output (dict): Kết quả từ Caption node (có thể là None)

    Trả về:
        dict: {"text": văn bản được chọn}
    """

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

    # Nếu OCR (Donut) thành công → ưu tiên dùng
    if ocr_output.get("success") and ocr_output.get("text"):
        print("🧾 OCR Output:", ocr_output)
        text = ocr_output["text"]
        if isinstance(text, dict):
            text = json_to_text(text)
        return {"text": text}

    # Nếu caption có kết quả và OCR thất bại
    elif caption_output and caption_output.get("success") and caption_output.get("text"):
        return {"text": caption_output["text"]}

    # Nếu cả hai đều thất bại → text rỗng
    return {"text": ""}

