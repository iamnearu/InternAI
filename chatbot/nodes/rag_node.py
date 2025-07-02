"""
File: rag_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Dùng mô hình Gemini (Google) để sinh câu trả lời RAG.
- Nhận context_chunks + câu hỏi → sinh prompt → trả về "answer" dạng dict.
"""
print("🚨 RAG_NODE.PY đang chạy từ:", __file__)

import google.generativeai as genai

# ⚠️ Thay bằng API Key Gemini của bạn
genai.configure(api_key="AIzaSyC6fS5gEB4WIuiXOMY-RRSDJOus8OnbjyU")

# Tạo mô hình Gemini đúng version hỗ trợ
model = genai.GenerativeModel("gemini-1.5-flash")  # hoặc "gemini-1.5-pro"

def build_prompt(context_chunks, question):
    """
    Ghép các đoạn context và câu hỏi thành prompt đầy đủ.
    """
    context = "\n".join(context_chunks)
    return f"""Dựa vào thông tin CCCD dưới đây, hãy trả lời câu hỏi.

Thông tin CCCD:
{context}

Câu hỏi: {question}
Trả lời:"""

def rag_node(state: dict) -> dict:
    """
    Node RAG: Nhận context_chunks + question → sinh answer bằng Gemini.

    Trả về:
        dict: {"answer": ...}
    """
    context_chunks = state.get("context_chunks", [])
    question = state.get("question", "")

    prompt = build_prompt(context_chunks, question)
    if not context_chunks:
        return {"answer": "❌ Không có thông tin context từ ảnh CCCD. OCR/Caption đều thất bại."}

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Lỗi khi gọi Gemini API: {str(e)}"}  # ✅ Bọc lỗi trong dict
