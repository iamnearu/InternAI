# """
# File: answer_node.py
# Thư mục: chatbot/nodes/
#
# Mục đích:
# - Nhận đầu ra từ RAG và format lại câu trả lời.
# - Trả về {"answer": ..., "success": True/False, "question": ...}
# """
#
# def answer_node(state: dict) -> dict:
#     """
#     Truy xuất từ state:
#         - state["answer"] (do RAG sinh ra)
#         - state["question"] (câu hỏi gốc)
#
#     Trả về dict kết quả cuối cùng.
#     """
#     rag_output = state.get("answer", "")
#     original_question = state.get("question", "")
#
#     if not rag_output or not rag_output.strip():
#         return {
#             "question": original_question,
#             "answer": "Xin lỗi, tôi không thể tìm thấy thông tin phù hợp trong ảnh CCCD.",
#             "success": False
#         }
#
#     return {
#         "question": original_question,
#         "answer": rag_output.strip(),
#         "success": True
#     }
"""
File: answer_node.py
Mục đích: Định dạng kết quả từ mô hình RAG để gửi về frontend
"""
"""
File: answer_node.py
Mục đích:
- Bảo vệ chống lỗi .get() khi input không đúng
- Dùng cho cả trường hợp truyền str hoặc dict
"""

def answer_node(state) -> dict:
    # ✅ Nếu nhận vào là string, thì convert thành dict giả
    if isinstance(state, str):
        print("⚠️ [answer_node] Nhận string → convert thành dict")
        state = {"answer": state, "question": ""}

    # ✅ Nếu không phải dict thì trả lỗi rõ ràng luôn
    if not isinstance(state, dict):
        print("❌ [answer_node] Không phải dict. Kiểu:", type(state))
        return {
            "question": "",
            "answer": "❌ Lỗi: Đầu vào sai định dạng.",
            "success": False
        }

    rag_output = state.get("answer", "")
    original_question = state.get("question", "")

    if not rag_output or not isinstance(rag_output, str) or not rag_output.strip():
        return {
            "question": original_question,
            "answer": "❌ Không thể tìm thấy thông tin phù hợp.",
            "success": False
        }

    return {
        "question": original_question,
        "answer": rag_output.strip(),
        "success": True
    }
