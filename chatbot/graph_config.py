"""
File: graph_config.py
Thư mục: chatbot/

Mục đích:
- Định nghĩa cấu trúc LangGraph pipeline.
- Gắn các node theo thứ tự xử lý ảnh CCCD → sinh câu trả lời.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph

# Import các node cần dùng từ hệ thống
from chatbot.nodes.ocr_node import ocr_node
from chatbot.nodes.caption_node import caption_node
from chatbot.nodes.merge_node import merge_node
from chatbot.nodes.embed_node import embed_node
from chatbot.nodes.query_node import query_node
from chatbot.nodes.rag_node import rag_node
from chatbot.nodes.answer_node import answer_node


# ✅ Định nghĩa schema pipeline bằng TypedDict
class ChatbotState(TypedDict):
    image_path: str           # Đường dẫn ảnh CCCD
    question: str             # Câu hỏi người dùng
    text: str                 # Văn bản trích từ ảnh (OCR + caption)
    embedding: List[float]    # Vector embedding văn bản
    context_chunks: List[str] # Các đoạn context lấy từ Qdrant
    answer: str               # Câu trả lời cuối cùng từ RAG


def build_pipeline():
    """
    Hàm khởi tạo pipeline LangGraph cho quá trình OCR + Caption + RAG.

    Trả về:
        pipeline LangGraph đã compile sẵn.
    """
    # 1. Khởi tạo đồ thị với schema đã định nghĩa
    graph = StateGraph(ChatbotState)

    # 2. Thêm các node tương ứng vào pipeline
    graph.add_node("ocr", ocr_node)               # Nhận dạng văn bản từ ảnh CCCD
    graph.add_node("caption", caption_node)       # Sinh mô tả ảnh (fallback)
    graph.add_node("merge", merge_node)           # Hợp nhất kết quả OCR + caption
    graph.add_node("embed", embed_node)           # Tính embedding của văn bản
    graph.add_node("query", query_node)           # Truy vấn Qdrant theo ngữ nghĩa
    graph.add_node("rag", rag_node)               # Tạo câu trả lời bằng mô hình RAG
    graph.add_node("answer", answer_node)         # Chuẩn hóa câu trả lời trả về frontend

    # 3. Thiết lập luồng xử lý giữa các node
    graph.set_entry_point("ocr")
    graph.add_edge("ocr", "caption")
    graph.add_edge("caption", "merge")
    graph.add_edge("merge", "embed")
    graph.add_edge("embed", "query")
    graph.add_edge("query", "rag")
    graph.add_edge("rag", "answer")

    # 4. Compile đồ thị thành pipeline có thể chạy
    return graph.compile()
