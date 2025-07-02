"""
File: query_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Nhận câu hỏi của người dùng.
- Tính embedding → truy vấn Qdrant → trả về các đoạn văn bản gần nhất.
- Trả về dict {"context_chunks": [...]} để tương thích với LangGraph.
"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import torch

# Load mô hình embedding cho câu hỏi
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = embedder.to("cuda" if torch.cuda.is_available() else "cpu")

# Kết nối Qdrant
qdrant = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "cccd_info"
def query_node(state: dict, top_k: int = 3) -> dict:
    """
    Nhận toàn bộ state → truy vấn Qdrant theo `state["question"]`
    Trả về: {"context_chunks": [...]}
    """
    question = state.get("question")

    # Kiểm tra để tránh lỗi encode
    if not isinstance(question, str):
        raise ValueError(f"[query_node] ❌ 'question' phải là str, nhận được: {type(question)} - {question}")

    # Encode embedding
    vector = embedder.encode(question, normalize_embeddings=True).tolist()

    # Truy vấn Qdrant
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )

    context_chunks = []
    for hit in hits:
        payload = hit.payload
        if isinstance(payload, dict):
            text = payload.get("text") or str(payload)
            context_chunks.append(text)

    return {"context_chunks": context_chunks}
