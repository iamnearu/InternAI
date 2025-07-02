"""
File: embed_node.py
Thư mục: chatbot/nodes/

Mục đích:
- Dùng mô hình SentenceTransformer để nhúng văn bản thành vector.
- Trả kết quả dưới dạng dict để tương thích với LangGraph pipeline.
"""

from sentence_transformers import SentenceTransformer  # Thư viện embedding văn bản
import numpy as np
import torch

# Load mô hình embedding
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Đưa model lên GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def embed_node(text: str) -> dict:
    """
    Nhúng văn bản thành vector embedding và trả về dict.

    Tham số:
        text (str): văn bản cần nhúng (đã qua OCR + caption)

    Trả về:
        dict: {"embedding": vector dưới dạng list[float]}
    """

    # Kiểm tra đầu vào
    if not text or not isinstance(text, str):
        return {"embedding": []}

    # Nhúng văn bản → numpy array
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    # Trả ra dict để LangGraph hiểu được
    return {"embedding": embedding.tolist()}
