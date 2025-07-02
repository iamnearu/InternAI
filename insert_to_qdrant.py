"""
File: insert_to_qdrant.py
Mục đích:
- Nhúng thông tin CCCD sau khi OCR bằng Donut
- Lưu vector + metadata vào Qdrant để phục vụ RAG
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
import uuid

# ✅ Khởi tạo Qdrant client
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "cccd_info"

# ✅ Khởi tạo mô hình embedding
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ✅ Tạo collection nếu chưa tồn tại
if COLLECTION_NAME not in [col.name for col in client.get_collections().collections]:
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✅ Đã tạo collection `{COLLECTION_NAME}`")

# ✅ Hàm format dict → chuỗi văn bản
def dict_to_text(data: dict) -> str:
    return " | ".join([f"{k}: {v}" for k, v in data.items() if v])

# ✅ Hàm insert
def insert_cccd_info(data: dict):
    """
    - data: dict đã OCR, ví dụ:
        {
            "so_cccd": "0123456789",
            "ho_va_ten": "NGUYEN VAN A",
            "ngay_sinh": "01/01/2000"
        }
    """
    text = dict_to_text(data)
    embedding = embedder.encode(text, normalize_embeddings=True).tolist()

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "text": text,
            **data  # lưu toàn bộ trường gốc luôn
        }
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])
    print("✅ Đã upsert vào Qdrant:", data["so_cccd"])

# ✅ Ví dụ test
if __name__ == "__main__":
    fake_cccd = {
        "so_cccd": "001200000123",
        "ho_va_ten": "NGUYEN VAN A",
        "ngay_sinh": "10/01/2000",
        "gioi_tinh": "Nam",
        "nguyen_quan": "Ha Noi"
    }

    insert_cccd_info(fake_cccd)
