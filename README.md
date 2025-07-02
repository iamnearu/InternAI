# InternAI
# 📄 CCCD QA Pipeline (LangGraph + Gemini)

Dự án xây dựng hệ thống nhận diện và hỏi đáp tự động từ ảnh căn cước công dân (CCCD) bằng pipeline LangGraph. Hệ thống sử dụng mô hình OCR Donut để trích xuất dữ liệu, Gemini 1.5 Flash để sinh câu trả lời, kết hợp Qdrant để lưu trữ tri thức ngữ nghĩa.

---

## 🧠 Kiến trúc tổng quan

```text
Ảnh CCCD → OCR (Donut) + Caption (Gemini/MiniGPT4) 
         ↓
   Văn bản mô tả CCCD
         ↓
   → Embed (SentenceTransformer)
         ↓
   → Truy vấn Qdrant (semantic search)
         ↓
   → Rag → Gemini → Trả lời
├── chatbot/
│   ├── graph_config.py         # Định nghĩa pipeline LangGraph
│   └── nodes/
│       ├── ocr_node.py         # Node OCR (Donut)
│       ├── caption_node.py     # Node Caption ảnh CCCD (giả lập / Gemini 2.5 sau này)
│       ├── merge_node.py       # Hợp nhất OCR + Caption
│       ├── embed_node.py       # Embedding văn bản
│       ├── query_node.py       # Truy vấn Qdrant
│       ├── rag_node.py         # Rag: sinh câu trả lời bằng Gemini
│       └── answer_node.py      # Gửi trả kết quả cuối cùng
│
├── ocr/
│   └── recognizer.py           # Hàm nhận diện ảnh bằng Donut
│
├── caption/
│   └── caption_generator.py    # Tạm sinh caption mô tả ảnh
│
├── insert_to_qdrant.py         # Hàm lưu thông tin vào Qdrant
├── main.py                     # File chạy chính
1. Cài đặt môi trường
pip install -r requirements.txt
2. Chạy Qdrant local bằng Docker
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
3. Chạy pipeline
\python main.py
\4. Kết quả
📸 Ảnh: dataset_donut/testsets/img932.jpg
❓ Câu hỏi: Người này tên là gì?
✅ Trả lời: Họ tên: H TẤN HUY
🔑 Thiết lập API Key Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")