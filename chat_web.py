# File: chat_web.py
# Web đơn giản hỏi ảnh CCCD + câu hỏi → trả lời bằng RAG
import streamlit as st
import requests

st.set_page_config(page_title="Chatbot CCCD", page_icon="🪪")
st.title("🪪 Chatbot CCCD – Đọc ảnh & trả lời")

# Giao diện upload ảnh
uploaded_file = st.file_uploader("📤 Tải ảnh CCCD", type=["png", "jpg", "jpeg"])

# Nhập câu hỏi
question = st.text_input("💬 Bạn muốn hỏi gì về CCCD?")

# Hiển thị nút gửi
if st.button("🚀 Gửi") and uploaded_file and question:
    with st.spinner("⏳ Đang gửi..."):
        try:
            files = {"image": uploaded_file}
            data = {"question": question}
            response = requests.post("http://127.0.0.1:8000/chat", files=files, data=data)
            result = response.json()

            # Hiển thị kết quả
            st.markdown("### ✅ Kết quả:")
            st.markdown(f"**Bạn hỏi:** {question}")
            st.markdown(f"**Bot trả lời:** {result['answer']}")

        except Exception as e:
            st.error(f"Lỗi khi gọi API: {e}")
