import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model và tokenizer
model_name = "./vit5_finetuned"  # Đường dẫn tới model đã huấn luyện của bạn
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base")

# Hàm để tóm tắt văn bản
def summarize_text(text):
    # Tokenize đầu vào
    inputs = tokenizer("Tóm tắt: " + text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    # Đưa mô hình vào chế độ GPU nếu có CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Sinh kết quả tóm tắt
    summary_ids = model.generate(inputs["input_ids"], max_length=200, num_beams=4, early_stopping=True)

    # Giải mã kết quả
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Tạo giao diện người dùng với Streamlit
st.title("Text Summarization App")

# Nhập văn bản cần tóm tắt
input_text = st.text_area("Nhập văn bản cần tóm tắt", height=300)

# Nút để tạo tóm tắt
if st.button("Tóm tắt"):
    if input_text.strip():
        summary = summarize_text(input_text)
        st.subheader("Tóm tắt kết quả:")
        st.write(summary)
    else:
        st.error("Vui lòng nhập văn bản để tóm tắt.")
