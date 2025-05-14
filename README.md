# Text Summarization App (Vietnamese)

Ứng dụng này sử dụng mô hình **ViT5 (VietAI)** đã được fine-tune để tạo tóm tắt tự động cho văn bản tiếng Việt. Giao diện người dùng được xây dựng với **Streamlit**, giúp bạn dễ dàng nhập văn bản và nhận kết quả tóm tắt nhanh chóng.

---

## Cách Hoạt Động

1. **Huấn luyện mô hình**:
   - Dữ liệu gồm 2 cột: `văn bản` (nội dung cần tóm tắt) và `tóm tắt` (tóm tắt tương ứng).
   - Mỗi dòng được chuyển thành format: `"Tóm tắt: <văn bản>"` làm đầu vào và `<tóm tắt>` là đầu ra.
   - Mô hình sử dụng là `VietAI/vit5-base` được huấn luyện lại bằng `Trainer` từ Hugging Face.

2. **Tóm tắt văn bản**:
   - Nhập văn bản trong giao diện Streamlit.
   - Mô hình sẽ sinh ra bản tóm tắt với beam search (`num_beams=4`) và độ dài tối đa 200 token.
   - Kết quả sẽ được hiển thị trực tiếp.

---

## Cài Đặt

### 1. Tạo môi trường Python (nên dùng Python 3.9+)

```bash
python -m venv venv
source venv/bin/activate       # Trên Linux/Mac
venv\Scripts\activate          # Trên Windows
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install torch transformers datasets streamlit pandas
```

### 3. Chạy app

```bash
streamlit run app.py
```

### 4. Cấu trúc thư mục khuyến nghị

```
text-summary-project/
│
├── vit5_finetuned/         # Model đã fine-tune từ VietAI/vit5-base
├── app.py                  # File giao diện Streamlit
├── model.py                # Script huấn luyện mô hình (hoặc notebook)
├── data.csv                # Dữ liệu huấn luyện (2 cột: văn bản, tóm tắt)
└── README.md               # File hướng dẫn
```
