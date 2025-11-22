# 📸 Streamlit Face Registration App

Giao diện Streamlit đơn giản để đăng ký khuôn mặt sinh viên.

## ✨ Tính năng

- ✅ Đăng ký sinh viên mới với ảnh khuôn mặt
- 👥 Xem danh sách sinh viên đã đăng ký
- 🔍 Tìm kiếm sinh viên theo mã hoặc lớp
- 🗑️ Xóa sinh viên khỏi hệ thống
- ⚙️ Cấu hình URL API linh hoạt

## 📋 Yêu cầu

- Python 3.8+
- FastAPI server đang chạy (main.py)

## 🚀 Cách chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Chạy server API

```bash
# Terminal 1 - Chạy FastAPI server
cd /home/ducpham/workspace/Face-Matching
python main.py

# Hoặc dùng uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 3. Chạy Streamlit app

```bash
# Terminal 2 - Chạy Streamlit
streamlit run streamlit_app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 📖 Hướng dẫn sử dụng

### Tab "Đăng Ký Mới"

1. Nhập thông tin sinh viên:
   - **Tên đầy đủ** (*bắt buộc)
   - **Mã sinh viên** (*bắt buộc)
   - **Email** (tùy chọn)
   - **Số điện thoại** (tùy chọn)

2. Tải ảnh khuôn mặt (*bắt buộc)
   - Định dạng: JPG, JPEG, PNG, BMP
   - Ảnh phải có khuôn mặt rõ ràng

3. Nhấn "✅ Đăng Ký Sinh Viên"

4. Xem kết quả:
   - Thành công: Hiển thị ID sinh viên mới
   - Lỗi: Kiểm tra lại thông tin

### Tab "Danh Sách Sinh Viên"

1. Xem tất cả sinh viên đã đăng ký
2. Tìm kiếm sinh viên theo mã hoặc lớp
3. Xóa sinh viên nếu cần (click nút 🗑️)
4. Làm mới danh sách (click 🔄)

## ⚙️ Cấu hình

- **API URL**: Mặc định `http://localhost:8001`
- Có thể thay đổi tại sidebar "⚙️ Cấu hình"

## 🐛 Troubleshooting

### Lỗi: "Không thể kết nối đến server"
- Kiểm tra server API có chạy không
- Kiểm tra cổng 8001 có đúng không

### Ảnh không được nhận diện khuôn mặt
- Tải ảnh khác có khuôn mặt rõ hơn
- Kiểm tra logs từ server API

## 📞 Support

Nếu có vấn đề, kiểm tra:
1. Logs trong terminal chạy Streamlit
2. Logs trong terminal chạy FastAPI server
3. Đảm bảo database Milvus đang chạy
