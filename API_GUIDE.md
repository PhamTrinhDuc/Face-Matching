# Face Matching API Guide

Hệ thống API cho việc đăng ký và xác thực khuôn mặt sinh viên.

## Cài đặt

1. Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

2. Khởi động Milvus server:
```bash
docker-compose up -d
```

3. Khởi tạo database và collection:
```bash
python init_system.py
```

4. Chạy API server:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## Cấu trúc dữ liệu

### Database Schema (SQLite)

#### Bảng Students (SQLite)
- `id`: Primary key (auto increment)
- `student_id`: Mã sinh viên (unique, auto-generated)
- `full_name`: Họ tên đầy đủ
- `email`: Email sinh viên (optional)
- `phone`: Số điện thoại (optional)
- `student_code`: Mã sinh viên/lớp (required)
- `image_path`: Đường dẫn file ảnh đã lưu
- `created_at`: Thời gian tạo
- `updated_at`: Thời gian cập nhật

### Milvus Collection Schema

#### Collection: student_faces
- `id`: Primary key (auto generated)
- `student_id`: Mã sinh viên (VARCHAR, 50)
- `image_path`: Đường dẫn ảnh (VARCHAR, 500)
- `embedding`: Vector embedding (FLOAT_VECTOR, dim=512)
- `metadata`: Metadata bổ sung (VARCHAR, 1000)

#### Index
- Type: IVF_FLAT
- Metric: COSINE
- Parameters: nlist=128

## API Endpoints

### 1. Health Check
```
GET /health
```
Kiểm tra trạng thái của API server.

**Response:**
```json
{
  "status": "running",
  "model_loaded": true
}
```

### 2. Đăng ký khuôn mặt sinh viên
```
POST /register
```
Đăng ký khuôn mặt mới cho sinh viên.

**Parameters:**
- `full_name` (form): Họ tên đầy đủ
- `student_code` (form): Mã sinh viên/lớp (required)
- `email` (form, optional): Email sinh viên
- `phone` (form, optional): Số điện thoại
- `file` (file): File ảnh chứa khuôn mặt

**Response:**
```json
{
  "success": true,
  "message": "Student SV001 registered successfully.",
  "student_id": "SV001",
  "embedding_id": 12345,
  "db_id": 1
}
```

**Curl Example:**
```bash
curl -X POST "http://localhost:8001/register" \
  -F "full_name=Nguyen Van A" \
  -F "student_code=CNTT2025" \
  -F "email=nguyenvana@example.com" \
  -F "file=@/path/to/student_photo.jpg"
```

**Note:** `student_id` sẽ được tự động tạo theo format: `{student_code}_{timestamp}_{random}` (ví dụ: CNTT2025_123456_A1B2)

### 3. Xác thực khuôn mặt
```
POST /verify
```
Kiểm tra khuôn mặt với cơ sở dữ liệu để tìm sinh viên tương ứng.

**Parameters:**
- `file` (file): File ảnh cần kiểm tra
- `threshold` (form, optional): Ngưỡng độ tương đồng (default: 0.7)

**Response:**
```json
{
  "success": true,
  "matched": true,
  "message": "Face matched successfully! Welcome Nguyen Van A",
  "student_id": "SV001",
  "student_name": "Nguyen Van A",
  "confidence": 0.89,
  "similar_faces": [
    {
      "id": 12345,
      "student_id": "SV001",
      "image_path": "student_SV001_photo.jpg",
      "metadata": "{\"full_name\": \"Nguyen Van A\", \"class\": \"CNTT2021\"}",
      "distance": 0.89,
      "confidence": 0.89
    }
  ]
}
```

**Curl Example:**
```bash
curl -X POST "http://localhost:8001/verify" \
  -F "file=@/path/to/check_photo.jpg" \
  -F "threshold=0.7"
```

### 4. Lấy danh sách sinh viên đã đăng ký
```
GET /students?class_filter=CNTT2021
```
Lấy danh sách tất cả sinh viên đã đăng ký trong hệ thống.

**Parameters (Query):**
- `student_code_filter` (optional): Lọc theo mã sinh viên/lớp

**Response:**
```json
{
  "success": true,
  "total_students": 2,
  "students": [
    {
      "id": 1,
      "student_id": "SV001",
      "full_name": "Nguyen Van A",
      "email": "nguyenvana@example.com",
      "phone": "0123456789",
      "student_code": "CNTT2025",
      "image_path": "/path/to/saved/image.jpg",
      "created_at": "2025-11-17T10:30:00",
      "updated_at": "2025-11-17T10:30:00"
    }
  ]
}
```

### 5. Xóa sinh viên
```
DELETE /student/{student_id}
```
Xóa thông tin khuôn mặt của sinh viên khỏi hệ thống.

**Parameters:**
- `student_id` (path): ID của sinh viên cần xóa

**Response:**
```json
{
  "success": true,
  "message": "Student SV001 deleted successfully.",
  "deleted_records": 1
}
```

**Curl Example:**
```bash
curl -X DELETE "http://localhost:8001/student/SV001"
```

## Cấu trúc Response

### RegisterResponse
- `success`: Boolean - Trạng thái thành công
- `message`: String - Thông báo kết quả
- `student_id`: String (optional) - ID sinh viên đã đăng ký
- `embedding_id`: Integer (optional) - ID của embedding trong Milvus

### VerifyResponse
- `success`: Boolean - Trạng thái thành công
- `matched`: Boolean - Có khớp khuôn mặt không
- `message`: String - Thông báo kết quả
- `student_id`: String (optional) - ID sinh viên được tìm thấy
- `confidence`: Float (optional) - Độ tin cậy (0.0 - 1.0)
- `similar_faces`: List (optional) - Danh sách các khuôn mặt tương tự

## Lưu ý quan trọng

1. **Định dạng ảnh**: API chỉ chấp nhận các file ảnh (JPEG, PNG, etc.)

2. **Chất lượng ảnh**: Ảnh cần rõ nét và chứa ít nhất 1 khuôn mặt để có thể detect được

3. **Ngưỡng confidence**: Ngưỡng mặc định là 0.7. Có thể điều chỉnh tùy theo yêu cầu:
   - 0.6-0.7: Loose matching (ít nghiêm ngặt)
   - 0.7-0.8: Normal matching (khuyến nghị)
   - 0.8-0.9: Strict matching (rất nghiêm ngặt)

4. **Milvus Server**: Đảm bảo Milvus server đang chạy trước khi khởi động API

5. **Model file**: Đảm bảo file model `backbone_ir50_ms1m_epoch120.pth` có trong thư mục `checkpoint/`

## Swagger UI

Sau khi khởi động server, có thể truy cập Swagger UI tại:
```
http://localhost:8001/docs
```

## Testing

Có thể test API bằng cách sử dụng các file ảnh mẫu trong thư mục `data/`.