import streamlit as st
import requests
from PIL import Image
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Face Registration System",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
st.sidebar.title("⚙️ Cấu hình")
API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8001",
    help="Nhập địa chỉ server API"
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Ứng dụng này dùng để đăng ký khuôn mặt sinh viên vào hệ thống nhận dạng"
)

# Main title
st.title("📸 Hệ Thống Đăng Ký Khuôn Mặt Sinh Viên")
st.markdown("---")

# Create tabs
tab1, tab2 = st.tabs(["📝 Đăng Ký Mới", "👥 Danh Sách Sinh Viên"])

with tab1:
    st.subheader("Đăng Ký Sinh Viên Mới")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        full_name = st.text_input(
            "Tên đầy đủ (*)",
            placeholder="Ví dụ: Nguyễn Văn A",
            help="Nhập tên đầy đủ của sinh viên"
        )
        
        student_code = st.text_input(
            "Mã sinh viên (*)",
            placeholder="Ví dụ: SV001",
            help="Nhập mã sinh viên duy nhất"
        )
    
    with col2:
        email = st.text_input(
            "Email (tùy chọn)",
            placeholder="Ví dụ: student@university.edu",
            help="Nhập email sinh viên"
        )
        
        phone = st.text_input(
            "Số điện thoại (tùy chọn)",
            placeholder="Ví dụ: 0912345678",
            help="Nhập số điện thoại"
        )
    
    st.markdown("---")
    st.subheader("📸 Tải Ảnh Khuôn Mặt")
    
    uploaded_file = st.file_uploader(
        "Chọn ảnh khuôn mặt (*)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Tải ảnh có chứa khuôn mặt rõ ràng"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã chọn", use_column_width=True)
        
        with col2:
            st.info(
                f"""
                **Thông tin ảnh:**
                - Tên file: {uploaded_file.name}
                - Kích thước: {uploaded_file.size / 1024:.2f} KB
                - Loại: {uploaded_file.type}
                """
            )
    
    st.markdown("---")
    
    # Register button
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        submit_button = st.button(
            "✅ Đăng Ký Sinh Viên",
            use_container_width=True,
            type="primary",
            disabled=not (full_name and student_code and uploaded_file)
        )
    
    if submit_button:
        if not full_name:
            st.error("❌ Vui lòng nhập tên đầy đủ")
        elif not student_code:
            st.error("❌ Vui lòng nhập mã sinh viên")
        elif not uploaded_file:
            st.error("❌ Vui lòng chọn ảnh")
        else:
            with st.spinner("⏳ Đang xử lý..."):
                try:
                    # Prepare files and data
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)
                    }
                    
                    data = {
                        "full_name": full_name,
                        "student_code": student_code,
                        "email": email if email else None,
                        "phone": phone if phone else None,
                    }
                    
                    # Send request to API
                    response = requests.post(
                        f"{API_URL}/register",
                        files=files,
                        data=data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("success"):
                            st.success(
                                f"""
                                ✅ **Đăng Ký Thành Công!**
                                
                                - **Mã sinh viên**: {result.get('student_id')}
                                - **Tên**: {full_name}
                                - **Mã code**: {student_code}
                                - **ID nhúng**: {result.get('embedding_id')}
                                
                                {result.get('message')}
                                """
                            )
                            
                            # Clear form
                            st.session_state.clear()
                            st.rerun()
                        else:
                            st.error(f"❌ Lỗi: {result.get('message', 'Không xác định')}")
                    else:
                        st.error(f"❌ Lỗi server: {response.status_code}")
                        st.error(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"❌ Không thể kết nối đến server tại {API_URL}"
                        "\n\nVui lòng kiểm tra:\n"
                        "1. Server API đang chạy?\n"
                        "2. Địa chỉ API đúng không?"
                    )
                except requests.exceptions.Timeout:
                    st.error("❌ Yêu cầu vượt quá thời gian chờ. Vui lòng thử lại")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")

with tab2:
    st.subheader("👥 Danh Sách Sinh Viên Đã Đăng Ký")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        student_code_filter = st.text_input(
            "🔍 Tìm kiếm theo mã sinh viên hoặc lớp (tùy chọn)",
            placeholder="Nhập mã sinh viên hoặc lớp học..."
        )
    
    with col2:
        refresh_button = st.button("🔄 Làm mới", use_container_width=True)
    
    if refresh_button or "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    with st.spinner("⏳ Đang tải danh sách..."):
        try:
            params = {}
            if student_code_filter:
                params["student_code_filter"] = student_code_filter
            
            response = requests.get(
                f"{API_URL}/students",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    total = result.get("total_students", 0)
                    students = result.get("students", [])
                    
                    st.info(f"📊 Tổng số sinh viên: **{total}**")
                    
                    if students:
                        # Create a table
                        st.markdown("### Danh sách chi tiết")
                        
                        for idx, student in enumerate(students, 1):
                            col1, col2, col3 = st.columns([2, 2, 1])
                            
                            with col1:
                                st.markdown(f"**{idx}. {student.get('full_name', 'N/A')}**")
                                st.caption(f"Mã sinh viên: {student.get('student_id', 'N/A')}")
                            
                            with col2:
                                st.markdown(f"📧 {student.get('email', 'Không có')}")
                                st.markdown(f"📱 {student.get('phone', 'Không có')}")
                            
                            with col3:
                                if st.button(
                                    "🗑️ Xóa",
                                    key=f"delete_{student.get('student_id')}",
                                    help="Xóa sinh viên này"
                                ):
                                    with st.spinner("⏳ Đang xóa..."):
                                        try:
                                            del_response = requests.delete(
                                                f"{API_URL}/student/{student.get('student_id')}",
                                                timeout=10
                                            )
                                            
                                            if del_response.status_code == 200:
                                                del_result = del_response.json()
                                                if del_result.get("success"):
                                                    st.success("✅ Xóa thành công!")
                                                    st.rerun()
                                                else:
                                                    st.error(f"❌ {del_result.get('message')}")
                                            else:
                                                st.error("❌ Lỗi xóa sinh viên")
                                        except Exception as e:
                                            st.error(f"❌ Lỗi: {str(e)}")
                            
                            st.markdown("---")
                    else:
                        st.info("Không tìm thấy sinh viên nào")
                else:
                    st.error(f"❌ {result.get('message', 'Không xác định')}")
            else:
                st.error(f"❌ Lỗi server: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Không thể kết nối đến server tại {API_URL}"
            )
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
    🎓 Hệ Thống Nhận Dạng Khuôn Mặt Sinh Viên | v1.0
    </div>
    """,
    unsafe_allow_html=True
)

# streamlit run streamlit_app.py