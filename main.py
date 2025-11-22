import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import io
from PIL import Image
import tempfile
from typing import Optional, Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from face_embedder import FaceEmbedder
from face_detector import cropper_yunet, cropper_medipipe
from vectordb import MilvusClient
from models.database import get_db, create_tables, Student
from models.student_service import StudentService, create_student_service


app = FastAPI(
    title="Face Matching System",
    description="API for student face registration and verification",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face embedder
MODEL_PATH = "./checkpoint/backbone_ir50_ms1m_epoch120.pth"
embedder = FaceEmbedder(MODEL_PATH)

# Initialize Milvus client
milvus_client = MilvusClient(
    host="localhost", 
    port="19530", 
    collection_name="student_faces"
)

# Response models
class StudentCreateRequest(BaseModel):
    student_id: str
    full_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    class_name: Optional[str] = None

class RegisterResponse(BaseModel):
    success: bool
    message: str
    student_id: Optional[str] = None
    embedding_id: Optional[int] = None
    db_id: Optional[int] = None

class VerifyResponse(BaseModel):
    success: bool
    message: str
    matched: bool
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    confidence: Optional[float] = None
    similar_faces: Optional[List[Dict]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize database and Milvus connection on startup"""
    try:
        # Initialize database
        create_tables()
        print("Database initialized successfully")
        
        # Initialize Milvus
        milvus_client.connect()
        
        try:
            milvus_client.create_collection(dim=512)
        except Exception as e:
            print(f"Error creating collection: {e}")
            print("Trying to recreate collection...")
            milvus_client.recreate_collection(dim=512)
        
        milvus_client.load_collection()
        print("Milvus initialized successfully")
    except Exception as e:
        print(f"Error initializing services: {e}")
        print("API will continue to run, but face matching may not work properly")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up Milvus connection on shutdown"""
    try:
        milvus_client.disconnect()
        print("Milvus disconnected successfully")
    except Exception as e:
        print(f"Error disconnecting Milvus: {e}")

@app.get("/health")
async def get_status():
    """Health check endpoint"""
    return {"status": "running", "model_loaded": True}

@app.post("/register", response_model=RegisterResponse)
async def register_student_face(
    full_name: str = Form(...),
    student_code: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    file: UploadFile = File(...)
):
    """
    Đăng ký khuôn mặt cho sinh viên
    
    Args:
        full_name: Tên đầy đủ của sinh viên
        email: Email sinh viên (optional)
        phone: Số điện thoại (optional)
        class_name: Lớp học (optional)
        file: File ảnh chứa khuôn mặt sinh viên
    
    Returns:
        RegisterResponse: Kết quả đăng ký
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image."
            )
        
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save temporary file for face detection
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Detect and crop face
            cropped_face = cropper_yunet(temp_path)
            if cropped_face is None:
                return RegisterResponse(
                    success=False,
                    message="No face detected in the image. Please upload a clear face image."
                )
            
            # Extract embedding from the CROPPED face (not original image)
            # This ensures consistency with verification step
            embedding = embedder.embed_single_image(cropped_face)
            
            # Use StudentService to handle database operations
            with create_student_service() as service:
                student = service.get_student_by_code(student_code)
                if student: 
                    return RegisterResponse(
                        success=False,
                        message=f"Student with code {student_code} already registered.",
                        student_id=student.student_id,
                        db_id=student.id
                    )
                # Create student record in database first (student_id will be auto-generated)
                student = service.create_student(
                    full_name=full_name,
                    student_code=student_code,
                    email=email,
                    phone=phone,
                    image_file=None  # Will save image manually below
                )
                
                student_id = student.student_id  # Get the auto-generated student_id
                
                # Save image manually
                image_path = service._save_student_image(student_id, image_data, file.filename)
                service.update_student(student_id, image_path=image_path)
                
                # Store embedding in Milvus
                image_filename = f"student_{student_id}_{file.filename}"
                milvus_ids = milvus_client.insert_student_embedding(
                    student_id=student_id,
                    image_path=image_filename,
                    embedding=embedding,
                    metadata=f"{{\"full_name\": \"{full_name}\", \"student_code\": \"{student_code}\"}}"
                )
                
                milvus_id = milvus_ids[0] if milvus_ids else None
                
                return RegisterResponse(
                    success=True,
                    message=f"Student {student.student_id} registered successfully.",
                    student_id=student.student_id,
                    embedding_id=milvus_id,
                    db_id=student.id
                )
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except ValueError as ve:
        return RegisterResponse(
            success=False,
            message=str(ve)
        )
    except Exception as e:
        print(f"Error in register_student_face: {e}")
        return RegisterResponse(
            success=False,
            message=f"Registration failed: {str(e)}"
        )

@app.post("/verify", response_model=VerifyResponse)
async def verify_student_face(
    file: UploadFile = File(...),
    threshold: float = Form(0.3)
):
    """
    Kiểm tra khuôn mặt sinh viên với cơ sở dữ liệu
    
    Args:
        file: File ảnh chứa khuôn mặt cần kiểm tra
        threshold: Ngưỡng độ tương đồng (0.0 - 1.0)
    
    Returns:
        VerifyResponse: Kết quả kiểm tra
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Please upload an image."
            )
        
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save temporary file for face detection
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Detect face
            cropped_face = cropper_yunet(temp_path)
            if cropped_face is None:
                return VerifyResponse(
                    success=False,
                    matched=False,
                    message="No face detected in the image. Please upload a clear face image."
                )
            
            # Extract embedding
            query_embedding = embedder.embed_single_image(cropped_face)
            
            # Search for similar faces in Milvus
            similar_faces = milvus_client.search_similar_students(
                query_embedding=query_embedding,
                top_k=5,
                search_params={"metric_type": "IP", "params": {"nprobe": 10}}
            )
            
            if not similar_faces:
                return VerifyResponse(
                    success=True,
                    matched=False,
                    message="No matching face found in database.",
                    similar_faces=[]
                )
            
            # Check if best match exceeds threshold
            best_match = similar_faces[0]
            confidence = best_match.get('distance', 0.0)  # Cosine similarity score
            matched_student_id = best_match.get('student_id')
            
            # Get student details from database
            student_name = None
            with create_student_service() as service:
                if matched_student_id:
                    student = service.get_student_by_id(matched_student_id)
                    if student:
                        student_name = student.full_name
            
            if confidence >= threshold:
                return VerifyResponse(
                    success=True,
                    matched=True,
                    message=f"Face matched successfully! Welcome {student_name or matched_student_id}",
                    student_id=matched_student_id,
                    student_name=student_name,
                    confidence=confidence,
                    similar_faces=similar_faces[:3]  # Return top 3 matches
                )
            else:
                return VerifyResponse(
                    success=True,
                    matched=False,
                    message=f"Face not matched. Confidence {confidence:.3f} below threshold {threshold}.",
                    confidence=confidence,
                    similar_faces=similar_faces[:3]
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"Error in verify_student_face: {e}")
        return VerifyResponse(
            success=False,
            matched=False,
            message=f"Verification failed: {str(e)}"
        )

@app.post("/verify/raw", response_model=VerifyResponse)
async def verify_student_face_raw(
    request: Request,
    threshold: float = 0.7
):
    """
    Endpoint cho ESP32-CAM gửi ảnh binary thẳng
    Gửi: Content-Type: image/jpeg + threshold trong query param
    """
    try:
        # Đọc raw image data
        image_data = await request.body()
        
        if not image_data:
            return VerifyResponse(
                success=False,
                matched=False,
                message="No image data received."
            )
        
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Save temporary file for face detection
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            image.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Detect face
            cropped_face = cropper_yunet(temp_path)
            if cropped_face is None:
                return VerifyResponse(
                    success=False,
                    matched=False,
                    message="No face detected in the image. Please upload a clear face image."
                )
            
            # Extract embedding
            query_embedding = embedder.embed_single_image(cropped_face)
            
            # Search for similar faces in Milvus
            similar_faces = milvus_client.search_similar_students(
                query_embedding=query_embedding,
                top_k=5,
                search_params={"metric_type": "IP", "params": {"nprobe": 10}}
            )
            
            if not similar_faces:
                return VerifyResponse(
                    success=True,
                    matched=False,
                    message="No matching face found in database.",
                    similar_faces=[]
                )
            
            # Check if best match exceeds threshold
            best_match = similar_faces[0]
            confidence = best_match.get('distance', 0.0)  # Cosine similarity score
            matched_student_id = best_match.get('student_id')
            
            # Get student details from database
            student_name = None
            with create_student_service() as service:
                if matched_student_id:
                    student = service.get_student_by_id(matched_student_id)
                    if student:
                        student_name = student.full_name
            
            if confidence >= threshold:
                return VerifyResponse(
                    success=True,
                    matched=True,
                    message=f"Face matched successfully! Welcome {student_name or matched_student_id}",
                    student_id=matched_student_id,
                    student_name=student_name,
                    confidence=confidence,
                    similar_faces=similar_faces[:3]  # Return top 3 matches
                )
            else:
                return VerifyResponse(
                    success=True,
                    matched=False,
                    message=f"Face not matched. Confidence {confidence:.3f} below threshold {threshold}.",
                    confidence=confidence,
                    similar_faces=similar_faces[:3]
                )
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"Error in verify_student_face: {e}")
        return VerifyResponse(
            success=False,
            matched=False,
            message=f"Verification failed: {str(e)}"
        )
    
@app.get("/students")
async def list_registered_students(student_code_filter: Optional[str] = None):
    """
    Lấy danh sách sinh viên đã đăng ký
    
    Args:
        student_code_filter: Lọc theo mã sinh viên/lớp (optional)
    
    Returns:
        List of registered students
    """
    try:
        with create_student_service() as service:
            students = service.get_all_students(student_code_filter=student_code_filter)
            
            student_list = []
            for student in students:
                student_list.append({
                    "id": student.id,
                    "student_id": student.student_id,
                    "full_name": student.full_name,
                    "email": student.email,
                    "phone": student.phone,
                    "student_code": student.student_code,
                    "image_path": student.image_path,
                    "created_at": student.created_at.isoformat(),
                    "updated_at": student.updated_at.isoformat()
                })
        
        return {
            "success": True,
            "total_students": len(student_list),
            "students": student_list
        }
        
    except Exception as e:
        print(f"Error in list_registered_students: {e}")
        return {
            "success": False,
            "message": f"Failed to retrieve students: {str(e)}"
        }

@app.delete("/student/{student_id}")
async def delete_student(student_id: str):
    """
    Xóa sinh viên khỏi cơ sở dữ liệu và Milvus
    
    Args:
        student_id: ID của sinh viên cần xóa
    
    Returns:
        Deletion result
    """
    try:
        with create_student_service() as service:
            # Get student info before deletion
            student = service.get_student_by_id(student_id)
            if not student:
                return {
                    "success": False,
                    "message": f"Student {student_id} not found."
                }
            
            # Delete from Milvus - search by student_id since we don't store milvus_id
            try:
                # Search for records with this student_id
                results = milvus_client.collection.query(
                    expr=f"student_id == '{student_id}'",
                    output_fields=["id"]
                )
                
                milvus_deleted = 0
                if results:
                    ids_to_delete = [result["id"] for result in results]
                    milvus_client.delete_by_ids(ids_to_delete)
                    milvus_deleted = len(ids_to_delete)
            except Exception as e:
                print(f"Warning: Failed to delete from Milvus: {e}")
                milvus_deleted = 0
            
            # Delete from database
            success = service.delete_student(student_id)
            
            if success:
                return {
                    "success": True,
                    "message": f"Student {student_id} deleted successfully.",
                    "deleted_from_milvus": milvus_deleted > 0,
                    "milvus_records_deleted": milvus_deleted
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to delete student {student_id}."
                }
        
    except Exception as e:
        print(f"Error in delete_student: {e}")
        return {
            "success": False,
            "message": f"Failed to delete student: {str(e)}"
        }

@app.get("/student/{student_id}")
async def get_student_details(student_id: str):
    """
    Lấy thông tin chi tiết của sinh viên
    
    Args:
        student_id: ID của sinh viên
    
    Returns:
        Student details
    """
    try:
        with create_student_service() as service:
            student = service.get_student_by_id(student_id)
            
            if not student:
                return {
                    "success": False,
                    "message": f"Student {student_id} not found."
                }
            
            return {
                "success": True,
                "student": {
                    "id": student.id,
                    "student_id": student.student_id,
                    "full_name": student.full_name,
                    "email": student.email,
                    "phone": student.phone,
                    "student_code": student.student_code,
                    "image_path": student.image_path,
                    "created_at": student.created_at.isoformat(),
                    "updated_at": student.updated_at.isoformat()
                }
            }
        
    except Exception as e:
        print(f"Error in get_student_details: {e}")
        return {
            "success": False,
            "message": f"Failed to get student details: {str(e)}"
        }

# uvicorn main:app --host 0.0.0.0 --port 8001 --reload