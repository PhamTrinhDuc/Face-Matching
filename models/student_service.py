"""
Student service for database operations
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_
from models.database import Student, get_db, SessionLocal
from PIL import Image
import uuid

# Directory for storing student images
STUDENT_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "student_images")
os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)

class StudentService:
    """Simple service class for student operations"""
    
    def __init__(self):
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def create_student(self, 
                      full_name: str,
                      student_code: str,
                      email: str = None,
                      phone: str = None,
                      image_file = None) -> Student:
        """Create new student record"""
        
        # Generate unique student_id
        student_id = self._generate_student_id(student_code)
        
        # Save image if provided
        image_path = None
        if image_file:
            image_path = self._save_student_image(student_id, image_file)
        
        # Create student record
        student = Student(
            student_id=student_id,
            full_name=full_name,
            email=email,
            phone=phone,
            student_code=student_code,
            image_path=image_path
        )
        
        self.db.add(student)
        self.db.commit()
        self.db.refresh(student)
        
        return student
    
    def get_student_by_id(self, student_id: str) -> Optional[Student]:
        """Get student by student ID"""
        return self.db.query(Student).filter(
            Student.student_id == student_id
        ).first()
    
    def get_all_students(self, student_code_filter: str = None) -> List[Student]:
        """Get all students with optional filters"""
        query = self.db.query(Student)
        
        if student_code_filter:
            query = query.filter(Student.student_code == student_code_filter)
        
        return query.order_by(Student.created_at.desc()).all()
    
    def update_student(self, 
                      student_id: str,
                      **kwargs) -> Optional[Student]:
        """Update student information"""
        student = self.get_student_by_id(student_id)
        if not student:
            return None
        
        # Handle image update
        if 'image_file' in kwargs and kwargs['image_file']:
            # Delete old image
            if student.image_path and os.path.exists(student.image_path):
                os.remove(student.image_path)
            
            # Save new image
            kwargs['image_path'] = self._save_student_image(student_id, kwargs['image_file'])
            del kwargs['image_file']
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(student, key):
                setattr(student, key, value)
        
        student.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(student)
        
        return student
    
    def delete_student(self, student_id: str) -> bool:
        """Delete student"""
        student = self.get_student_by_id(student_id)
        if not student:
            return False
        
        # Delete image file
        if student.image_path and os.path.exists(student.image_path):
            os.remove(student.image_path)
        
        # Delete from database
        self.db.delete(student)
        self.db.commit()
        return True
    
    def search_students(self, 
                       query: str,
                       fields: List[str] = None) -> List[Student]:
        """Search students by text query"""
        if fields is None:
            fields = ['student_id', 'full_name', 'email', 'student_code']
        
        db_query = self.db.query(Student)
        
        # Build search conditions
        conditions = []
        for field in fields:
            if hasattr(Student, field):
                column = getattr(Student, field)
                conditions.append(column.like(f"%{query}%"))
        
        if conditions:
            db_query = db_query.filter(or_(*conditions))
        
        return db_query.all()
    
    def get_student_stats(self) -> Dict[str, Any]:
        """Get simple student statistics"""
        total_students = self.db.query(Student).count()
        
        student_codes = self.db.query(Student.student_code).filter(
            Student.student_code.isnot(None)
        ).distinct().all()
        
        return {
            "total_students": total_students,
            "total_student_codes": len(student_codes),
            "student_codes": [c[0] for c in student_codes]
        }
    
    def _generate_student_id(self, student_code: str) -> str:
        """Generate unique student ID based on student_code and timestamp"""
        import time
        timestamp = str(int(time.time()))[-6:]  # Last 6 digits of timestamp
        random_suffix = uuid.uuid4().hex[:4].upper()  # 4 random chars
        
        # Format: STUDENTCODE_TIMESTAMP_RANDOM (e.g., CNTT2025_123456_A1B2)
        base_id = f"{student_code}_{timestamp}_{random_suffix}"
        
        # Ensure uniqueness by checking database
        counter = 1
        student_id = base_id
        while self.db.query(Student).filter(Student.student_id == student_id).first():
            student_id = f"{base_id}_{counter:02d}"
            counter += 1
        
        return student_id
    
    def _save_student_image(self, student_id: str, image_file, filename: str = None) -> str:
        """Save student image to local directory"""
        # Generate unique filename
        if filename:
            file_extension = os.path.splitext(filename)[1]
            base_filename = os.path.splitext(filename)[0]
        else:
            file_extension = '.jpg'
            base_filename = 'image'
        
        unique_filename = f"{student_id}_{uuid.uuid4().hex[:8]}{file_extension}"
        file_path = os.path.join(STUDENT_IMAGES_DIR, unique_filename)
        
        # Save image
        if isinstance(image_file, bytes):
            # Bytes data from file.read()
            with open(file_path, "wb") as f:
                f.write(image_file)
        elif hasattr(image_file, 'save'):
            # FastAPI UploadFile
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
        elif hasattr(image_file, 'read'):
            # File-like object
            with open(file_path, "wb") as f:
                f.write(image_file.read())
        elif isinstance(image_file, str):
            # File path
            shutil.copy2(image_file, file_path)
        else:
            raise ValueError("Unsupported image file type")
        
        return file_path

# Convenience functions
def create_student_service() -> StudentService:
    """Create student service instance"""
    return StudentService()

if __name__ == "__main__":
    # Test the service
    from models.database import create_tables
    
    create_tables()
    
    with create_student_service() as service:
        print("Student service test completed!")