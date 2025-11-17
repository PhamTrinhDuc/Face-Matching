"""
Database models for student management system
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATABASE_URL = f"sqlite:///{DATABASE_DIR}/students.db"

# Create database directory if it doesn't exist
os.makedirs(DATABASE_DIR, exist_ok=True)

# SQLAlchemy setup
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Student(Base):
    """Student model for storing basic student information"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)  # Auto-generated
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    student_code = Column(String(50), nullable=False)  # Required student code (class/program)
    image_path = Column(String(500), nullable=True)  # Local path to saved image
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Student(student_id='{self.student_id}', name='{self.full_name}')>"

# Create tables
def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

# Database session dependency
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
if __name__ == "__main__":
    create_tables()