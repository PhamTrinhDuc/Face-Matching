#!/usr/bin/env python3
"""
Initialize database and Milvus collection
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import create_tables
from vectordb import MilvusClient

def init_database():
    """Initialize SQLite database"""
    print("Initializing database...")
    create_tables()
    print("Database initialized successfully!")

def init_milvus():
    """Initialize Milvus collection"""
    print("Initializing Milvus collection...")
    
    client = MilvusClient(
        host="localhost",
        port="19530", 
        collection_name="student_faces"
    )
    
    try:
        client.connect()
        
        try:
            client.create_collection(dim=512)
        except Exception as e:
            print(f"Error creating collection: {e}")
            print("Trying to recreate collection...")
            client.recreate_collection(dim=512)
        
        client.load_collection()
        print("Milvus collection initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing Milvus: {e}")
        raise
    finally:
        try:
            client.disconnect()
        except:
            pass

def main():
    """Initialize all components"""
    print("="*50)
    print("Face Matching System Initialization")
    print("="*50)
    
    # Initialize database
    init_database()
    
    # Initialize Milvus 
    init_milvus()
    
    print("="*50)
    print("Initialization completed successfully!")
    print("You can now start the API server with:")
    print("cd api && uvicorn main:app --host 0.0.0.0 --port 8001 --reload")

if __name__ == "__main__":
    main()