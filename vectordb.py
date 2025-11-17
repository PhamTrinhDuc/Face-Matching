import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import numpy as np
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility, exceptions
from typing import List, Dict, Any

class MilvusClient: 
  def __init__(self, host: str, port: str, collection_name: str):
    """
    Initialize Milvus client
    
    Args:
        host: Milvus server host
        port: Milvus server port  
        collection_name: Name of the collection to work with
    """
    self.host = host
    self.port = port
    self.collection_name = collection_name
    self.alias = "default"
    self.collection = None

  def connect(self):
    try: 
      connections.connect(alias=self.alias, host=self.host, port=self.port)
      print(f"Connected to Milvus at {self.host}:{self.port}")
      return True
    except exceptions.ConnectError as ce:
      print(f"Connection timeout or refused: {ce}")
      raise Exception(f"Connection timeout or refused: {ce}")
    except Exception as e: 
      print(f"Failed to connect to Milvus: {e}")
      raise Exception(f"Failed to connect to Milvus: {e}")
    
  def create_collection(self, dim: int=512, description:str="Student face embeddings collection"):
    """
    Create a collection for storing student face embeddings
    
    Args:
        dim: Dimension of the embedding vectors (default: 512)
        description: Collection description
    """
    try: 
      if utility.has_collection(self.collection_name):
          print(f"Collection '{self.collection_name}' already exists. Checking index...")
          self.collection = Collection(self.collection_name)
          
          # Check if index exists and is compatible
          try:
              indexes = self.collection.indexes
              has_valid_index = False
              
              for index in indexes:
                  if index.field_name == "embedding":
                      print(f"Found existing index on embedding field")
                      has_valid_index = True
                      break
              
              if not has_valid_index:
                  print("No valid index found. Creating new index...")
                  index_params = {
                      "metric_type": "IP",
                      "index_type": "IVF_FLAT", 
                      "params": {"nlist": 128}
                  }
                  self.collection.create_index(field_name="embedding", index_params=index_params)
                  print("Index created successfully")
              
              # Try to load the collection
              self.collection.load()
              print(f"Collection '{self.collection_name}' loaded successfully")
              return
              
          except Exception as index_error:
              print(f"Error with existing collection: {index_error}")
              print("Dropping and recreating collection...")
              utility.drop_collection(self.collection_name)
              # Continue to create new collection below

      # Define schema fields for student faces
      fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
        FieldSchema(name="student_id", dtype=DataType.VARCHAR, max_length=50),  # Student ID
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),  # Image file path
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),     # Face embedding vector
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000)   # Additional metadata (JSON string)
      ]
      schema = CollectionSchema(fields=fields, description=description)
      self.collection = Collection(name=self.collection_name, schema=schema)

      index_params = {
        "metric_type": "IP",  # Inner Product for cosine similarity (after normalization)
        "index_type": "IVF_FLAT", 
        "params": {"nlist": 128}
      }

      self.collection.create_index(field_name="embedding", index_params=index_params)
      print(f"Collection '{self.collection_name}' created successfully")
    except Exception as e:
      print(f"Failed to create collection: {e}")
      raise

  def load_collection(self):
    """Load collection into memory"""
    try:
      if not self.collection:
          self.collection = Collection(self.collection_name)
      self.collection.load()
      print(f"Collection '{self.collection_name}' loaded into memory")
    except Exception as e:
        print(f"Failed to load collection: {e}")
        raise
    
  def insert_student_embedding(self, 
                             student_id: str, 
                             image_path: str, 
                             embedding: np.ndarray, 
                             metadata: str = None) -> List[int]:
    """
    Insert student face embedding into collection
    
    Args:
        student_id: Student ID
        image_path: Path to student image file
        embedding: Face embedding vector (numpy array)
        metadata: Additional metadata as JSON string
        
    Returns:
        List of inserted IDs
    """
    try:
      # Ensure embedding is 1D and float32
      if len(embedding.shape) == 2:
          embedding = embedding.flatten()
      embedding = embedding.astype(np.float32)
      
      # Prepare data for insertion
      # Format: list of [list_for_field1, list_for_field2, ...]
      data = [
          [str(student_id)],           # student_id field
          [str(image_path)],           # image_path field  
          [embedding.tolist()],        # embedding field - list of one embedding vector
          [str(metadata or "")]        # metadata field
      ]
      
      # Insert data
      mr = self.collection.insert(data)
      
      print(f"Successfully inserted embedding for student {student_id}")
      return mr.primary_keys
      
    except Exception as e:
        print(f"Error inserting student embedding: {e}")
        raise e

  def search_similar_students(self, 
                            query_embedding: np.ndarray, 
                            top_k: int = 5, 
                            search_params: Dict = None) -> List[Dict]:
    """
    Search for similar student faces
    
    Args:
        query_embedding: Query face embedding vector
        top_k: Number of similar faces to return (default: 5)
        search_params: Search parameters for Milvus
        
    Returns:
        List of search results with student info and similarity scores
    """
    try:
      # Ensure embedding is 2D
      if len(query_embedding.shape) == 1:
          query_embedding = query_embedding.reshape(1, -1)
      
      # Default search parameters
      if search_params is None:
          search_params = {
              "metric_type": "IP",
              "params": {"nprobe": 16}
          }
      
      # Perform search
      results = self.collection.search(
          data=query_embedding.tolist(),
          anns_field="embedding",
          param=search_params,
          limit=top_k,
          output_fields=["student_id", "image_path", "metadata"]
      )
      
      # Format results
      formatted_results = []
      if results and len(results) > 0:
          for hit in results[0]:  # results[0] for first query
              formatted_results.append({
                  "id": hit.id,
                  "student_id": hit.entity.get("student_id"),
                  "image_path": hit.entity.get("image_path"), 
                  "metadata": hit.entity.get("metadata"),
                  "distance": hit.distance,  # Cosine similarity score
                  "confidence": float(hit.distance)  # Alias for distance
              })
      
      return formatted_results
      
    except Exception as e:
        print(f"Error searching similar students: {e}")
        return []
   
  def delete_by_ids(self, ids: List[int]):
    """Delete embeddings by IDs"""
    try:
      expr = f"id in {ids}"
      self.collection.delete(expr)
      self.collection.flush()
      print(f"Deleted {len(ids)} embeddings")
    except Exception as e:
      print(f"Failed to delete embeddings: {e}")
      raise
  
  def get_embedding_by_id(self, id: int): 
      try:
          expr = f"id == {id}"
          results = self.collection.query(expr=expr, output_fields=["embedding"])
          if results:
              return np.array(results[0]["embedding"])
          else:
              print(f"No embedding found for ID {id}")
              return None
      except Exception as e:
          print(f"Failed to get embedding by ID: {e}")
          raise

  def get_collection_stats(self) -> Dict[str, Any]:
    """Get collection statistics"""
    try:
      # Get basic collection info
      num_entities = self.collection.num_entities
      
      # Get collection description and schema info
      collection_info = {
          "num_entities": num_entities,
          "collection_name": self.collection_name,
          "description": self.collection.description if hasattr(self.collection, 'description') else "N/A"
      }
      
      # Try to get additional stats if available
      try:
          # Get index information
          indexes = self.collection.indexes
          collection_info["indexes"] = [{"field_name": idx.field_name, "index_type": idx.params.get("index_type", "N/A")} for idx in indexes]
      except Exception:
          collection_info["indexes"] = []
      
      # Try to get memory usage if available  
      try:
          stats = utility.get_query_segment_info(self.collection_name)
          collection_info["segments"] = len(stats) if stats else 0
      except Exception:
          collection_info["segments"] = 0
      
      # Check if collection is loaded
      try:
          collection_info["is_loaded"] = utility.loading_progress(self.collection_name)
      except Exception:
          collection_info["is_loaded"] = "Unknown"

      collection_info.update(self.collection.describe())

      return collection_info
      
    except Exception as e:
      print(f"Failed to get collection stats: {e}")
      return {
          "num_entities": 0,
          "collection_name": self.collection_name,
          "error": str(e)
      }
    
  def drop_collection(self):
    """Drop the collection"""
    try:
      utility.drop_collection(self.collection_name)
      print(f"Collection '{self.collection_name}' dropped")
    except Exception as e:
      print(f"Failed to drop collection: {e}")
      raise

  def recreate_collection(self, dim: int=512, description:str="Student face embeddings collection"):
    """Force drop and recreate collection"""
    try:
      if utility.has_collection(self.collection_name):
          print(f"Dropping existing collection '{self.collection_name}'...")
          utility.drop_collection(self.collection_name)
      
      self.create_collection(dim, description)
      print(f"Collection '{self.collection_name}' recreated successfully")
    except Exception as e:
      print(f"Failed to recreate collection: {e}")
      raise
  
  def disconnect(self):
    """Disconnect from Milvus server"""
    try:
      connections.disconnect(alias=self.alias)
      print("Disconnected from Milvus")
    except Exception as e:
      print(f"Error disconnecting from Milvus: {e}")
      raise Exception(f"Error disconnecting from Milvus: {e}")

  def list_collections(self) -> List[str]:
    """List all collections in Milvus"""
    try:
      collections = utility.list_collections()
      print(f"Found {len(collections)} collections")
      return collections
    except Exception as e:
      print(f"Failed to list collections: {e}")
      return []
    
  def __enter__(self):
    """Context manager entry"""
    self.connect()
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit"""
    self.disconnect()

if __name__ == "__main__": 
   client = MilvusClient(host="localhost", port=19530, collection_name="student_faces")
   client.connect()
   client.drop_collection()
  #  client.create_collection()
  #  client.load_collection()
  #  result = client.search_similar()
   