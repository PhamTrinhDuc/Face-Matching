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
    super().__init__(host, port, collection_name)
    self.alias = "default"

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
    
  def create_collection(self, dim: int=512, description:str="Image embeddings collection"):
    """
    Create a collection for storing image embeddings
    
    Args:
        dim: Dimension of the embedding vectors
        description: Collection description
    """
    try: 
      if utility.has_collection(self.collection_name):
          print(f"Collection '{self.collection_name}' already exists. Reloading it")
          self.collection = Collection(self.collection_name)
          return

      fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), 
        FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500), 
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim), 
        FieldSchema(name="schema", dtype=DataType.VARCHAR, max_length=1000)
      ]
      schema = CollectionSchema(fields=fields, description=description)
      self.collection = Collection(name=self.collection_name, schema=schema)

      index_params = {
        "metric_type": "COSINE",  # Changed from L2 to COSINE
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
    
  def insert_embeddings(self, image_paths: List[str], embeddings: np.ndarray, metadata = None):
    """
    Insert image embeddings into collection
    
    Args:
        image_paths: List of image file paths
        embeddings: Numpy array of embeddings
        metadata: Optional metadata for each image
        
    Returns:
        List of inserted IDs
    """
    try: 
      if metadata is None: 
        metadata = [""] * len(image_paths)

      data = [
        image_paths, 
        embeddings.tolist(), 
        metadata
      ]

      mr = self.collection.insert(data=data)
      self.collection.flush()

      print(f"Inserted {len(image_paths)} embeddings")
      return mr.primary_keys
            
    except Exception as e:
        print(f"Failed to insert embeddings: {e}")
        raise
    
  def search_similar(self, 
                     query_embedding: np.ndarray, 
                     top_k: int = 10, 
                     search_params: Dict = None) -> List[Dict]:
    """
    Search for similar images
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of similar images to return
        search_params: Search parameters
        
    Returns:
        List of search results with scores and metadata
    """
    try:
      if search_params is None:
          search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
      
      results = self.collection.search(
          data=[query_embedding.tolist()],
          anns_field="embedding",
          param=search_params,
          limit=top_k,
          output_fields=["image_path", "schema"]  # Fixed: "metadata" -> "schema" to match your field definition
      )
      
      # Format results
      formatted_results = []
      for hits in results:
        for hit in hits:
          formatted_results.append({
              "id": hit.id,
              "distance": hit.distance,
              "score": hit.score,
              "image_path": hit.entity.get("image_path"),
              "schema": hit.entity.get("schema")  # Fixed: "metadata" -> "schema"
          })
      
      print(f"Found {len(formatted_results)} similar images")
      return formatted_results
      
    except Exception as e:
        print(f"Failed to search similar images: {e}")
        raise
      
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
   client = MilvusClient(host="localhost", port=19530, collection_name="iot_face_matching")
   client.connect()
   client.create_collection()
   client.load_collection()
   result = client.search_similar()
   