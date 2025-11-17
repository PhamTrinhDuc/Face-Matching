# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/extract_feature_v1.py

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from typing import Union, List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from backbone import Backbone
from tqdm import tqdm


class FaceEmbedder:
    """Face embedding extractor that supports both single image and batch processing."""
    
    def __init__(self, model_root: str, input_size: List[int] = [112, 112], embedding_size: int = 512):
        """
        Initialize FaceEmbedder.
        
        Args:
            model_root: Path to the model checkpoint
            input_size: Input image size [height, width]
            embedding_size: Size of the embedding vector
        """
        self.model_root = model_root
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Validate model path
        if not os.path.exists(model_root):
            raise FileNotFoundError(f"Model file not found at {model_root}")
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the face recognition model."""
        print(f"Loading model from: {self.model_root}")
        self.backbone = Backbone(self.input_size)
        self.backbone.load_state_dict(torch.load(self.model_root, map_location=torch.device("cpu")))
        self.backbone.to(self.device)
        self.backbone.eval()
        print(f"Model loaded successfully on device: {self.device}")
    
    def _preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for embedding extraction.
        
        Args:
            image: Image as file path, numpy array, or PIL Image
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            # Load from file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image file not found at {image}")
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError("Image must be a file path, numpy array, or PIL Image")
        
        # Apply transforms
        tensor = self.transform(img)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def embed_single_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract embedding for a single image.
        
        Args:
            image: Image as file path, numpy array, or PIL Image
            
        Returns:
            Normalized embedding vector as numpy array
        """
        with torch.no_grad():
            # Preprocess image
            tensor = self._preprocess_image(image)
            
            # Extract embedding
            embedding = F.normalize(self.backbone(tensor.to(self.device))).cpu().numpy()
            
            return embedding.squeeze()  # Remove batch dimension
    
    def embed_batch_images(self, images: List[Union[str, np.ndarray, Image.Image]], 
                          batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Extract embeddings for a batch of images.
        
        Args:
            images: List of images (file paths, numpy arrays, or PIL Images)
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Normalized embeddings matrix of shape (num_images, embedding_size)
        """
        num_images = len(images)
        embeddings = np.zeros([num_images, self.embedding_size])
        
        with torch.no_grad():
            # Process in batches
            for i in tqdm(range(0, num_images, batch_size), 
                         desc="Extracting embeddings", 
                         disable=not show_progress):
                
                batch_end = min(i + batch_size, num_images)
                batch_images = images[i:batch_end]
                
                # Preprocess batch
                batch_tensors = []
                for img in batch_images:
                    try:
                        tensor = self._preprocess_image(img)
                        batch_tensors.append(tensor)
                    except Exception as e:
                        print(f"Error processing image {img}: {e}")
                        # Use zero embedding for failed images
                        batch_tensors.append(torch.zeros(1, 3, *self.input_size))
                
                if batch_tensors:
                    # Stack tensors and extract embeddings
                    batch_tensor = torch.cat(batch_tensors, dim=0)
                    batch_embeddings = F.normalize(self.backbone(batch_tensor.to(self.device))).cpu().numpy()
                    
                    # Store embeddings
                    embeddings[i:batch_end] = batch_embeddings
        
        return embeddings
    

# Legacy function for backward compatibility
def get_embeddings(data_root, model_root, input_size=[112, 112], embedding_size=512):
    """
    Legacy function for backward compatibility.
    Use FaceEmbedder class for better functionality.
    """
    embedder = FaceEmbedder(model_root, input_size, embedding_size)
    images, embeddings, _ = embedder.embed_from_directory(data_root)
    return images, embeddings


# Example usage
if __name__ == "__main__":
    # Initialize embedder
    model_path = "./checkpoint/backbone_ir50_ms1m_epoch120.pth"
    embedder = FaceEmbedder(model_path)
    
    # Single image embedding
    single_image = "./cropped/0.jpg"
    embedding = embedder.embed_single_image(single_image)
    print(f"Single image embedding shape: {embedding.shape}")
    
    # Batch embedding from list of images
    image_list = ["cropped/0.jpg.jpg", "cropped/1.jpg.jpg", "cropped/2.jpg.jpg"]
    batch_embeddings = embedder.embed_batch_images(image_list, batch_size=16)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")