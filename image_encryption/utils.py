#utlis.py
import numpy as np
from PIL import Image

def save_encrypted_image(blob, shape, dtype, output_path):
    """Save AES-GCM blob  metadata."""
    np.savez_compressed(output_path,
        blob=blob,
        shape=shape,
        dtype=dtype
    )

def load_encrypted_image(input_path):
    """Load AES-GCM blob + metadata."""
    npz = np.load(input_path, allow_pickle=True)
    blob = npz["blob"].item() if hasattr(npz["blob"], "item") else npz["blob"]
    return blob, tuple(npz["shape"]), npz["dtype"].item()

def visualize_encrypted_image(encrypted_bytes, shape, dtype):
    """Create a visualization of encrypted image data.
    
    Args:
        encrypted_bytes: Encrypted image bytes
        shape: Original image shape
        dtype: Original image dtype
    
    Returns:
        PIL.Image: Visualization of encrypted data
    """
    encrypted_array = np.frombuffer(encrypted_bytes, dtype=dtype)[:np.prod(shape)].reshape(shape)
    return Image.fromarray(encrypted_array)