#__init__.py
"""
Image Encryption Module

Provides secure encryption/decryption of images using Armstrong number-based CBC encryption.
"""

from .encryption import (
    encrypt_image_file,
    decrypt_image_file,
    encrypt_image_pixels,
    decrypt_image_pixels
)

from .utils import save_encrypted_image, load_encrypted_image

__all__ = [
    'encrypt_image_file',
    'decrypt_image_file',
    'encrypt_image_pixels',
    'decrypt_image_pixels',
    'save_encrypted_image',
    'load_encrypted_image'
]

__version__ = "1.0.0"