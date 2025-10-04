import os, base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image
import numpy as np

# Load a 256-bit key from an environment variable (base64-encoded)
_b64key = os.getenv("IMAGE_ENC_KEY")
if not _b64key:
    raise RuntimeError("Set IMAGE_ENC_KEY to a 32-byte base64 key")
AES_KEY = base64.b64decode(_b64key)

def encrypt_image_pixels(img: Image.Image):
    """
    AEAD-encrypt a PIL Image under AES-GCM.
    Returns (blob: bytes, shape: tuple, dtype: str).
    """
    arr = np.array(img)
    data = arr.tobytes()
    aesgcm = AESGCM(AES_KEY)
    nonce = os.urandom(12)                # 96-bit nonce
    ct = aesgcm.encrypt(nonce, data, None)
    blob = nonce + ct
    return blob, arr.shape, str(arr.dtype)

def decrypt_image_pixels(blob: bytes, shape: tuple, dtype_str: str):
    """
    Decrypt the blob produced by encrypt_image_pixels back into a PIL Image.
    """
    nonce, ct = blob[:12], blob[12:]
    data = AESGCM(AES_KEY).decrypt(nonce, ct, None)
    arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
    return Image.fromarray(arr)

# File-level helpers
def encrypt_image_file(in_path, out_path=None):
    img = Image.open(in_path)
    blob, shape, dtype = encrypt_image_pixels(img)
    if out_path:
        np.savez_compressed(out_path,
            blob=blob, shape=shape, dtype=dtype
        )
    else:
        return blob, shape, dtype

def decrypt_image_file(in_path, out_path=None):
    npz = np.load(in_path, allow_pickle=True)
    blob = npz["blob"].item() if hasattr(npz["blob"], "item") else npz["blob"]
    shape = tuple(npz["shape"])
    dtype = npz["dtype"].item()
    img = decrypt_image_pixels(blob, shape, dtype)
    if out_path:
        img.save(out_path)
    else:
        return img
