
#IPFS.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ─── Force DirectShow (and disable MSMF) ───
# os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"]  = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
# os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "5"

import cv2
# ─── Silence all OpenCV logs except real errors ───
cv2.setLogLevel(0)

from PIL import ImageOps
import streamlit as st
from datetime import datetime
import cv2
import numpy as np
import face_recognition
import os
import textwrap
import pandas as pd
import requests
from dotenv import load_dotenv
from mtcnn import MTCNN
import threading
import io
import numpy as np
from PIL import Image
from image_encryption import encrypt_image_pixels, decrypt_image_pixels


# Asynchronous Video Capture Class
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        # Set desired camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()
    
    def start(self):
        if self.started:
            print("[Warning] Video capture already started")
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self
    
    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
    
    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame
    
    def stop(self):
        self.started = False
        self.thread.join()
        self.cap.release()

# Load environment variables
load_dotenv()

PINATA_JWT_TOKEN = os.getenv('PINATA_JWT_TOKEN')
PINATA_API_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_GATEWAY = "https://gateway.pinata.cloud/ipfs/"

# Streamlit configuration
st.set_page_config(
    page_title="BioSecure",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS
st.markdown("""
    <style>
        :root {
            --primary-color: #1a1a1a;
            --secondary-color: #2d2d2d;
            --accent-blue: #0071e3;
            --light-gray: #f5f5f7;
            --text-dark: #1d1d1f;
            --text-light: #86868b;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: white;
        }
        .stApp {
            max-width: 1600px;
            padding: 0 40px;
        }
        h1 {
            font-size: 4rem;
            font-weight: 700;
            color: var(--text-dark);
            margin: 2rem 0;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
# PINATA_JWT_TOKEN = os.getenv('PINATA_JWT_TOKEN')
# IPFS_GATEWAY = "https://gateway.pinata.cloud/ipfs/"
ACCESS_LOG = "access_logs.csv"

# Session state initialization
if 'access_logs' not in st.session_state:
    st.session_state.access_logs = []
if 'system_active' not in st.session_state:
    st.session_state.system_active = False

# Navigation
nav_options = ["Home", "Biometric ID", "Access Logs", "About", "Loaded Biometric Profiles"]
selected_nav = st.sidebar.radio("Navigation", nav_options)

def fetch_ipfs_data(cid):
    """Retrieve biometric data from IPFS"""
    try:
        response = requests.get(f"{PINATA_GATEWAY}{cid}", timeout=10)
        response.raise_for_status()
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        st.error(f"Data retrieval error: {str(e)}")
        return None

def fetch_and_decrypt(cid):
    """Download the encrypted .npz from IPFS and decrypt back to a BGR image."""
    resp = requests.get(f"{PINATA_GATEWAY}{cid}", timeout=10)
    resp.raise_for_status()

    # Load everything from the .npz
    data = np.load(io.BytesIO(resp.content), allow_pickle=True)
    
    # 1️⃣ Encrypted bytes → Python bytes
    arr = data["encrypted_bytes"]
    encrypted_bytes = arr.tobytes() if isinstance(arr, np.ndarray) else arr

    # 2️⃣ Shape & dtype
    shape = tuple(data["shape"])
    dtype = np.dtype(data["dtype"].item())

    # 3️⃣ IV (scalar) → Python int
    iv = int(data["iv"].item())

    # 4️⃣ Key sequence → Python list/bytearray
    raw_seq = data["key_sequence"]
    key_seq = raw_seq.tolist() if isinstance(raw_seq, np.ndarray) else raw_seq

    # 5️⃣ Decrypt and return a BGR image
    pil = decrypt_image_pixels(blob, shape, dtype)
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)



def load_biometric_profiles():
    """Load and encode biometric profiles from decentralized storage"""
    try:
        headers = {'Authorization': f'Bearer {PINATA_JWT_TOKEN}'}
        params = {"status":"pinned"}
        # querystring = {"status":"pinned"}
        response = requests.get("https://api.pinata.cloud/data/pinList", headers=headers, params=params)
        response.raise_for_status()

        encodings, identifiers = [], []
        for item in response.json().get('rows', []):
            name = item['metadata']['name'].lower()
            cid  = item['ipfs_pin_hash']

            # 1️⃣ Encrypted .npz?
            if name.endswith('.npz'):
                img = fetch_and_decrypt(cid)

            # 2️⃣ Legacy raw image?
            elif name.endswith(('.png', '.jpg', '.jpeg')):
                img = fetch_ipfs_data(cid)

            else:
                continue  # skip any other file types

            if img is None:
                continue

            # downstream: face_recognition wants RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_enc = face_recognition.face_encodings(rgb_img)
            if face_enc:
                encodings.append(face_enc[0])
                identifiers.append(os.path.splitext(item['metadata']['name'])[0])

        return encodings, identifiers

    except Exception as e:
        st.error(f"Biometric load error: {str(e)}")
        return [], []


def show_loaded_faces():
    """Display all biometric profiles loaded from IPFS with correct orientation."""
    try:
        headers = {'Authorization': f'Bearer {PINATA_JWT_TOKEN}'}
        params = {"status":"pinned"}
        response = requests.get("https://api.pinata.cloud/data/pinList", headers=headers, params=params)
        response.raise_for_status()

        st.markdown("### Loaded Biometric Profiles")
        if not response.json().get('rows', []):
            st.info("No biometric profiles found in IPFS storage.")
            return

        cols = st.columns(3)  # Display in a grid of 3 columns
        for idx, item in enumerate(response.json().get('rows', [])):
            name = item['metadata']['name'].lower()
            cid = item['ipfs_pin_hash']

            # Handle encrypted .npz files
            if name.endswith('.npz'):
                resp = requests.get(f"{PINATA_GATEWAY}{cid}", timeout=10)
                resp.raise_for_status()
                data = np.load(io.BytesIO(resp.content), allow_pickle=True)
                
                # Decrypt the image
                arr = data["encrypted_bytes"]
                encrypted_bytes = arr.tobytes() if isinstance(arr, np.ndarray) else arr
                shape = tuple(data["shape"])
                dtype = np.dtype(data["dtype"].item())
                iv = int(data["iv"].item())
                raw_seq = data["key_sequence"]
                key_seq = raw_seq.tolist() if isinstance(raw_seq, np.ndarray) else raw_seq
                
                pil_img = decrypt_image_pixels(encrypted_bytes, shape, dtype, iv, key_seq)
                # Ensure correct orientation (in case decryption doesn't preserve it)
                pil_img = ImageOps.exif_transpose(pil_img)  # Apply EXIF rotation if any
                
            # Handle raw images
            elif name.endswith(('.png', '.jpg', '.jpeg')):
                resp = requests.get(f"{PINATA_GATEWAY}{cid}", timeout=10)
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content))
                pil_img = ImageOps.exif_transpose(pil_img)  # Apply EXIF rotation
            
            else:
                continue  # Skip unsupported file types

            if pil_img is not None:
                with cols[idx % 3]:
                    # Convert PIL image to RGB for Streamlit (no BGR)
                    img_rgb = pil_img.convert("RGB")
                    st.image(img_rgb, caption=os.path.splitext(item['metadata']['name'])[0], use_container_width=True)
                    st.write(f"CID: {cid[:10]}...{cid[-10:]}")

    except Exception as e:
        st.error(f"Error loading profiles: {str(e)}")

def initialize_camera():
    """Camera setup with asynchronous video capture and higher resolution"""
    try:
        # Try device indexes 0 and 1; adjust as necessary. Here we use higher resolution.
        for index in [0, 1]:
            for api in [cv2.CAP_ANY, cv2.CAP_DSHOW]:
                cap = VideoCaptureAsync(index, width=1280, height=720)
                grabbed, _ = cap.read()
                if grabbed:
                    return cap.start()
        return None
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        return None

def log_verification(identity, status):
    """Record verification attempts in decentralized log"""
    entry = {
        "Identity": identity,
        "Status": status,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if entry not in st.session_state.access_logs:
        st.session_state.access_logs.append(entry)
        pd.DataFrame(st.session_state.access_logs).to_csv(ACCESS_LOG, index=False)

# --------------------- Navigation: Home ---------------------
if selected_nav == "Home":
    st.markdown("""
    <style>
        .hero-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
            padding: 4rem 2rem;
            border-radius: 30px;
            margin: 2rem 0;
            text-align: center;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 20px !important;
            padding: 2rem !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1) !important;
            margin: 1rem 0 !important;
            transition: transform 0.2s ease !important;
        }
        .feature-card:hover {
            transform: translateY(-5px) !important;
        }
        .feature-icon {
            font-size: 2.5rem !important;
            margin-bottom: 1.5rem !important;
            background: linear-gradient(45deg, #0071e3, #00A6FF) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }
    </style>
    <div class="hero-section">
        <h1 style="color: white; font-size: 3.5rem; margin-bottom: 1rem;">
            Decentralized Biometric Security
        </h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
            IPFS-Powered Identity Verification System
        </p>
    </div>
    """, unsafe_allow_html=True)
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🛡</div>
                <h3 style="color: #1d1d1f; margin-bottom: 1rem;">Secure Storage</h3>
                <p style="color: #86868b; line-height: 1.6;">
                    Custom encrypted biometric templates distributed across IPFS nodes
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🌐</div>
                <h3 style="color: #1d1d1f; margin-bottom: 1rem;">Distributed Network</h3>
                <p style="color: #86868b; line-height: 1.6;">
                    Zero-point failure architecture with decentralized storage
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3 style="color: #1d1d1f; margin-bottom: 1rem;">Instant Auth</h3>
                <p style="color: #86868b; line-height: 1.6;">
                    Real-time identity confirmation using edge processing
                </p>
            </div>
            """, unsafe_allow_html=True)


elif selected_nav == "Loaded Biometric Profiles":
    st.markdown("<h1 style='text-align:center;'>Loaded Biometric Profiles</h1>", unsafe_allow_html=True) 
    show_loaded_faces()


# --------------------- Navigation: Biometric ID ---------------------
elif selected_nav == "Biometric ID":
    st.markdown("<h1 style='font-size: 3rem!important;'>Identity Verification</h1>", unsafe_allow_html=True)
    # show_loaded_faces()
    with st.sidebar:
        st.header("System Controls")
        if st.button("Activate Scanner") and not st.session_state.system_active:
            st.session_state.system_active = True
        if st.button("Deactivate Scanner") and st.session_state.system_active:
            st.session_state.system_active = False
        st.divider()
        st.header("Add New Profile")
        uploaded_image = st.file_uploader("Upload Facial Profile", 
                                        type=["jpg", "jpeg", "png"],
                                        help="Upload clear frontal face image for biometric registration")
        if uploaded_image:
            try:
                # 0️⃣ Stop the live camera capture so we can repurpose the device
                if 'cam' in st.session_state:
                    st.session_state.cam.stop()
                    del st.session_state.cam

                # 1️⃣ Read the uploaded file into a PIL Image and fix orientation
                img = Image.open(io.BytesIO(uploaded_image.getvalue()))
                img = ImageOps.exif_transpose(img)  # Apply EXIF rotation to normalize orientation

                # 2️⃣ Encrypt using your custom model
                blob, shape, dtype = encrypt_image_pixels(img)

                # 3️⃣ Pack ciphertext + metadata into an in-memory .npz
                buf = io.BytesIO()
                np.savez_compressed(
                    buf,
                    blob=blob,
                    shape=shape,
                    dtype=  dtype, #str(dtype),
                )
                buf.seek(0)

                # 4️⃣ Upload the encrypted .npz to Pinata
                headers = {"Authorization": f"Bearer {PINATA_JWT_TOKEN}"}
                files = {
                    "file": (
                        f"{os.path.splitext(uploaded_image.name)[0]}.npz",
                        buf,
                        "application/octet-stream"
                    )
                }
                response = requests.post(PINATA_API_URL, headers=headers, files=files)
                response.raise_for_status()

                st.success("Encrypted profile uploaded to IPFS!")
                st.code(f"Encrypted IPFS CID: {response.json()['IpfsHash']}")

                # 5️⃣ Restart the camera for live scanning
                st.session_state.cam = initialize_camera()

            except Exception as e:
                st.error(f"Upload/encryption error: {e}")
                        
    @st.cache_resource
    def get_biometric_profiles():
        return load_biometric_profiles()
    
    known_encodings, known_identities = get_biometric_profiles()
    
    # Initialize camera (asynchronous)
    if 'cam' not in st.session_state:
        st.session_state.cam = initialize_camera()
    frame_placeholder = st.empty()
    
    # Initialize MTCNN detector
    detector = MTCNN()
    frame_counter = 0
    detection_interval = 5  # Process detection every 2nd frame
    
    if st.session_state.cam and st.session_state.system_active:
        try:
            last_detections = []
            while st.session_state.system_active:
                grabbed, frame = st.session_state.cam.read()
                if not grabbed:
                    st.error("Frame capture issue")
                    break
                frame_counter += 1
                # Process the frame (we do not downscale here to preserve quality)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_small = cv2.resize(rgb_full, (0, 0), fx=0.25, fy=0.25)
                # run detector on the tiny frame
                detections = detector.detect_faces(rgb_small)
                # then for each detection:
                # x, y, w, h = detection['box']
                # scale back up by 4×
                # x, y, w, h = x*4, y*4, w*4, h*4
                
                try:
                    if frame_counter % detection_interval == detection_interval - 1:
                        # run detection every Nth frame
                        detections = detector.detect_faces(rgb_frame)
                        last_detections = detections
                    else:
                        detections = last_detections
                    for detection in detections:
                         x, y, w, h = detection['box']

                except Exception as e:
                    # MTCNN sometimes throws a TF shape-mismatch when no faces are found;
                    # just reuse the previous detections and move on.
                    detections = last_detections


                for detection in detections:
                    x, y, w, h = detection['box']
                    face_location = (y, x + w, y + h, x)
                    encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=[face_location])
                    identity = "Unknown"
                    status = "Unauthorized"
                    if encodings:
                        encoding = encodings[0]
                        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                        if True in matches:
                            face_distances = face_recognition.face_distance(known_encodings, encoding)
                            best_match = np.argmin(face_distances)
                            identity = known_identities[best_match]
                            status = "Verified"
                        log_verification(identity, status)
                    color = (0, 255, 0) if status == "Verified" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{identity} ({status})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                
            st.session_state.cam.stop()
            del st.session_state.cam
        except Exception as e:
            st.error(f"System error: {str(e)}")
            if 'cam' in st.session_state:
                st.session_state.cam.stop()
                del st.session_state.cam

# --------------------- Navigation: Access Logs ---------------------
elif selected_nav == "Access Logs":
    st.markdown("<h1>Verification Logs</h1>", unsafe_allow_html=True)
    try:
        df = pd.read_csv(ACCESS_LOG)
        if not df.empty:
            st.dataframe(
                df.style.map(lambda x: "background-color: #e6f4ea" if x == "Verified" else "background-color: #fce8e6",
                             subset=["Status"]),
                use_container_width=True
            )
            if st.button("Clear Historical Data"):
                os.remove(ACCESS_LOG)
                st.session_state.access_logs = []
                st.rerun()
        else:
            st.info("No verification records available")
    except FileNotFoundError:
        st.info("No access logs found")

# --------------------- Navigation: About ---------------------
elif selected_nav == "About":
    st.markdown("""
    <style>
        .about-container {
            background: white;
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            margin: 2rem 0;
        }
        .tech-card {
            padding: 1.5rem;
            background: #f5f5f7;
            border-radius: 16px;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        }
        .tech-icon {
            font-size: 24px;
            margin-right: 1rem;
            color: #0071e3;
        }
    </style>
    """, unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="about-container">', unsafe_allow_html=True)
        st.markdown("""
        <h1 style='
            color: #0071e3;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            font-weight: 700;
            text-align: center;
        '>
            SYSTEM ARCHITECTURE
        </h1>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="tech-card">
                <span class="tech-icon">🌐</span>
                <div>
                    <h3 style='color: #1a1a1a; margin: 0 0 0.4rem 0; font-weight: 600;'>
                        IPFS Network
                    </h3>
                    <p style='color: #6e6e73; margin: 0; line-height: 1.5;'>
                        Distributed biometric storage across decentralized nodes
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="tech-card">
                <span class="tech-icon">🔒</span>
                <div>
                    <h3 style='color: #1a1a1a; margin: 0 0 0.4rem 0; font-weight: 600;'>
                        Secure Encryption
                    </h3>
                    <p style='color: #6e6e73; margin: 0; line-height: 1.5;'>
                        Custom encryption model for biometric data protection
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="tech-card">
                <span class="tech-icon">🤖</span>
                <div>
                    <h3 style='color: #1a1a1a; margin: 0 0 0.4rem 0; font-weight: 600;'>
                        Edge Processing
                    </h3>
                    <p style='color: #6e6e73; margin: 0; line-height: 1.5;'>
                        On-device facial recognition processing
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="tech-card">
                <span class="tech-icon">⚡</span>
                <div>
                    <h3 style='color: #1a1a1a; margin: 0 0 0.4rem 0; font-weight: 600;'>
                        Fast Verification
                    </h3>
                    <p style='color: #6e6e73; margin: 0; line-height: 1.5;'>
                        Sub-second identity confirmation
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --------------------- Footer ---------------------
st.markdown("""
    <div style='text-align: center; color: var(--text-light); margin: 4rem 0 2rem;'>
        🔒 Secure Biometric Storage System | IPFS-Powered Identity Verification
    </div>
""", unsafe_allow_html=True)