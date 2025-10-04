# 🔐 BioSecure - Biometric Authentication Framework over IPFS

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-FF4B4B.svg)](https://streamlit.io/)
[![IPFS](https://img.shields.io/badge/IPFS-Enabled-65C2CB.svg)](https://ipfs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A secure, decentralized biometric authentication framework that leverages **IPFS (InterPlanetary File System)** for distributed storage and **AES-GCM encryption** for biometric data protection. This system provides real-time facial recognition with military-grade security and zero-point failure architecture.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## 🎯 Overview

**BioSecure** is a cutting-edge biometric authentication system that combines the power of decentralized storage with advanced encryption to create a highly secure, tamper-proof identity verification solution. By storing encrypted biometric templates on IPFS, the system eliminates single points of failure while ensuring data privacy and integrity.

### Problem Statement

Traditional biometric authentication systems store sensitive biometric data in centralized databases, creating:
- **Single points of failure** - If the central server is compromised, all data is at risk
- **Privacy concerns** - Centralized storage makes data vulnerable to mass breaches
- **Scalability issues** - Central servers can become bottlenecks
- **Trust requirements** - Users must trust a single entity with their biometric data

### Solution

BioSecure addresses these challenges by:
- **Distributing encrypted biometric data** across IPFS nodes
- **Using AES-GCM encryption** to protect biometric templates at rest
- **Enabling decentralized verification** without compromising security
- **Providing tamper-proof storage** through content-addressed storage (CID)

## ✨ Features

### Core Functionality

- 🛡️ **Secure Biometric Storage**
  - Custom AES-GCM encryption for all biometric templates
  - Encrypted data distributed across IPFS network
  - Content-addressable storage for data integrity verification

- 🌐 **Decentralized Architecture**
  - Zero single-point-of-failure design
  - Distributed storage across multiple IPFS nodes
  - Resilient to network failures and attacks

- 🤖 **Real-time Facial Recognition**
  - Live camera feed processing with MTCNN face detection
  - Sub-second identity verification
  - Support for multiple enrolled profiles

- 📊 **Access Logging & Monitoring**
  - Comprehensive verification attempt logging
  - Timestamp tracking for audit trails
  - CSV export for record keeping

- 🖼️ **Profile Management**
  - Easy biometric profile enrollment via image upload
  - View all loaded biometric profiles
  - Support for both encrypted and legacy image formats

- 🎨 **Modern User Interface**
  - Clean, intuitive Streamlit-based web interface
  - Responsive design with real-time updates
  - Apple-inspired modern aesthetics

## 🛠️ Technology Stack

### Backend Technologies

- **Python 3.8+** - Core programming language
- **Streamlit** - Web application framework
- **OpenCV** - Computer vision and image processing
- **face_recognition** - Facial recognition library built on dlib
- **MTCNN** - Multi-task Cascaded Convolutional Networks for face detection
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and CSV logging

### Security & Encryption

- **cryptography** - AES-GCM encryption implementation
- **python-dotenv** - Environment variable management for secrets
- **Custom encryption module** - Specialized image encryption utilities

### Distributed Storage

- **IPFS/Pinata** - Decentralized storage network
- Content-addressable storage with CID (Content Identifier)
- HTTP gateway for seamless data retrieval

### Additional Libraries

- **PIL (Pillow)** - Image processing and manipulation
- **Requests** - HTTP client for IPFS API interactions
- **Threading** - Asynchronous video capture for performance

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                     │
│                    (Streamlit Web App)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                  Application Logic Layer                     │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Biometric      │  │ Encryption      │  │ Camera       │ │
│  │ Processing     │  │ Engine          │  │ Management   │ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              IPFS Distributed Network                   │ │
│  │  [Node 1] ←→ [Node 2] ←→ [Node 3] ←→ [Node N]        │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Enrollment:**
   - User uploads biometric image
   - Image is encrypted using AES-GCM
   - Encrypted data (.npz format) is uploaded to IPFS via Pinata
   - Content Identifier (CID) is returned for future reference

2. **Verification:**
   - Camera captures live video feed
   - MTCNN detects faces in real-time
   - Face encodings are extracted
   - Encrypted profiles are fetched from IPFS and decrypted
   - Comparison is performed against known encodings
   - Access decision is logged

## 📦 Installation

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Webcam** (for real-time verification)
- **IPFS/Pinata Account** (for distributed storage)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AnirudhDattu/BioSecure-Biometric_Authentication_Framework_over_IPFS.git
cd BioSecure-Biometric_Authentication_Framework_over_IPFS
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit==1.30.0
- opencv-python==4.9.0.80
- numpy>=1.26.4
- face-recognition==1.3.0
- pandas==2.2.0
- python-dotenv==1.0.1
- requests==2.31.0
- cryptography>=40.0.0

### Step 3: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
touch .env
```

Add the following environment variables:

```env
# Pinata IPFS Configuration
PINATA_JWT_TOKEN=your_pinata_jwt_token_here

# Image Encryption Key (32-byte base64 encoded)
IMAGE_ENC_KEY=your_base64_encoded_32_byte_key_here
```

**To generate an encryption key:**

```python
import base64
import os

# Generate a 32-byte (256-bit) key
key = os.urandom(32)
b64_key = base64.b64encode(key).decode('utf-8')
print(f"IMAGE_ENC_KEY={b64_key}")
```

**To get a Pinata JWT Token:**
1. Sign up at [Pinata Cloud](https://www.pinata.cloud/)
2. Navigate to API Keys section
3. Create a new API key with pinning permissions
4. Copy the JWT token

### Step 4: Verify Installation

```bash
python -c "import cv2, face_recognition, streamlit; print('All dependencies installed successfully!')"
```

## 🚀 Usage

### Starting the Application

Run the Streamlit application:

```bash
streamlit run IPFS.py
```

The application will open in your default web browser at `http://localhost:8501`

### Navigation

The application has four main sections accessible via the sidebar:

#### 1. **Home** 🏠
- Overview of the system features
- Visual introduction to the platform
- Key capabilities showcase

#### 2. **Biometric ID** 👤

**Enrollment Process:**
1. Click "Browse files" to upload a facial image
2. Supported formats: JPG, JPEG, PNG
3. Image is automatically encrypted using AES-GCM
4. Encrypted profile is uploaded to IPFS
5. Content Identifier (CID) is displayed for reference

**Real-time Verification:**
1. Click "Start System" to activate the camera
2. Face detection runs automatically using MTCNN
3. Detected faces are compared against enrolled profiles
4. Verification status is displayed in real-time
5. Click "Stop System" to deactivate

**Features:**
- Asynchronous video capture for smooth performance
- Frame-by-frame processing with detection intervals
- Visual feedback with bounding boxes
- Identity and status overlay on video feed

#### 3. **Access Logs** 📝
- View all verification attempts
- Columns: Identity, Status, Timestamp
- Export to CSV for external analysis
- Persistent logging across sessions

#### 4. **Loaded Biometric Profiles** 🖼️
- Gallery view of all enrolled profiles
- Display names and truncated CIDs
- Support for both encrypted (.npz) and legacy formats
- Visual confirmation of enrolled identities

### Advanced Usage

#### Using the Encryption Module Standalone

```python
from image_encryption import encrypt_image_file, decrypt_image_file

# Encrypt an image
encrypt_image_file("input.jpg", "encrypted.npz")

# Decrypt the image
decrypt_image_file("encrypted.npz", "decrypted.jpg")
```

#### Pixel-level Encryption

```python
from image_encryption import encrypt_image_pixels, decrypt_image_pixels
from PIL import Image

# Load and encrypt
img = Image.open("input.jpg")
blob, shape, dtype = encrypt_image_pixels(img)

# Decrypt
decrypted_img = decrypt_image_pixels(blob, shape, dtype)
decrypted_img.save("output.jpg")
```

## 🔧 Environment Setup

### Development Environment

1. **Python Virtual Environment (Recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **System Requirements:**
   - **OS:** Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)
   - **RAM:** Minimum 4GB (8GB recommended)
   - **CPU:** Multi-core processor recommended for real-time processing
   - **Webcam:** Any USB or built-in webcam

3. **Optional: GPU Acceleration**
   - For faster face detection, install CUDA and cuDNN
   - Update OpenCV to GPU-enabled version

### Production Deployment

For production deployment, consider:
- Using a process manager like **systemd** or **supervisor**
- Setting up reverse proxy with **nginx**
- Implementing rate limiting for API calls
- Using environment-specific configuration files
- Enabling HTTPS with SSL certificates

## 📁 Project Structure

```
BioSecure-Biometric_Authentication_Framework_over_IPFS/
│
├── IPFS.py                          # Main Streamlit application
├── app2.py                          # Alternative application version
├── pinata.py                        # Pinata IPFS utilities
├── requirements.txt                 # Python dependencies
├── .env                             # Environment variables (create this)
├── .gitignore                       # Git ignore rules
├── access_logs.csv                  # Verification logs (auto-generated)
│
├── image_encryption/                # Encryption module
│   ├── __init__.py                  # Module initialization
│   ├── encryption.py                # Core encryption functions (AES-GCM)
│   ├── utils.py                     # Encryption utilities
│   └── usage.txt                    # Usage examples
│
└── README.md                        # This file
```

### Key Files Description

- **IPFS.py**: Primary application with full feature set including profile viewing
- **app2.py**: Simplified version without profile viewing feature
- **image_encryption/**: Custom encryption module for biometric data
  - Uses AES-GCM (Authenticated Encryption with Associated Data)
  - Provides file-level and pixel-level encryption APIs
- **access_logs.csv**: Stores verification attempts with timestamps
- **.env**: Contains sensitive credentials (not tracked in git)

## 🤝 Contributing

We welcome contributions to BioSecure! Here's how you can help:

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

### Suggesting Features

Feature requests are welcome! Please provide:
- Clear description of the feature
- Use case and benefits
- Any technical considerations

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes:**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation if needed
4. **Test your changes:**
   - Ensure existing functionality still works
   - Test new features thoroughly
5. **Commit with clear messages:**
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
6. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request:**
   - Provide a clear description
   - Reference any related issues
   - Include screenshots for UI changes

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex algorithms

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/BioSecure-Biometric_Authentication_Framework_over_IPFS.git

# Add upstream remote
git remote add upstream https://github.com/AnirudhDattu/BioSecure-Biometric_Authentication_Framework_over_IPFS.git

# Create a feature branch
git checkout -b feature/my-new-feature

# Keep your fork updated
git fetch upstream
git merge upstream/main
```

## 🙏 Acknowledgements

### Libraries and Frameworks

- **[Streamlit](https://streamlit.io/)** - For the amazing web application framework
- **[OpenCV](https://opencv.org/)** - For comprehensive computer vision tools
- **[face_recognition](https://github.com/ageitgey/face_recognition)** - For simple and powerful facial recognition
- **[dlib](http://dlib.net/)** - For machine learning algorithms powering face_recognition
- **[MTCNN](https://github.com/ipazc/mtcnn)** - For robust face detection
- **[Pinata](https://www.pinata.cloud/)** - For IPFS pinning services

### Inspiration

- **IPFS Protocol** - For pioneering decentralized storage solutions
- **Biometric Authentication Research** - For advancing secure identity verification



**Built with ❤️ for a more secure and decentralized future**

*For questions, feedback, or collaboration opportunities, please open an issue or reach out through GitHub.*
