# AetherAttend
# 📸 AetherAttend: Deep Learning Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![dlib](https://img.shields.io/badge/dlib-ResNet--128D-blue?style=for-the-badge)
![Tkinter](https://img.shields.io/badge/UI-Tkinter-ff69b4?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A high-performance, multi-threaded desktop attendance system utilizing dlib's state-of-the-art deep learning face recognition models and OpenCV.*

[Features](#-key-features--technical-innovations) • [Architecture](#-system-architecture--data-flow) • [Installation](#-installation--setup) • [Dependencies](#-libraries--external-repositories)

</div>

---

## 🖼️ Application Preview

> **Interface Design:** Built with a modern slate-blue Tkinter theme, featuring real-time bounding box overlays, active frame statistics, and an administrative control panel.

*(Tip: Add a GIF or screenshot of your running application here! Save it as `assets/demo.png` and replace this line)*

---

## 🚀 Key Features & Technical Innovations

### 1. High-Speed Decoupled Multi-Threading (~60 FPS)
Normally, running deep learning face recognition directly inside a video display loop causes severe frame freezing. **AetherAttend resolves this using a decoupled dual-thread architecture**:
* **Camera Feed Worker Thread**: Continuously captures webcam frames and renders them to the Tkinter canvas, maintaining a fluid **60 FPS stream**.
* **Recognition Processor Thread**: Runs asynchronously in the background. It grabs active frames, resizes them to 0.25x for rapid CPU computation, detects bounding boxes, extracts 128-D face embeddings, and updates UI overlays without blocking the interface.

### 2. Multi-Angle Enrollment Wizard (Pose Validation Filter)
To ensure high recognition accuracy across head tilts and rotations, students enroll with **10 distinct photo snapshots**.
* During registration, the wizard automatically captures snapshots when a face is detected.
* It calculates the Euclidean distance (`face_distance`) between the current face vector and previously accepted poses.
* **If the Euclidean distance is less than 0.28 (duplicate angle), the wizard rejects the frame** and prompts: `"Same angle! Rotate head slightly"`.
* This guarantees 10 mathematically distinct angles for maximum classifier precision.

### 3. Secure Admin Gateway & Student Directory Explorer
* **Role-Based Security**: Administrative actions (registering new students, deleting records) require password verification (`default: admin123`).
* **Visual Directory Manager**: A split-pane directory with an interactive Treeview list and robust lookup handlers.
* **Dynamic Photo Preview**: Automatically loads and resizes reference face snapshots (`1.jpg`) using PIL when a profile is selected.
* **Recursive Clean Deletion**: Safely removes JSON student metadata, recursively deletes image dataset directories (`shutil.rmtree`), and flushes the in-memory cache instantly.

---

## 📐 System Architecture & Data Flow

```mermaid
graph TD
    classDef hardware fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff
    classDef process fill:#0f172a,stroke:#8b5cf6,stroke-width:2px,color:#fff
    classDef ai fill:#312e81,stroke:#6366f1,stroke-width:2px,color:#fff
    classDef storage fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#fff

    A[Webcam Feed]:::hardware -->|Raw Frames ~60 FPS| B(UI Thread):::process
    B -->|Render UI Display| C[Tkinter Canvas]:::hardware
    B -->|Cache Active Frame| D{Thread Queue}:::process
    D -->|Asynchronous Copy| E(Recognition Thread):::process
    
    E -->|Downsample 0.25x| F[dlib HOG Detector]:::ai
    F -->|128-D Vector Extraction| G[ResNet Model]:::ai
    G -->|Euclidean Distance Comparison| H{Match Cached Encodings}:::ai
    
    H -->|Distance less than 0.55| I[Recognized Profile]:::storage
    H -->|Distance 0.55 or more| J[Unknown Face]:::process
    
    I -->|Log Check-in| K[CSV Attendance Register]:::storage
    I -->|Update Overlay Coordinates| B
    J -->|Update Visual Box| B


