# 🤖 Real-Time Virtual Background with YOLOv8 Segmentation

## 🌟 Project 1: Foundational Segmentation Pipeline

This project demonstrates the complete pipeline for real-time video segmentation, using the powerful YOLOv8-Seg model to achieve pixel-perfect foreground/background separation and applying computer vision techniques for live video manipulation. This work establishes the foundation for more advanced geometry-based analysis in robotics and sports tracking.

### Key Features

* **Real-Time Inference:** High-speed segmentation using the optimized **YOLOv8n-Seg** model.
* **Mask Processing:** Custom Python logic using NumPy and OpenCV to extract, resize, and threshold the raw mask data.
* **Alpha Blending:** Implementation of the $\text{Final} = (\text{Foreground} \times \alpha) + (\text{Background} \times (1 - \alpha))$ blending formula to seamlessly merge the sharp foreground with a blurred background.
* **Webcam Integration:** Direct, efficient processing of a live webcam feed.

### ⚙️ How It Works (First Principles)

The entire application relies on transforming the model's probabilistic output into precise array-based instructions for pixel manipulation.

1.  **Model Inference:** YOLOv8 outputs a low-resolution probability array ($\mathbf{P}$) for where the foreground object is located.
2.  **Resizing & Thresholding:** The low-res mask is scaled to the camera resolution ($H \times W$) and converted into a **binary mask** ($\alpha$), where pixels are either $1.0$ (person) or $0.0$ (background).
3.  **Broadcasting:** The single-channel mask ($\alpha$) is expanded to three channels ($\alpha_{3ch}$) using NumPy's `[:, :, None]` broadcasting trick, enabling element-wise multiplication with the BGR image frames. 
4.  **Blending:** The final image is calculated by combining two weighted components:
    * **Foreground:** $\text{Frame} \times \alpha_{3ch}$
    * **Background:** $\text{Blurred Frame} \times (1 - \alpha_{3ch})$

### 🚀 Setup and Run

#### Prerequisites

* Python 3.8+
* Working Webcam

#### Installation

``bash

# Install the Ultralytics library for YOLOv8

pip install ultralytics opencv-python numpy


Save the project code and run it from your terminal:

Bash

python Yolov8-seg.py

Press 'q' to exit the live stream.


Author: Amaar A.
Date: 12-12-2025
