# Face Mask Detection

A real-time face mask detection system using deep learning and computer vision. This application uses a webcam to detect faces and classify them as either **"Mask"** or **"No Mask"**, with a confidence percentage displayed on the screen.

## Features

- **Real-time Face Detection**: Uses OpenCV's Haar cascade classifier to detect faces in live video streams.
- **Deep Learning Classification**: Classifies faces as "Mask" or "No Mask" using a pre-trained TensorFlow/Keras model.
- **Color-Coded Feedback**:
  - **Green**: Mask detected.
  - **Red**: No mask detected.
- **Confidence Scores**: Displays the classification confidence percentage for each face.

---

## Demo

Here’s how the program works:

- The application captures frames from the webcam in real-time.
- Detected faces are highlighted with a bounding box.
- Each face is classified as "Mask" or "No Mask" with a confidence score.
- The output is displayed in a live video window.

---

## Prerequisites

Before running the project, ensure the following dependencies are installed:

```bash
pip install opencv-python-headless numpy tensorflow keras
