```markdown

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
```

### Files Required
1. `haarcascade_frontalface_default.xml`: Pre-trained Haar cascade for face detection. This is included in the OpenCV library.
2. `mask_detector.model`: A pre-trained TensorFlow/Keras model for mask detection. You can either:
   - Use a pre-trained model (if available).
   - Train your own model (instructions below).

---

## How to Run

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection
   ```

2. Ensure the required files (`haarcascade_frontalface_default.xml` and `mask_detector.model`) are in the project directory.

3. Run the script:

   ```bash
   python mask_detection.py
   ```

4. Press **`q`** to quit the program.

---

## Training a Custom Model

If you don’t have a pre-trained `mask_detector.model`, you can train one using a labeled dataset of faces with and without masks.

### Steps to Train:
1. **Dataset**: Use datasets like [Kaggle Face Mask Dataset](https://www.kaggle.com/) or collect your own labeled data.
2. **Model Architecture**: Build a CNN model using TensorFlow/Keras.
3. **Preprocessing**:
   - Resize all images to `(128, 128, 3)`.
   - Normalize pixel values to `[0, 1]`.
4. **Training Script**:
   ```python
   model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
   model.fit(train_data, validation_data, epochs=10)
   model.save("mask_detector.model")
   ```
