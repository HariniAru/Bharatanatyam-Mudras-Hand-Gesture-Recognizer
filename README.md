# Bharatanatyam Mudra Classifier

This project is a real-time hand gesture recognition system for **Bharatanatyam Mudras (Hand Gestures)**, built using **MediaPipe**, **OpenCV**, and **scikit-learn**.

The system identifies classical hand gestures using 3D hand pose landmarks and a trained Random Forest classifier.

---

## Recognized Gestures

The model is trained on the **first 14 gestures** from the [Asamyuta Hasta Mudras](https://www.natyasutraonline.com/picture-gallery/asamyuta-hasta-bharatanatyam) list:

1. Pataka  
2. Tripataka  
3. Ardhapataka  
4. Kartarimukha  
5. Mayura  
6. Ardhachandra  
7. Arala  
8. Shukatundaka 
9. Mushti  
10. Shikhara  
11. Kapittha  
12. Katakamukha  
13. Suchi  
14. Chandrakala  

---

## Dependencies

Make sure you have the following Python packages installed:

```bash
pip install mediapipe opencv-python scikit-learn numpy joblib cv2
```

I also ran this to generate the requirements.txt file:

```bash
pip freeze > requirements.txt                       
```

## How to Run

To launch the real-time gesture recognizer:

```bash
python live_mudra_recognizer.py
```

You may need to enable camera access in your system's privacy settings.

If the webcam doesn't open or freezes, open live_mudra_recognizer.py and locate this line:

```bash
cap = cv2.VideoCapture(1)
```

Then change 1 to 0, 2, etc., depending on your webcam's device index:
```bash
cap = cv2.VideoCapture(0)
```

This should resolve issues related to incorrect camera selection.

## Train Your Own Model

I recorded around You can retrain the classifier using your own .npy samples:

1. Record samples using record_mudra.py using:

```bash
python train_gesture_classifier.py <MUDRA_LABEL>
```

This saves (21, 3) NumPy arrays to:

```bash
mudra_data/<gesture_name>/<gesture_name>_###.npy
```

Examples:

```bash
mudra_data/pataka/pataka_001.npy
mudra_data/mushti/mushti_005.npy
```

2. Once you have all your labeled data, train the model by running:

```bash
python train_gesture_classifier.py
```

This script will:
- Extract joint angle features from the landmark data
- Train a **RandomForestClassifier** using scikit-learn
- Save the trained model to **gesture_classifier.pkl**

This model is automatically loaded in **live_mudra_recognizer.py** for real-time predictions.

## How It Works

MediaPipe detects 21 3D landmarks for the hand in each frame.

The system extracts meaningful features:
- PIP joint angles (bending of index, middle, ring, pinky)
- Thumb angle (wrist → base → tip)

These features are passed to a machine learning model (Random Forest) trained on labeled gestures.

During real-time webcam input, the same features are extracted and used to predict the current mudra.

The result is displayed live on screen via OpenCV.