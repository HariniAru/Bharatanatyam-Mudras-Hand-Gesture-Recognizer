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

## How to Run

To launch the real-time gesture recognizer:

```bash
python live_mudra_recognizer.py
```

You may need to enable camera access in your system's privacy settings.

If the webcam doesn't open or freezes, open live_mudra_recognizer.py and locate this line:

```bash
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
```

Then change 1 to 0, 2, etc., depending on your webcam's device index:
```bash
cap = cv2.VideoCapture(0)
```

This should resolve issues related to incorrect camera selection.

## Train Your Own Model

You can retrain the classifier using your own .npy samples:

1. Record samples using record_mudra.py, which saves (21, 3) NumPy arrays to:

```bash
mudra_data/<gesture_name>/<gesture_name>_###.npy
```

Example:

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
    PIP joint angles (bending of index, middle, ring, pinky)
    Thumb angle (wrist → base → tip)
These features are passed to a machine learning model (Random Forest) trained on labeled gestures.
During real-time webcam input, the same features are extracted and used to predict the current mudra.
The result is displayed live on screen via OpenCV.




pip install mediapipe opencv-python numpy
pip install cv2
pip install scikit-learn joblib numpy
pip freeze > requirements.txt                       



Report notes:
in report, talk about why you shifted from a rule-based classifier to a RandomForestClassifier (it was hard to code out rules)

This is what I used as reference for the gestures (28 total) - https://www.natyasutraonline.com/picture-gallery/asamyuta-hasta-bharatanatyam






Limitations of MediaPipe:

You **should avoid rotating your wrist too much** in your samples because:

---

### 🤖 MediaPipe is View-Dependent

MediaPipe Hands is **not invariant to extreme hand rotations**. Its landmark detector is trained mostly on:

* **front-facing palms**
* moderate variation in pitch/yaw
* **clear finger separations**

When you **rotate your wrist** too much (e.g., show the back of your hand, or tilt sideways), MediaPipe may:

* **mislabel landmarks** (e.g., index and middle get swapped)
* **miss some fingers**
* give **inconsistent 3D coordinates** across samples of the same gesture

This leads to:

* **inaccurate angles**
* poor generalization in your `is_mudra()` rules
* your gesture being classified differently **just because of viewpoint**

---

### ✅ What You *Should* Vary Instead:

1. **Hand position** (left/right/up/down in the frame)
2. **Distance from the camera**
3. **Slight rotations**, but not more than \~30–45° from frontal
4. **Lighting/background** if possible (for robustness)

---

### 🔁 Later Option (Optional):

If you *really* want to include rotated views (e.g., for robustness), do so later — **after your base model or rules work well front-on**. Then you can:

* Label them separately
* Train a model or rule to handle multiple “poses” of the same mudra





Tools You Can Use (All in Python):

Tool	Use Case
scikit-learn	Train ML models on features (angles, vectors)
NumPy / Pandas	Prepare and structure your dataset
MediaPipe	Already used — gives the hand landmarks
Optional: PyTorch / TensorFlow	For deep learning (not needed now)


