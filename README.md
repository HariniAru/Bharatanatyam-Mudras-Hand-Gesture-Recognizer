# Bharatanatyam Mudra Recognizer

A real-time computer vision tool that recognizes **Bharatanatyam mudras** (classical Indian dance hand gestures) from a webcam and shows the mudra's name live on screen. Built with **MediaPipe**, **OpenCV**, and **scikit-learn**.

**Results:** 89% accuracy and a 0.88 macro F1-score across 14 mudras, trained on a dataset of 231 samples I recorded myself.

[Watch the demo video](https://drive.google.com/file/d/1x13j0s3KFnbkX-vKbxNR3zUE-gk6vo-2/view?usp=share_link) · [Read the final report](https://docs.google.com/document/d/1sTDlDQG-pkDBUVVw8cXcZmhpct04P3zRY4-Glf5PNG4/edit?usp=sharing)

---

## Why I Built This

I started learning Bharatanatyam as a young child, and I still remember how long it took to get each mudra right: shaping my fingers just so, moving smoothly between gestures, and learning what each one means. For a beginner, it can be overwhelming.

This project brings that experience together with computer vision. It helps learners practice mudras accurately and learn their names, and it's a small step toward preserving a cultural art form through technology.

## Tech Stack

- **Python**
- **MediaPipe:** hand landmark detection (21 landmarks per hand)
- **OpenCV:** webcam capture and live display
- **scikit-learn:** Random Forest classifier
- **NumPy** and **joblib:** data handling and model persistence

## How It Works

1. MediaPipe detects 21 3D landmarks on the hand in each webcam frame.
2. The system turns those landmarks into joint-angle features:
   - Four PIP joint angles (how much the index, middle, ring, and pinky fingers bend)
   - A custom thumb angle (wrist → thumb base → thumb tip)
3. A Random Forest classifier, trained on my labeled samples, predicts the mudra from those features.
4. OpenCV displays the predicted mudra name on the live video.

Angles describe the shape of the hand rather than where it is in the frame, so the same mudra is recognized whether your hand is closer, farther, or off to the side.

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

## Results

The classifier was evaluated on a dataset of **231 gesture samples across 14 mudras** (at least 15 per mudra), using an 80/20 train/test split.

| Metric | Score |
| --- | --- |
| Accuracy | **89%** |
| Macro F1-score | **0.88** |

- **Perfect scores:** Mushti, Suchi, Kapittha, Chandrakala, and Shikhara reached 1.0 precision, recall, and F1. The joint-angle features separate these clearly.
- **Strong:** Katakamukha, Mayura, and Pataka scored around 0.75–0.89 in precision and recall.
- **Hardest:** Ardhapataka had high recall but low precision (F1 0.50), and Shukatundaka and Kartarimukha scored lower in precision or recall. These mudras differ from their neighbors only in subtle finger positions.

## Challenges and Design Decisions

- **Custom landmarks instead of a pre-built recognizer:** MediaPipe's built-in gesture recognizer only knows common gestures like a thumbs-up, so I used its hand landmark model and engineered my own features for Bharatanatyam.
- **From hand-written rules to machine learning:** I first tried a rule-based classifier that mapped finger positions and angles by hand. It became too complex and inaccurate for mudras with subtle differences, so I switched to a Random Forest. It works well on a small dataset, handles those subtle differences, and makes accuracy easy to measure.
- **Re-recording the dataset:** MediaPipe is mostly trained on front-facing palms with clearly separated fingers. Many mudras don't look like that, so I re-recorded much of my dataset with higher-quality samples to improve accuracy.

## Future Work

- Collect more samples per mudra, recorded from consistent angles
- Try SVMs or neural networks, and add features that capture movement
- Expand to motion-based recognition, two-handed (Samyuta) mudras, and full-body pose tracking

## Dependencies

Make sure you have the following Python packages installed:

```bash
pip install mediapipe opencv-python scikit-learn numpy joblib
```

I also ran this to generate the requirements.txt file:

```bash
pip freeze > requirements.txt                       
```

## How to Run

To launch the real-time gesture recognizer:

```bash
pip install -r requirements.txt # to ensure you have all dependencies installed
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

If you want to figure out which camera index is available to you, you can run: 

```bash
python list_cameras.py
```

## Train Your Own Model

I recorded at least 15 .npy samples for each gesture.

You can retrain the classifier using your own .npy samples:

1. Record samples using record_mudra.py using:

```bash
python record_mudra.py <MUDRA_LABEL>
```

Then press 's' to take a snapshot of your gesture (and 'q' when you're done). This saves (21, 3) NumPy arrays to:

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
- Print a classification report that quantifies the accuracy of the trained model

This model is automatically loaded in **live_mudra_recognizer.py** for real-time predictions.

## Visualize Gestures

To visualize the .npy samples you’ve recorded for a given gesture, you can run:

```bash
python preview_mudra_samples.py <MUDRA_LABEL>
```

Example:

```bash
python preview_mudra_samples.py suchi
```

This will:
- Display each saved frame as a 2D hand skeleton
- Show one sample per second
- Automatically exit on pressing q or Esc

Make sure the folder mudra_data/<gesture_name>/ exists and contains .npy files.