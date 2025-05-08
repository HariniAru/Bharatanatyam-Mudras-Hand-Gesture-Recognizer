# cs445_finalproject



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








pip install mediapipe opencv-python numpy
pip install cv2
pip install scikit-learn joblib numpy
pip freeze > requirements.txt                       



Report notes:
in report, talk about why you shifted from a rule-based classifier to a RandomForestClassifier (it was hard to code out rules )