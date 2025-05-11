# Trains RandomForestClassifier model on mudra data
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def vector(a, b):
    return np.array(b) - np.array(a)

def angle_between(v1, v2):
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def extract_features(landmarks):
    features = []
    pip_pairs = [(6, 5, 8), (10, 9, 12), (14, 13, 16), (18, 17, 20)]
    for pip, mcp, tip in pip_pairs:
        v1 = vector(landmarks[mcp], landmarks[pip])
        v2 = vector(landmarks[tip], landmarks[pip])
        features.append(angle_between(v1, v2))
    wrist = landmarks[0]
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    v1 = vector(wrist, thumb_base)
    v2 = vector(thumb_base, thumb_tip)
    features.append(angle_between(v1, v2))
    return features

def load_dataset(data_dir="mudra_data"):
    X, y = [], []
    for gesture in sorted(os.listdir(data_dir)):
        gesture_path = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue
        for filename in sorted(os.listdir(gesture_path)):
            if not filename.endswith(".npy"):
                continue
            path = os.path.join(gesture_path, filename)
            landmarks = np.load(path)
            X.append(extract_features(landmarks))
            y.append(gesture)
    return np.array(X), np.array(y)

# prints out classification report for each gesture along with different parameters and model accuracy
def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    joblib.dump(clf, "gesture_classifier.pkl")
    print("Model saved as 'gesture_classifier.pkl'")

if __name__ == "__main__":
    main()
