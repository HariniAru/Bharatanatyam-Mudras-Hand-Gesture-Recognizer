# MediaPipe’s hand landmark reference:
# 0: wrist
# 1–4: thumb (tip is 4)
# 5–8: index (tip is 8)
# 9–12: middle (tip is 12)
# 13–16: ring (tip is 16)
# 17–20: pinky (tip is 20)

import numpy as np

def is_pataka(landmarks: np.ndarray, angle_thresh=25) -> bool:
    """
    Return True if hand appears to be in Pataka mudra.
    Rule: All fingers extended straight, thumb bent slightly in.
    """
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # Wrist to finger MCPs
    wrist = landmarks[0]
    straight_fingers = True

    for tip_idx, base_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        v1 = vector(landmarks[base_idx], wrist)
        v2 = vector(landmarks[tip_idx], landmarks[base_idx])
        angle = angle_between(v1, v2)
        if angle > angle_thresh:
            straight_fingers = False
            break

    # Optional: check thumb is not sticking out
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb angle = {thumb_angle:.2f}")

    # return straight_fingers and (30 < thumb_angle < 80)
    return straight_fingers and (20 < thumb_angle < 80)
