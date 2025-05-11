# Outlines rules for identifying each gesture (rule-based classification approach)
import numpy as np

# MediaPipe’s hand landmark reference:
# 0: wrist
# 1–4: thumb (tip is 4)
# 5–8: index (tip is 8)
# 9–12: middle (tip is 12)
# 13–16: ring (tip is 16)
# 17–20: pinky (tip is 20)

def is_pataka(landmarks: np.ndarray, angle_thresh=25) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    wrist = landmarks[0]
    straight_fingers = True

    for tip_idx, base_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        v1 = vector(landmarks[base_idx], wrist)
        v2 = vector(landmarks[tip_idx], landmarks[base_idx])
        angle = angle_between(v1, v2)
        if angle > angle_thresh:
            straight_fingers = False
            break

    # ensure thumb is not sticking out
    thumb_tip = landmarks[4]
    thumb_base = landmarks[2]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb angle = {thumb_angle:.2f}")

    # return straight_fingers and (30 < thumb_angle < 80)
    return straight_fingers and (20 < thumb_angle < 80)

def is_tripataka(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    wrist = landmarks[0]

    # ensure ring finger is bent
    ring_mcp = landmarks[13]
    ring_pip = landmarks[14]
    ring_tip = landmarks[16]

    # check angle at PIP joint
    v1 = vector(ring_pip, ring_mcp)
    v2 = vector(ring_tip, ring_pip)
    ring_angle = angle_between(v1, v2)

    # ring tip should be more inward
    z_depth_bent = ring_tip[2] < ring_pip[2] - 0.01

    print(f"Thumb angle = {angle_between(vector(wrist, landmarks[2]), vector(landmarks[2], landmarks[4])):.2f}")
    print(f"Ring finger angle = {ring_angle:.2f} | Z-depth check = {z_depth_bent}")

    if ring_angle < 10 and not z_depth_bent: # if ring finger too straight and not bent inward, return false
        return False

    # since it's kind of similar to pataka gesture, check for similar conditions
    def is_partial_pataka_base(landmarks, finger_thresh=35, thumb_range=(20, 80)) -> bool:
        for tip_idx, base_idx in [(8, 6), (12, 10), (20, 18)]:
            v1 = vector(landmarks[base_idx], wrist)
            v2 = vector(landmarks[tip_idx], landmarks[base_idx])
            angle = angle_between(v1, v2)
            if angle > finger_thresh:
                return False

        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
        return thumb_range[0] < thumb_angle < thumb_range[1]

    return is_partial_pataka_base(landmarks)

def is_ardhapataka(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    wrist = landmarks[0]
    passed = True

    # index and middle fingers should be straight
    for label, (tip, pip) in zip(["Index", "Middle"], [(8, 6), (12, 10)]):
        v1 = vector(landmarks[pip], wrist)
        v2 = vector(landmarks[tip], landmarks[pip])
        angle = angle_between(v1, v2)
        print(f"{label} finger angle = {angle:.2f}")
        if angle > 25:
            passed = False

    # ring and pinky fingers should be bent
    for label, (tip, pip) in zip(["Ring", "Pinky"], [(16, 14), (20, 18)]):
        v1 = vector(landmarks[pip], landmarks[pip - 1])
        v2 = vector(landmarks[tip], landmarks[pip])
        angle = angle_between(v1, v2)
        print(f"{label} finger angle = {angle:.2f}")
        if angle < 25:
            passed = False

    # thumb should be slightly bent
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb angle = {thumb_angle:.2f}")
    if not (20 < thumb_angle < 60):
        passed = False

    return passed

def is_mushti(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    finger_names = ["Index", "Middle", "Ring", "Pinky"]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    # basically a fist so all fingers should be bent
    bent_fingers = 0
    for name, tip, pip in zip(finger_names, finger_tips, finger_pips):
        v1 = vector(landmarks[pip], landmarks[pip - 1])
        v2 = vector(landmarks[tip], landmarks[pip])
        angle = angle_between(v1, v2)
        print(f"{name} finger bend angle = {angle:.2f}")
        if angle > 40:
            bent_fingers += 1

    if bent_fingers < 4:
        return False

    wrist = landmarks[0]
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb bend angle = {thumb_angle:.2f}")

    return 40 < thumb_angle < 100