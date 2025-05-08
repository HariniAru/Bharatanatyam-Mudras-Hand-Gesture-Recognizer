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

# def is_tripataka(landmarks: np.ndarray) -> bool:
#     def vector(a, b):
#         return np.array(b) - np.array(a)

#     def angle_between(v1, v2):
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         return np.degrees(np.arccos(cos_angle))

#     wrist = landmarks[0]

#     # Check index, middle, pinky are straight
#     straight_fingers = True
#     for tip_idx, base_idx in [(8, 6), (12, 10), (20, 18)]:
#         v1 = vector(landmarks[base_idx], wrist)
#         v2 = vector(landmarks[tip_idx], landmarks[base_idx])
#         angle = angle_between(v1, v2)
#         if angle > 25:
#             straight_fingers = False
#             break

#     # Check ring finger is bent
#     v1 = vector(landmarks[14], landmarks[13])  # PIP to MCP
#     v2 = vector(landmarks[16], landmarks[14])  # TIP to PIP
#     ring_angle = angle_between(v1, v2)
#     print(f"Ring finger angle = {ring_angle:.2f}")

#     # return straight_fingers and (ring_angle > 30)  # bent more than 30°
#     return straight_fingers and (30 < ring_angle < 100)



def is_tripataka(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    wrist = landmarks[0]

    # --------- Ring Finger Bent Check ---------
    ring_mcp = landmarks[13]
    ring_pip = landmarks[14]
    ring_tip = landmarks[16]

    # Angle at PIP joint
    v1 = vector(ring_pip, ring_mcp)
    v2 = vector(ring_tip, ring_pip)
    ring_angle = angle_between(v1, v2)

    # Z-depth difference: ring tip should be more "inward" (negative z direction)
    z_depth_bent = ring_tip[2] < ring_pip[2] - 0.01

    print(f"Thumb angle = {angle_between(vector(wrist, landmarks[2]), vector(landmarks[2], landmarks[4])):.2f}")
    print(f"Ring finger angle = {ring_angle:.2f} | Z-depth check = {z_depth_bent}")

    if ring_angle < 10 and not z_depth_bent:
        # If ring finger is very straight and not inward, reject early
        return False

    # --------- Pataka-like structure (excluding ring finger) ---------
    def is_partial_pataka_base(landmarks, finger_thresh=35, thumb_range=(20, 80)) -> bool:
        for tip_idx, base_idx in [(8, 6), (12, 10), (20, 18)]:  # Index, middle, pinky
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


# helper for tripataka
# def is_partial_pataka_base(landmarks: np.ndarray, finger_thresh=35, thumb_range=(20, 80)) -> bool:
#     def vector(a, b):
#         return np.array(b) - np.array(a)
#     def angle_between(v1, v2):
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         return np.degrees(np.arccos(cos_angle))

#     wrist = landmarks[0]

#     # Check index, middle, pinky are straight
#     for tip_idx, base_idx in [(8, 6), (12, 10), (20, 18)]:
#         v1 = vector(landmarks[base_idx], wrist)
#         v2 = vector(landmarks[tip_idx], landmarks[base_idx])
#         angle = angle_between(v1, v2)
#         if angle > finger_thresh:
#             return False

#     # Check thumb bend
#     thumb_tip = landmarks[4]
#     thumb_base = landmarks[2]
#     thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
#     print(f"Thumb angle = {thumb_angle:.2f}")
#     return thumb_range[0] < thumb_angle < thumb_range[1]

# def is_tripataka(landmarks: np.ndarray) -> bool:
#     def vector(a, b):
#         return np.array(b) - np.array(a)
#     def angle_between(v1, v2):
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         return np.degrees(np.arccos(cos_angle))

#     # Use relaxed version of pataka (excluding ring finger)
#     if is_ring_bent_enough(landmarks):
#         if is_partial_pataka_base(landmarks):
#             return True

#     # Check that ring finger is bent
#     v1 = vector(landmarks[14], landmarks[13])  # PIP to MCP
#     v2 = vector(landmarks[16], landmarks[14])  # TIP to PIP
#     ring_angle = angle_between(v1, v2)
#     print(f"Ring finger angle = {ring_angle:.2f}")

#     # return ring_angle > 25
#     # return 12 < ring_angle < 120
#     return 7 < ring_angle < 120




def is_ardhapataka(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    wrist = landmarks[0]
    passed = True

    # Index + middle should be fairly straight
    for label, (tip, pip) in zip(["Index", "Middle"], [(8, 6), (12, 10)]):
        v1 = vector(landmarks[pip], wrist)
        v2 = vector(landmarks[tip], landmarks[pip])
        angle = angle_between(v1, v2)
        print(f"{label} finger angle = {angle:.2f}")
        if angle > 25:
            passed = False

    # Ring + pinky should be bent
    for label, (tip, pip) in zip(["Ring", "Pinky"], [(16, 14), (20, 18)]):
        v1 = vector(landmarks[pip], landmarks[pip - 1])
        v2 = vector(landmarks[tip], landmarks[pip])
        angle = angle_between(v1, v2)
        print(f"{label} finger angle = {angle:.2f}")
        if angle < 25:  # ↓ reduced from 30
            passed = False

    # Thumb: slight to moderate bend
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb angle = {thumb_angle:.2f}")
    if not (20 < thumb_angle < 60):
        passed = False

    return passed


# def is_ardhapataka(landmarks: np.ndarray) -> bool:
#     def vector(a, b):
#         return np.array(b) - np.array(a)

#     def angle_between(v1, v2):
#         cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#         cos_angle = np.clip(cos_angle, -1.0, 1.0)
#         return np.degrees(np.arccos(cos_angle))

#     wrist = landmarks[0]

#     # Fingers to check
#     straight_fingers = [(8, 6), (12, 10)]      # Index, Middle
#     bent_fingers = [(16, 14), (20, 18)]        # Ring, Pinky

#     for tip, pip in straight_fingers:
#         v1 = vector(landmarks[pip], wrist)
#         v2 = vector(landmarks[tip], landmarks[pip])
#         angle = angle_between(v1, v2)
#         if angle > 25:
#             return False  # should be mostly straight

#     for tip, pip in bent_fingers:
#         v1 = vector(landmarks[pip], landmarks[pip - 1])  # MCP to PIP
#         v2 = vector(landmarks[tip], landmarks[pip])      # TIP to PIP
#         angle = angle_between(v1, v2)
#         if angle < 30:
#             return False  # should be clearly bent

#     # Thumb bend check
#     thumb_base = landmarks[2]
#     thumb_tip = landmarks[4]
#     thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
#     print(f"Thumb angle = {thumb_angle:.2f}")
#     if not (20 < thumb_angle < 60):
#         return False

#     return True



def is_mushti(landmarks: np.ndarray) -> bool:
    def vector(a, b):
        return np.array(b) - np.array(a)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    # Fingers to check
    finger_names = ["Index", "Middle", "Ring", "Pinky"]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    bent_fingers = 0
    for name, tip, pip in zip(finger_names, finger_tips, finger_pips):
        v1 = vector(landmarks[pip], landmarks[pip - 1])  # MCP to PIP
        v2 = vector(landmarks[tip], landmarks[pip])      # TIP to PIP
        angle = angle_between(v1, v2)
        print(f"{name} finger bend angle = {angle:.2f}")
        if angle > 40:  # was 60; lowered threshold based on real data
            bent_fingers += 1

    if bent_fingers < 4:
        return False

    # Thumb: should be bent and slightly crossing over (moderate angle)
    wrist = landmarks[0]
    thumb_base = landmarks[2]
    thumb_tip = landmarks[4]
    thumb_angle = angle_between(vector(wrist, thumb_base), vector(thumb_base, thumb_tip))
    print(f"Thumb bend angle = {thumb_angle:.2f}")

    return 40 < thumb_angle < 100
