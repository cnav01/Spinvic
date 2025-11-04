import numpy as np

def calculate_angle(landmark1: list, landmark2: list, landmark3: list) -> float:
    """
    Calculates the 3D angle between three landmarks.
    Assumes landmarks are in [x, y, z] format.
    """
    point1 = np.array(landmark1)
    point2 = np.array(landmark2)
    point3 = np.array(landmark3)

    vector1 = point1 - point2
    vector2 = point3 - point2

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0 # Avoid division by zero

    cosine_angle = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)