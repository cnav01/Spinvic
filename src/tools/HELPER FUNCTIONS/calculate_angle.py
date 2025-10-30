import numpy as np
import typing import Dict, Any  

#Helper function to calculate angle between three landmarks
def calculate_angle(landmark1: list, landmark2: list, landmark3: list) -> float:
    """
    Calculates the angle between three points(landmarks) in degrees.
    Args:
        landmark1 (list): Coordinates of the first landmark [x, y, z].
        landmark2 (list): Coordinates of the second landmark [x, y, z].
        landmark3 (list): Coordinates of the third landmark [x, y, z].
        
    """
    point1 = np.array(landmark1)  # First point
    point2 = np.array(landmark2)  # Mid point (VERTEX OF THE ANGLE)
    point3 = np.array(landmark3)  # End point

    vector1 = point1 - point2
    vector2 = point3 - point2

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is within valid range
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg

