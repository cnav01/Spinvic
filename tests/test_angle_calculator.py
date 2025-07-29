import unittest
import numpy as np
import sys
import os

# This helps Python find your 'scripts' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This line imports YOUR function from the other file
from scripts.angle_calculator import calculate_angle

class TestAngleCalculator(unittest.TestCase):

    def test_right_angle(self):
        """Tests a perfect 90-degree angle."""
        # We create 3 points that form a right angle
        point1 = [0, 1, 0]
        point2 = [0, 0, 0]  # The vertex
        point3 = [1, 0, 0]
        
        # We run YOUR function on these points
        angle = calculate_angle(point1, point2, point3)
        
        # We check if the result is almost 90
        self.assertAlmostEqual(angle, 90.0, places=1)

    def test_straight_angle(self):
        """Tests a perfect 180-degree angle."""
        point1 = [0, 1, 0]
        point2 = [0, 0, 0]
        point3 = [0, -1, 0]
        
        angle = calculate_angle(point1, point2, point3)
        
        self.assertAlmostEqual(angle, 180.0, places=1)

# This part makes the test runnable from the command line
if __name__ == '__main__':
    unittest.main()