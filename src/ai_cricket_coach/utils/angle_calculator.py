import numpy as np

def calculate_angle(a,b,c):

      a = np.array(a)
      b = np.array(b)
      c = np.array(c)
      
      ba = a - b
      bc = c - b
      cos_theta = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
      angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
      return np.degrees(angle)