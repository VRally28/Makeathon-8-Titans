import numpy as np
import cv2

def process_thermal(thermal_array):
    thermal_uint8 = thermal_array.astype(np.uint8)

    # Threshold high temperature
    _, mask = cv2.threshold(thermal_uint8, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    return centers
