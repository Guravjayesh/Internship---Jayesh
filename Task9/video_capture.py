import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform edge detection
    canny_edges = cv2.Canny(gray_frame, 50, 150, apertureSize=3)

    # Find lines in the frame
    detected_lines = cv2.HoughLinesP(canny_edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)

    thinnest_point = None
    thinnest_thickness = None

    for line in detected_lines:
        x1, y1, x2, y2 = line[0]

        # Calculate average distance to non-edge pixels for thickness estimation
        distance = cv2.distanceTransform(canny_edges, cv2.DIST_L2, maskSize=5)  # Specify mask size
        line_mask = np.zeros_like(distance)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
        non_edge = distance[line_mask != 0]
        thickness = np.mean(non_edge)

        # Find center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Update if thinner point found
        if thinnest_thickness is None or thickness < thinnest_thickness:
            thinnest_point = (center_x, center_y)
            thinnest_thickness = thickness

    # Draw the thinnest point and line segment
    if thinnest_point is not None:
        cv2.circle(frame, thinnest_point, 5, (0, 0, 255), -1)

    # Hough line transform
    detected_lines = cv2.HoughLines(canny_edges, 1, np.pi / 180, 100)

    # Drawing detected lines on a copy of the original frame
    line_image = np.copy(frame)
    for line in detected_lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
