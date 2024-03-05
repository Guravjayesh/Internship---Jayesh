import cv2
import numpy as np
import matplotlib.pyplot as plt
import  math

image= cv2.imread('line.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
canny_edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)


# Find lines in the image
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
    cv2.circle(image, thinnest_point, 5, (0, 0, 255), -1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()