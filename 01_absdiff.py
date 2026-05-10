import numpy as np
import cv2
import matplotlib.pyplot as plt

#black img
blackImgRef = np.zeros((200, 600), dtype=np.uint8)

# gray wireframe rectangle
start = (50, 50)
end = (550, 150)
gcolor = (128) # gray color
thickness = -1 # fill the rectangle
cv2.rectangle(blackImgRef, start, end, gcolor, thickness) # create the wireframe as rectangle
imgRef = blackImgRef.copy()

# add the defect - white circle
# actual defect
cordinates = (100, 100)
color = (255) 
thickness = -1
radius = 35
cv2.circle(blackImgRef, cordinates, radius, color, thickness)

# noise 1
cordinates = (70, 70)
color = (255) 
thickness = -1
radius = 1
cv2.circle(blackImgRef, cordinates, radius, color, thickness)

# noise 2
cordinates = (180, 60)
color = (255) 
thickness = -1
radius = 1
cv2.circle(blackImgRef, cordinates, radius, color, thickness)

# noise 3
cordinates = (500, 135)
color = (255) 
thickness = -1
radius = 1
cv2.circle(blackImgRef, cordinates, radius, color, thickness)

# noise 4
cordinates = (150, 120)
color = (255) 
thickness = -1
radius = 1
cv2.circle(blackImgRef, cordinates, radius, color, thickness)
imgTest = blackImgRef.copy()

# subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(imgRef, cmap='gray', vmin=0, vmax=255) # vmin and vmax set the range of pixel values for display
axs[0].set_title("Original Image")
axs[0].axis('off')


# test image
axs[1].imshow(imgTest, cmap='gray', vmin=0, vmax=255) # vmin and vmax set the range of pixel values for display
axs[1].set_title("Test Image")
axs[1].axis('off')

plt.tight_layout()
plt.show()


# Compute the absolute difference
abs_diff = cv2.absdiff(imgRef, imgTest) 
# compute the absolute difference between the reference and test images, 
# will make non differnce 0 and the difference will be the pixel value of the difference, 
# in this case 255 for white circle and noise, and 0 for the rest of the image

# threshold the absolute difference to create a binary mask of the defect
_, defect_mask = cv2.threshold(abs_diff, 30, 255, cv2.THRESH_BINARY) # threshold the absolute difference to create a binary mask of the defect,
# any pixel value above 30 will be set to 255 (white) and the rest will be set to 0 (black), this will help to isolate the defect from the background and noise


kernel = np.ones((3, 3), np.uint8) # create a kernel for morphological operations, in this case a 3x3 matrix of ones, 
# which will be used to perform opening and closing operations on the defect mask to remove noise and fill gaps in the detected defects.
# find contours in the clean mask, which will help to identify the individual defects and their properties such as area and bounding box,
solidMask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel) # perform morphological opening to remove small noise from the defect mask,
cleanMask = cv2.morphologyEx(solidMask, cv2.MORPH_CLOSE, kernel) # perform morphological closing to fill small holes in the detected defects, this will help to create a cleaner mask of the defects for further analysis.

contours, _ = cv2.findContours(cleanMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find contours in the clean mask, which will help to identify the individual defects and their properties such as area and bounding box,

# the contours will be stored in a list, and each contour will be represented as a numpy array of points that define the contour.
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    print(f"Defect {i + 1}: Area = {area}")
    print(f"bounding box: x={x}, y={y}, w={w}, h={h}")
    
    cv2.rectangle(imgTest, (x, y), (x + w, y + h), (255), 2)

# Display the absolute difference
plt.figure(figsize=(5, 5))
plt.imshow(cleanMask, cmap='gray', vmin=0, vmax=255)
plt.title("Clean Mask")
plt.axis('off')
plt.show()


if cleanMask.any():
    print("Defect detected!")
else:
    print("No defect detected.")