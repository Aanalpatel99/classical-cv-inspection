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

# add the defect - white circle
# actual defect 1
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

# defect 2 
cordinates1 = (400, 100)
cordinates2 = (400, 130)
color = (255)
thickness = 2
cv2.line(blackImgRef, (cordinates1[0], cordinates1[1]), (cordinates2[0], cordinates2[1]), color, thickness)

# defect 3
cordinates1 = (500, 100)
cordinates2 = (520, 100)
color = (255)
thickness = 2
cv2.line(blackImgRef, (cordinates1[0], cordinates1[1]), (cordinates2[0], cordinates2[1]), color, thickness)

imgTest = blackImgRef.copy()

# edge detection using Canny
edges = cv2.Canny(imgTest, 100, 250) # Canny edge detection with lower threshold 100 and upper threshold 200

# subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Original image
axs[0].imshow(imgTest, cmap='gray', vmin=0, vmax=255) # vmin and vmax set the range of pixel values for display
axs[0].set_title("Original Image")
axs[0].axis('off')

# Edges
axs[1].imshow(edges, cmap='gray', vmin=0, vmax=255) # vmin and vmax set the range of pixel values for display
axs[1].set_title("Edges")
axs[1].axis('off')

plt.tight_layout()
plt.show()
