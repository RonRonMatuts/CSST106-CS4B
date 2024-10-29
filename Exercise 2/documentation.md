# Exercise 2: Sift orb Feature Extraction

<h3>Task 1: SIFT Feature Extraction</h3>
<p><strong>About the task:</strong> It provides and explains us the code on how we should upload the images and use the uploaded images
for implementing SIFT Detector. Once the codes are properly followed and implemented, the expected output would be provided</p>

<h4>Sample Code</h4>

```py
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('payat.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize Sift Detector
sift = cv2.SIFT_create()

# Detect keypoint and descriptors
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)

# Display the image with keypoints
plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints')
plt.show()
```

<h4>Output</h4>
