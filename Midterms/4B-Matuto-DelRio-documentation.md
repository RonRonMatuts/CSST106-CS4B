# Task 1: Dataset Selection and Algorithm
<h4>Libraries Used:</h4>
<p>1. Open CV</p>
<p>2. Matplotlib</p>
<p>3. Numpy</p>
<p>4. Python</p>

<h4>Dataset Used in the Program:</h4>
<p><strong>1. Coco.Names:</strong> Name of the dataset that's been used that contains all the possible labels that corresponds to their assigned images</p>

<h4>Algorithm Used:</h4>
<p><strong>1. ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt:</strong> A pretrained SSD Model used to detect objects found on the image</p>
<p><strong>2. frozen_inference_graph.pb</strong>: TensorFlow model file that stands for "Protocol Buffer." It contains the serialized version of a TensorFlow model, including its architecture (the computation graph) and the trained weights (parameters). 
  This format is commonly used for saving and loading models in TensorFlow, making it easier to deploy machine learning models in production environments.</p>


# Task 2: Implementation
<h4>Data Preparatiom</h4>
<p>1. Import Libraries</p>

```py
import cv2
from google.colab.patches import cv2_imshow
```

<p>2. Load and Display Image</p>

```py
img = cv2.imread("/content/image/cat.jpg")

cv2_imshow(img)
cv2.waitKey(0)
```

<h4>Result:</h4>

![cat](https://github.com/user-attachments/assets/3d4522d4-5da6-48e5-8d5f-842901a5e1b1)

<p>3. Training and Testing the Pre-trained Model</p>

<p><b>Initialize the Model</b></p>

```py
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
```

<p><b>Adjust the weightPath and configPath</b></p>

```py
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.7)

print(classIds, bbox)
```

<p><b>Testing the Model</b></p>

```py
for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
  x, y, w, h = box
  cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=3)
  cv2.putText(img, classNames[classId-1].upper(), (x+10, y+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2_imshow(img)
```

<h4>Results:</h4>

![pusa](https://github.com/user-attachments/assets/34e3a602-c586-4477-b04b-bc78f9c9fe98)

<p>4. Full Code Script</p>

```py
# Add to the beginning of your code
import numpy as np

# Initialize lists to store confidence scores
confidence_scores = []

# Metrics storage
process_times = []  # To store processing times per image

# Process each image in the folder
for filename in os.listdir(image_folder):
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue  # Skip if the image could not be read

    start_time = time.time()  # Start timer

    # Keep copies for original and predicted images
    img_original = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)  # Original for display (RGB)
    img_predicted = img.copy()  # For drawing predictions

    # Perform detection
    classIds, confs, bbox = net.detect(img_predicted, confThreshold=confidence_threshold)

    # Draw bounding boxes and labels on the predicted image
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if w * h < min_box_area:
            continue  # Skip small boxes
        confidence_scores.append(confidence)  # Collect confidence scores

        # Draw bounding box and label with color
        color = colors[classNames[classId - 1]]
        cv2.rectangle(img_predicted, (x, y), (x + w, y + h), color=color, thickness=3)
        label = f"{classNames[classId - 1].upper()} {confidence:.2f}"
        cv2.putText(img_predicted, label, (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

    # Calculate processing time
    process_times.append(time.time() - start_time)

    # Convert img_predicted to RGB for matplotlib display
    img_predicted = cv2.cvtColor(img_predicted, cv2.COLOR_BGR2RGB)

    # Display original and predicted images side by side with titles
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(img_original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_predicted)
    plt.title("Predicted Image")
    plt.axis("off")
    plt.show()

    # Save the predicted image to the output folder with .jpg extension
    output_path = os.path.join(output_folder, "predicted_" + os.path.splitext(filename)[0] + ".jpg")
    cv2.imwrite(output_path, cv2.cvtColor(img_predicted, cv2.COLOR_RGB2BGR))  # Convert back to BGR for saving
    print(f"Processed and saved: {output_path}")

# Calculate and display performance metrics after processing all images
if confidence_scores:
    average_confidence = np.mean(confidence_scores)
    min_confidence = np.min(confidence_scores)
    max_confidence = np.max(confidence_scores)
    avg_processing_time = np.mean(process_times)
    total_processing_time = np.sum(process_times)

    print("\nPerformance Metrics:")
    print(f"Average Confidence Score: {average_confidence:.2f}")
    print(f"Min Confidence Score: {min_confidence:.2f}")
    print(f"Max Confidence Score: {max_confidence:.2f}")
    print(f"Average Processing Time per Image: {avg_processing_time:.2f} seconds")
    print(f"Total Processing Time: {total_processing_time:.2f} seconds")
else:
    print("\nNo detections were made.")
```

<h4>Results:</h4>

![predicted_dog](https://github.com/user-attachments/assets/b0b70b02-a0db-4610-a7fe-4f1a9ef48e0e)

![predicted_cat](https://github.com/user-attachments/assets/ad1708e2-e1c4-43e6-b1d1-7126cb7c1ab1)

![predicted_bike](https://github.com/user-attachments/assets/f2c3f16f-3490-4af7-974f-be40cea5e206)

<h4>Classification Results:</h4>

Performance Metrics:
  * Average Confidence Score: 0.74
  * Min Confidence Score: 0.71
  * Max Confidence Score: 0.76
  * Average Processing Time per Image: 0.13 seconds
  * Total Processing Time: 0.38 seconds

<p><strong>Note:</strong> This is a pre-trained model. We tried implementing evaluation metrics such as f1-score and accuracy since the necessary materials are incomplete</p>

# Task 3: Comparison

1. HOG (Histogram of Oriented Gradients)
Description: HOG is a feature descriptor used primarily for object detection. It works by counting occurrences of gradient orientation in localized portions of an image.

    <p><strong>Strengths:</strong></p>
    
      * Simple and effective for detecting objects like pedestrians.
      *  Works well in controlled environments and with specific object types.

    <p><strong>Weaknesses:</strong></p>

      * Slower than deep learning methods because it relies on classical computer vision techniques.
      * Not robust to variations in scale, orientation, and lighting conditions.
      * Limited in handling multiple classes of objects simultaneously.
      * Use Cases: Often used for detecting pedestrians in images, face detection, and simple object detection tasks.

2. YOLO (You Only Look Once)
Description: YOLO is a real-time object detection system that divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell simultaneously.

    <p><strong>Strengths</strong>:</p>

      * Extremely fast and capable of real-time processing, making it suitable for applications like video surveillance and autonomous driving.
      * Can detect multiple objects in a single pass, which improves efficiency.  
      * Good balance between accuracy and speed, especially in recent versions (like YOLOv5 and YOLOv7).
  
    <p><strong>Weaknesses:</strong></p>

      * May struggle with small objects or objects that are very close together due to grid limitations.
      * Earlier versions had issues with localization accuracy compared to two-stage detectors (e.g., Faster R-CNN).
      * Use Cases: Popular in applications requiring real-time object detection, such as robotics, automotive systems, and video analysis.

3. SSD (Single Shot MultiBox Detector)
Description: SSD also performs object detection in a single pass. It uses a series of convolutional layers to predict bounding boxes and class scores from feature maps of different resolutions.

    <p><strong>Strengths:</strong></p>

      * Faster than traditional methods while achieving competitive accuracy.
      * Can handle objects of varying sizes due to multiple feature maps at different scales.
      * More flexible and can detect a wider variety of object classes.

    <p><strong>Weaknesses:</strong></p>

      * While it is faster than two-stage detectors, it may not reach the same level of accuracy as methods like Faster R-CNN in complex scenarios.
      * May require more tuning for optimal performance on different datasets.
      * Use Cases: Suitable for real-time applications where a balance between speed and accuracy is needed, such as in mobile applications, drones, and robotics.

<h4>Video Demonstration Link: https://drive.google.com/file/d/1jwfG6jlkik4EWl_naBciy3GSEVdrE2Mp/view?usp=sharing</h4>
