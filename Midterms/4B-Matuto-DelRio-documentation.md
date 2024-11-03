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
