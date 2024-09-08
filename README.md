# CSST106-CS4B: Introduction to Computer Vision
<h4>Prepared By: Matuto Byron</h4>
<h4>Section: BSCS IS 4B</h4>
<h4>Instructor: Mark Bernardino</h4>

https://github.com/user-attachments/assets/c65b3229-14bd-4200-8fa8-e32755d14d10

<details open>
  <summary><h2>Research and Comprehend</h2></summary>
  
  <h3>What is Computer Vision?</h3>
  
  <p><strong>Computer Vision:</strong> A field of artificial intelligence that trains computers to interpret and understand the   visual world. By processing digital images from cameras and videos, computer vision systems aim to perform tasks that are usually associated with human vision.</p>
  
  <p><strong>Image Processing:</strong> Techniques used to enhance, manipulate, and analyze images. It involves the use of algorithms to process digital images to improve their quality or extract useful information.</p>

  <h3>Image Processing Techniques</h3>
  
  <p><strong>Filtering:</strong> A technique used to enhance or suppress specific features in an image. Examples include smoothing (to reduce noise) and sharpening (to enhance details).</p>
  
  <p><strong>Edge Detection:</strong> Methods used to identify the boundaries of objects within an image. Techniques such as the Sobel operator or Canny edge detector help in detecting changes in intensity that signify edges.</p>
  
  <p><strong>Segmentation:</strong> The process of partitioning an image into multiple segments or regions to simplify the analysis. Techniques include thresholding (separating objects from the background) and clustering (grouping similar pixels).</p>
  
</details>

<details open>
  <summary><h2>Hands on Exploration</h2></summary>

  <h3>Case Study Selection</h3>
  
  <p><strong>Facial Recognition Systems:</strong> AI systems designed to identify or verify individuals based on facial features. These systems often use techniques such as feature extraction and pattern recognition.</p>
  
 <h4>Facial recognition systems use computer vision to identify or verify individuals based on their facial features. Here’s a simplified explanation of how it works:</h4>

 <p><strong>1. Image Capture:</strong> The system captures an image of a face using a camera. This can be a live feed or a still   image.</p>

 <p><strong>2. Face Detection:</strong> The system detects and locates the face within the image. This involves identifying the    face's position and boundaries.</p>

 <p><strong>3. Feature Extraction:</strong> The facial features are extracted from the detected face. These features include the   distance between eyes, the shape of the nose, and the contour of the jawline.</p>

 <p><strong>4. Feature Represenation:</strong> The extracted features are converted into a numerical representation, often called  a facial embedding or faceprint. This is a unique set of values that represents the face’s distinct characteristics.</p>

 <p><strong>5. Comparison Matching:</strong> The facial embedding is compared against a database of known faceprints. The system   searches for a match or verifies if the face matches an individual’s stored profile.</p>

 <p><strong>6. Decision Making:</strong> Based on the comparison, the system determines if the face matches any known identity or  if it’s an unknown individual.</p>
   
<h4>Facial recognition systems rely on sophisticated algorithms and machine learning models to accurately detect and recognize faces, even under varying lighting conditions, angles, and expressions.</h4>

  <h3>Implementation Creation</h3>

  <h4>Here's the actual implementation and source code of my chosen application</h4>
  
  <p><strong>Step 1: </strong> Import Necessary libraries</p>

  ```py
  import cv2
  import numpy as np
  from skimage.metrics import structural_similarity as ssim
  import matplotlib.pyplot as plt
  ```

 1. ```cv2```: OpenCV library for image processing.
 2. ```numpy```: Library for numerical operations (used for handling image data).
 3. ```skimage.metrics.structural_similarity```: Function for comparing images using SSIM.
 4. ```matplotlib.pyplot```: Library for displaying images and results.

  <p><strong>Step 2:</strong> Load the Pre-Trained Face Detector</p>

  ```py
  # Load pre-trained face detector
  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  ```
  1. ```cv2.CascadeClassifier```: Loads the Haar cascade classifier for face detection.
  2. ```cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'```: Path to the pre-trained Haar cascade XML file for detecting faces.

  <p><strong>Step 3:</strong> Create a Function to Detect and Extract Faces</p>

  ```py
  def detect_face(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use the face detector to find faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are found, return the first face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return gray[y:y+h, x:x+w]
    
    # Return None if no face is detected
    return None
  ```
  1. ```cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)```: Converts the image from color to grayscale.
  2. ```detectMultiScale```: Detects faces in the grayscale image.
       * ```scaleFactor```: Compensates for face size variations.
       * ```minNeighbors```: Minimum number of neighbors a rectangle should have to retain it.
       * ```minSize```: Minimum size of the detected faces.
  3. ```faces[0]```: Returns the first detected face. You can modify this to handle multiple faces if needed.
  4. ```gray[y:y+h, x:x+w]```: Extracts the face region from the grayscale image.

  <p><strong>Step 4:</strong> Load the Known and Unknown Images</p>

  ```py
  # Load the known and unknown images
  known_image = cv2.imread('Known_Image.jfif')
  unknown_image = cv2.imread('Unknown_Image.jfif')

  # Display the images to verify they are loaded
  def display_image(image, title='Image'):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

  # Display the loaded images
  display_image(known_image, 'Known Image')
  display_image(unknown_image, 'Unknown Image')
  ```
  1. ```cv2.imread()```: Loads the images from the specified file paths.
  2. ```display_image()```: A function to display an image using matplotlib. It converts the image from BGR (OpenCV format) to           RGB (matplotlib format) for correct color representation.

  <p><strong>Step 5:</strong> Detect Faces in Both Images</p>

  ```py
  # Detect faces in both images
  known_face = detect_face(known_image)
  unknown_face = detect_face(unknown_image)

  # Check if faces were detected
  if known_face is None:
    print("No face detected in the known image.")
  if unknown_face is None:
    print("No face detected in the unknown image.")
  ```
  1. ```detect_face(known_image)```: Detects and extracts the face region from the known image.
  2. ```detect_face(unknown_image)```: Detects and extracts the face region from the unknown image.
  3. **Check for None**: Prints a message if no face is detected in either image. This helps in debugging and ensures that face           detection was successful.

  <p><strong>Step 6:</strong> Resized Both Face Regions</p>

  ```py
  # Resize both face regions to the same size for comparison
  if known_face is not None and unknown_face is not None:
    known_face = cv2.resize(known_face, (100, 100))
    unknown_face = cv2.resize(unknown_face, (100, 100))
  else:
    print("Face detection failed in one or both images.")
  ```
  1. ```cv2.resize()```: Resizes the detected face regions to a standard size (100x100 pixels in this case) to ensure that both         face images have the same dimensions for comparison.

  <p><strong>Step 7:</strong> Compare the Faces Using Structural Similarity Index (SSIM)</p>

  ```py
  from skimage.metrics import structural_similarity as ssim

  # Compare the two faces using Structural Similarity Index (SSIM)
  if known_face is not None and unknown_face is not None:
    similarity, _ = ssim(known_face, unknown_face, full=True)
    
    # Determine if the faces match based on similarity
      result = "The unknown face matches the known person!" if similarity > 0.5 else "The unknown face does not match the known        person."
  else:
    result = "Face detection failed in one or both images."

  # Print the result
  print(result)
  ```
  1. ```ssim(known_face, unknown_face, full=True)```: Computes the SSIM value between the two resized face regions.                   ```full=True``` returns the SSIM value and the SSIM image, but we only need the similarity score.

  2. ```Threshold for Matching```: We use a threshold (0.5 in this case) to decide whether the faces match. You can adjust this          threshold based on your images and requirements.

  <p><strong>Step 8:</strong> Display Images and Comparison Result</p>

  ```py
  def display_comparison(known_image, unknown_image, result):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    
    # Display the known image
    ax[0].imshow(cv2.cvtColor(known_image, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Known Image")
    ax[0].axis('off')
    
    # Display the unknown image
    ax[1].imshow(cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Unknown Image")
    ax[1].axis('off')
    
    # Display the result
    ax[2].text(0.5, 0.5, result, fontsize=12, ha='center')
    ax[2].set_title("Result")
    ax[2].axis('off')
    
    plt.show()

# Call the function to display the images and result
display_comparison(known_image, unknown_image, result)
  ```
![sample output](https://github.com/user-attachments/assets/66ab5470-d757-4823-9212-267f3f2e5249)

1. ```display_comparison()```: This function creates a figure with three subplots:
    * The first subplot shows the known image.
    * The second subplot shows the unknown image.
    * The third subplot displays the result of the comparison.
2. ```plt.subplots()```: Creates a figure and a set of subplots. We use a single row with three columns.
3. ```imshow()```: Displays images in the subplots.
4. ```text()```: Displays the comparison result in the third subplot.
5. ```plt.show()```: Renders the figure with the displayed images and result.
  
</details>

<details open> 
  <summary><h2>Summary</h2></summary> 
  
  <h3>Key Concepts</h3>

  1. **Computer Vision** is a field of artificial intelligence that allows computers to interpret and process visual data from the world. It encompasses techniques for image analysis, object detection, and facial recognition.

  2. **Image Processing** involves a range of techniques like filtering, edge detection, and segmentation to manipulate and extract useful information from digital images.

  3. **Facial Recognition Systems** utilize computer vision techniques to detect, extract, and analyze facial features, converting them into unique numerical representations (embeddings) for identity verification or recognition.

<h3>Facial Recognition Workflow</h3>

  1. **Image Capture**: The system captures an image of a face.
  2. **Face Detection**: The system detects the face and its boundaries within the image.
  3. **Feature Extraction**: Key facial features (such as eyes, nose, and jawline) are extracted.
  4. **Feature Representation**: The extracted features are transformed into a numerical faceprint.
  5. **Comparison Matching**: The faceprint is compared against a database of known faces.
  6. **Decision Making**: The system decides whether the face matches any stored profile.

<h3>Implementation Highlights</h3>

<h4>The hands-on exploration covered a facial recognition implementation using OpenCV for face detection and SSIM for image comparison:</h4>

  * We used the Haar cascade classifier to detect faces from both a known and an unknown image.
  * Once detected, the faces were resized for uniformity and compared using the Structural Similarity Index (SSIM).
  * The similarity score determined whether the two faces matched or not.

<h3>Conclusion</h3>

This exercise demonstrated the practical application of computer vision techniques in facial recognition, from detecting and processing images to using machine learning models for matching. Through OpenCV and SSIM, we explored how systems can automate the process of face comparison efficiently.

Throughout this project, we have gained a solid understanding of the fundamental concepts behind facial recognition systems, using OpenCV and image comparison techniques like SSIM.

1. **Image Handling**: We learned how to load, manipulate, and display images efficiently. Converting images to grayscale was an essential step to enhance face detection performance.

2. **Face Detection**: We explored the use of Haar Cascade classifiers for detecting faces within images. This method proved to be an effective and widely-used tool in the field of computer vision.

3. **Face Extraction**: We practiced extracting specific regions of interest from an image, focusing on face regions for comparison. This showed how critical it is to isolate relevant data in a larger image for accurate analysis.

4. **Resizing for Uniformity**: Ensuring both face images are of the same size before comparing them taught us the importance of image uniformity in pattern recognition tasks.

5. **Face Comparison Using SSIM**: We applied the Structural Similarity Index (SSIM) to assess the similarity between two face images. SSIM offered a clear, quantifiable way to measure how structurally similar the faces are, helping us determine whether they matched.

6. **Visualization**: Finally, visualizing the results allowed us to clearly interpret the outcomes, making it easier to communicate and understand the comparison results between the known and unknown faces.

By working through this project, we deepened our understanding of how computer vision systems, particularly facial recognition systems, work. We now have a better grasp of the essential techniques involved in image processing and comparison, and how they can be applied to practical applications in AI.

</details>

<details open>
  <summary><h2>Extension Activity</h2></summary>

  <h3>Research an Emerging Form of Image Processing</h3>
  
  <p><strong>Deep Learning-based Image Analysis:</strong> Advanced techniques that use neural networks, particularly convolutional neural networks (CNNs), to analyze and interpret complex image data. These methods are highly effective in tasks such as image classification, object detection, and segmentation.</p>

</details>
