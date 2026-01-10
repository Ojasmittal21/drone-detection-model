# drone-detection-model
#UAV/Drone detection 
An AI/ML project that detects UAVs and Drones in images using YOLOv8 object detection
This project implements an image-based aerial object detection system using YOLOv8 to identify and classify flying objects such as drones, birds, and airplanes.
It is designed for fast, accurate detection on still images and is suitable for academic, research, and demonstration purposes.

The increasing use of drones and aerial vehicles has created the need for automated systems to monitor and identify flying objects. This project uses a deep learning–based object detection model to detect and classify aerial objects in images captured from the sky.
The system:
-Takes an image as input
-Detects all aerial objects present
-Draws bounding boxes and class labels
-Saves the annotated output

##Features-
-Detects drones, birds, and airplanes
-Supports multiple image formats (.jpg, .png, .jpeg, .webp)
-Displays class labels and confidence scores
-Saves detection results automatically
-Built using YOLOv8 and OpenCV

##Tech Stack-
-Python
-YOLOv8 (Ultralytics)
-OpenCV
-NumPy

##Dataset Information-
   The model is trained using aerial image datasets collected from:
  -Roboflow
  -Custom web-collected images
  
  Classes Used
  -drone
  -bird
  -airplane
  
  Annotation Format
  -YOLO format (.txt)
  -Normalized bounding box coordinates
  
  Only a small set of sample images is included in the repository.

##Limitations-
-Detection accuracy depends on training data quality
-Very small or distant objects may be missed
-Performance may vary in cluttered skies

##Future Improvements-
-Add support for video and live camera feed
-Train on larger and more diverse datasets
-Integrate tracking for moving objects

##Author-
-Stuti Srivastava 
-Navya Agarwal
-Ojas Mittal
-Mukul Garg

Licence-
This project is intended for academic and research use only.
