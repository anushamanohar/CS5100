import cv2
import torch
import numpy as np
from time import time

# Load your custom YOLOv5 model
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', 
                      path='/Users/anusham/Downloads/fai_data_set/yolov5/runs/train/exp/weights/best.pt', 
                      force_reload=True)
model.eval()  # Set the model to evaluation mode

# Define class names
class_names = ['blue', 'green', 'pink', 'red', 'yellow']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break
    
    # Start time for FPS calculation
    start_time = time()
    
    # Perform detection
    results = model(frame)
    
    # Get detection results
    detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections
    
    # Draw bounding boxes for detected objects
    for idx, detection in detections.iterrows():
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = int(detection['class'])
        
        # Get color for bounding box (using class name as color)
        color = (0, 255, 0)  # Default to green
        if class_names[cls] == 'blue':
            color = (255, 0, 0)
        elif class_names[cls] == 'green':
            color = (0, 255, 0)
        elif class_names[cls] == 'pink':
            color = (255, 192, 203)
        elif class_names[cls] == 'red':
            color = (0, 0, 255)
        elif class_names[cls] == 'yellow':
            color = (0, 255, 255)
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{class_names[cls]}: {conf:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Calculate FPS
    fps = 1 / (time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the resulting frame
    cv2.imshow('Color Object Detection', frame)
    
    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
