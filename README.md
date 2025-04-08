# Object_Detection
This project performs real-time object detection using the YOLOv3 model with OpenCV in Python. It processes video input (from file or webcam), detects objects using pre-trained weights on the COCO dataset, and displays bounding boxes with class labels. The YOLO model is loaded using OpenCV’s DNN module, and Non-Maximum Suppression is used to remove overlapping detections. Each detected object is shown with a colored box and label on the video frame.
## Features
- Real-time object detection
- Supports custom videos or webcam
- Basic YOLOv8 integration
- Visualization of detection results with bounding boxes

## Technologies Used
- Python
- OpenCV
- TensorFlow
- NumPy
- Pre-trained YOLOv8

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Object_Detection.git
   cd Object_Detection

2. Install dependencies:
pip install -r requirements.txt

3. Run the script:
python detect.py --source path/to/image_or_video

Results
 ![image](https://github.com/user-attachments/assets/5e1e9615-af6b-42ca-ba72-5e8ab9581102)
![image](https://github.com/user-attachments/assets/c78648d3-5dea-43ea-b6bc-4e4f5692ebca)



Contributors
Revathi N
