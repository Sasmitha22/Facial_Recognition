# Facial_Recognition
## Project Description

This project consists of two Python scripts for facial recognition: `data_collection.py` and `face_recognise.py`. The purpose of the project is to collect facial data from a webcam using `data_collection.py` and then recognize faces using the collected data with `face_recognise.py`.

### `data_collection.py`

This script captures facial data from a webcam and saves it in a specified directory for training the facial recognition model. It detects faces using the OpenCV library and saves the grayscale images of the detected faces after resizing them to a standard size.

### `face_recognise.py`

This script trains a facial recognition model using the data collected by `data_collection.py` and then performs real-time face recognition using the trained model. It utilizes the LBPH (Local Binary Patterns Histograms) Face Recognizer provided by OpenCV. The script continuously captures frames from the webcam, detects faces, and compares them with the trained model to recognize known faces. If a face is not recognized, it labels it as 'Unknown'.

### Dependencies

- OpenCV (`cv2`): Used for capturing video from the webcam, detecting faces, and performing image operations.
- NumPy (`numpy`): Used for numerical operations and data manipulation.

### Usage

1. Run `data_collection.py` to collect facial data. Adjust the number of samples to be collected as needed.
2. Once sufficient data is collected, run `face_recognise.py` to train the model and perform real-time face recognition.

### Note

Make sure to have the necessary dependencies installed and the XML file containing the Haar Cascade for face detection (`haarcascade_frontalface_default.xml`) available in the project directory or provide the correct path to it.

To resolve the `AttributeError: module 'cv2.face' has no attribute` issue and utilize the LBPHFaceRecognizer, you can follow these steps:

1. Uninstall the existing OpenCV package:
   ```
   pip uninstall opencv-python
   ```

2. Install the OpenCV package with the `opencv-contrib-python` package, which includes additional modules:
   ```
   pip install opencv-contrib-python
   ```

3. Update your code to create the LBPHFaceRecognizer using the correct syntax:

   Replace:
   ```python
   model = cv2.face.LBPHFaceRecognizer.create()
   ```

   With:
   ```python
   model = cv2.face_LBPHFaceRecognizer.create()
   ```

Here's the corrected part of your code:

```python
model = cv2.face_LBPHFaceRecognizer.create()
```

After making these changes, your code should work without raising the `AttributeError` related to `cv2.face`. This will allow you to use the LBPHFaceRecognizer for your face recognition tasks.
