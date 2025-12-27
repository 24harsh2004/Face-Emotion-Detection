**Project Overview**

Face Emotion Detection is a computer vision–based deep learning project that identifies human emotions by analyzing facial expressions from images and real-time video streams. The system detects faces, extracts facial features, and classifies emotions with high accuracy, making it useful for mental health analysis, human–computer interaction, and intelligent surveillance systems.

 **Project Objectives**

Detect human faces accurately in images and live video

Classify facial expressions into predefined emotion categories

Build an efficient CNN-based deep learning model

Perform real-time emotion recognition using a webcam

Improve robustness against lighting and pose variations

 **Emotion Categories**

The system recognizes the following facial emotions:

Happy

Sad

Angry

Fear

Surprise

Disgust

Neutral

**Technical Architecture**

Face Detection: Haar Cascade Classifier (OpenCV)

Feature Extraction: Convolutional layers

Classification: Fully connected neural network

Activation Functions: ReLU, Softmax

Loss Function: Categorical Cross-Entropy

Optimizer: Adam

 **Dataset Details**

Facial expression image dataset with labeled emotion classes

Images converted to grayscale for faster computation

Image resizing and normalization performed

Data augmentation techniques used:

Rotation

Zoom

Horizontal flipping

 **Technologies & Tools Used**

Programming Language: Python

Libraries & Frameworks:

TensorFlow / Keras

OpenCV

NumPy

Pandas

Matplotlib

Seaborn

 **Methodology**

Data collection and preprocessing

Face detection using Haar Cascade

Image normalization and augmentation

CNN model design and training

Model validation and testing

Performance evaluation using accuracy and loss curves

Real-time emotion prediction via webcam

 **Model Evaluation Metrics**

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

 **Results & Observations**

Achieved stable training and validation accuracy

Model successfully predicts emotions in real time

Works efficiently on low-resolution images

Performs well in controlled and semi-controlled environments

 **Experiments Conducted**

Comparison between different CNN depths

Effect of data augmentation on accuracy

Performance testing under different lighting conditions

Real-time vs static image evaluation


 **Applications**

Mental health and emotion monitoring systems

Online learning engagement analysis

Human–computer interaction systems

Smart surveillance and security systems

Customer sentiment analysis

 **Limitations**

Performance may reduce in extreme lighting conditions

Accuracy depends on face alignment and clarity

Occluded faces (masks, glasses) affect predictions

 **Future Scope**

Use transfer learning (ResNet, MobileNet, EfficientNet)

Improve accuracy with attention mechanisms

Integrate facial landmark detection

Deploy model using Flask/FastAPI

Extend to multi-face emotion detection

 **Project Structure**
face-emotion-detection/
│
├── dataset/
├── models/
├── haarcascade/
├── emotion_detection.py
├── requirements.txt
└── README.md

 **Author**

Harsh Agarwal
B.Tech – Artificial Intelligence & Machine Learning

 This project demonstrates practical expertise in deep learning, computer vision, and real-time AI systems.
