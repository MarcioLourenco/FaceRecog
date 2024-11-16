
# Siamese Neural Network for Facial Recognition  

## Data 
http://vis-www.cs.umass.edu/lfw/

## 📋 Project  

This project implements a **Siamese Neural Network (SNN)** for facial recognition. The SNN learns to distinguish between pairs of images based on similarity metrics. The goal is to accurately identify whether two images belong to the same person.  

The application is optimized to run on GPU-enabled systems with NVIDIA support, leveraging TensorFlow for efficient training and inference.  

---

## 🚀 Features  
1. **Facial Recognition with Siamese Networks**  
   - Compares two images to determine if they belong to the same person.  

2. **Preprocessing and Data Augmentation**  
   - Resizes images to a fixed size (100x100).  
   - Applies data augmentation techniques to improve generalization.  

3. **Checkpointing**  
   - Automatically saves model weights every 10 epochs.  

4. **GPU Support**  
   - Configured to utilize NVIDIA GPUs for accelerated training and inference.  

5. **Future Enhancement with FaceNet**  
   - The model can be replaced with FaceNet for generating more robust embeddings.  

---

## 📁 Project Structure  
```plaintext
├── data/                  # Data directory
│   ├── positive/          # Positive face images
│   ├── negative/          # Negative face images
│   ├── anchor/            # Anchor images for training
├── checkpoints/           # Saved model weights
├── notebooks/             # Notebooks for analysis and prototyping
├── README.md              # This documentation file
├── main.py                # Main script for training and inference
└── requirements.txt       # Project dependencies
```

## 🖥️ Requirements
Required Libraries
Install the dependencies with:

bash
Copy code
pip install -r requirements.txt
GPU Configuration
Ensure your system has an NVIDIA GPU.
Install the correct NVIDIA driver and CUDA Toolkit.
Install cuDNN for TensorFlow compatibility.
Verify GPU detection:
python
Copy code
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

## 🧪 How to Run
1. **Prepare the Data**
Import positive, negative, and anchor images into the respective directories.
Use the script to capture images via webcam:
bash
Copy code
python main.py --capture
2. **Training**
Train the model with GPU support:
bash
Copy code
python main.py --train
3. **Evaluation**
Evaluate the model using the test dataset:
bash
Copy code
python main.py --evaluate

## 🛠️ Improvements and Notes
Issues Faced
Confusion Between Similar People
The model confused my image with my wife's.
Planned Solutions
Increase Training Data:
Add more diverse images for both myself and my wife.
Hard Negative Mining:
Train the model with image pairs it failed on to focus on challenging distinctions.
Switch to FaceNet:
Use FaceNet for generating more robust and accurate embeddings.
GPU Usage Notes
This project is optimized for NVIDIA GPUs using TensorFlow. Ensure proper system setup for best performance.

## 📊 Performance Metrics
After initial training (10 epochs):
Precision: 0.97
Recall: 0.96
Accuracy: 0.97

## 📌 Checkpoints
The model automatically saves weights every 10 epochs. If training is interrupted, you can resume from the last checkpoint.

To load a checkpoint:
python
Copy code
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=siamese_model)
checkpoint.restore(tf.train.latest_checkpoint('checkpoints/'))

## 📖 References
[Siamese Networks](https://en.wikipedia.org/wiki/Siamese_neural_network)
[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
[https://www.tensorflow.org/](https://www.tensorflow.org/)