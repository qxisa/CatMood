<h1>CatMood: Cat Emotion Classifier</h1>

**Project Overview**

CatMood is a machine learning project designed to classify the mood of cats based on images. The system can identify seven distinct moods: playful, angry, sad, sick, curious, relaxed, and hungry. This project demonstrates the complete ML workflow, including dataset collection, preprocessing, model training, and evaluation.

**Dataset**

The dataset was collected and annotated using Roboflow. Images are organized into folders by mood and preprocessed to a consistent size. Each image has a single label representing the cat's dominant mood. The dataset is split into training, validation, and test sets.

**Methodology**

Model: Pretrained ResNet18 from PyTorch's torchvision library.

Transforms: All images resized to 224x224 and converted to tensors.

Training: The model is trained using cross-entropy loss and Adam optimizer. Epochs and batch size are adjustable.

Evaluation: Validation accuracy is measured per epoch, and a dedicated testing function allows for random image evaluation with top-3 predictions.

**Usage**

Clone the repository.

Install dependencies:

pip install torch torchvision roboflow pillow matplotlib


Download the dataset from Roboflow using the provided API key.

Run the training notebook or script to train the model.

Use the test function to evaluate random images and see top-3 predictions.

**Results**

The model achieves accurate classification for most moods and can provide the top-3 predicted moods with probabilities for any given cat image.

**Web Application**

CatMood includes a mobile-friendly web UI that can be deployed using GitHub Pages. The web app allows users to:

- Take photos using their device camera (front or back)
- Upload existing cat images
- Get mood analysis with top-3 predictions and confidence scores

To use the web app:
1. Enable GitHub Pages in your repository settings (Settings → Pages → Source: Deploy from branch → main)
2. Access the app at `https://<username>.github.io/CatMood/`

The web interface is fully responsive and works on both mobile devices and desktop browsers.

**License**

This project is for educational purposes. Dataset and model usage should respect any source image copyright.
