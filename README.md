<h1>CatMood: Cat Emotion Classifier</h1>
https://qxisa.github.io/CatMood/
designed only for mobile view not compatible with desktop view

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

**Exporting the Model for Web Browser**

To run the trained model in the browser, you need to export it to ONNX format. There are two ways to do this:

**Option 1: Export directly from Colab (Recommended)**

After training your model in the Colab notebook, run the export cell (already included in the notebook). This will:
1. Save the model weights to `catmood_model.pth`
2. Export the model to `catmood_model.onnx`
3. Automatically download the ONNX file to your computer

Then place the downloaded `catmood_model.onnx` file in the root of this repository.

**Option 2: Export using the Python script**

1. After training your model in the Colab notebook, save the model weights:
   ```python
   torch.save(model.state_dict(), 'catmood_model.pth')
   ```

2. Download the saved weights file to your local machine.

3. Run the export script:
   ```bash
   pip install torch torchvision onnx
   python export_to_onnx.py -i catmood_model.pth -o catmood_model.onnx
   ```

4. Place the generated `catmood_model.onnx` file in the root of this repository.

**Running the Web Application Locally**

After placing the ONNX model file, serve the application locally:
```bash
python -m http.server 8000
```
Then open http://localhost:8000 in your browser.

The web app uses ONNX Runtime Web to run the model directly in the browser - no backend server required!

**Deploying to GitHub Pages**

1. Upload the `catmood_model.onnx` file to your repository (or use GitHub LFS for large files)
2. Enable GitHub Pages in your repository settings
3. The web application will be available at your GitHub Pages URL

**Results**

The model achieves accurate classification for most moods and can provide the top-3 predicted moods with probabilities for any given cat image.

**Web Application**

CatMood includes a mobile-friendly web UI that can be deployed using GitHub Pages. The web app allows users to:

- Take photos using their device camera (front or back)
- Upload existing cat images
- Get mood analysis with top-3 predictions and confidence scores

The web application uses ONNX Runtime Web to run the trained ResNet18 model directly in the browser for real-time inference.

**License**

This project is for educational purposes. Dataset and model usage should respect any source image copyright.
