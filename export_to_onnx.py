"""
Export the trained CatMood PyTorch model to ONNX format for browser inference.

This script loads the trained ResNet18 model and exports it to ONNX format
that can be used with ONNX Runtime Web in the browser.

Usage:
    1. Train your model first using the CatMoodelGC.ipynb notebook
    2. Save your model: torch.save(model.state_dict(), 'catmood_model.pth')
    3. Run this script: python export_to_onnx.py

Requirements:
    pip install torch torchvision onnx
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=7):
    """
    Create the ResNet18 model architecture matching the training notebook.
    
    Args:
        num_classes: Number of mood classes (default: 7 for CatMood)
    
    Returns:
        PyTorch model with ResNet18 architecture
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def export_to_onnx(model_path='catmood_model.pth', output_path='catmood_model.onnx', num_classes=7):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to the saved PyTorch model weights (.pth file)
        output_path: Path for the output ONNX model
        num_classes: Number of mood classes
    """
    # Create model architecture
    model = create_model(num_classes)
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input matching the expected input shape (batch, channels, height, width)
    # The model expects 224x224 RGB images normalized with ImageNet stats
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    print(f"Input shape: (batch_size, 3, 224, 224)")
    print(f"Output shape: (batch_size, {num_classes})")
    print("\nMood classes (in order):")
    classes = ['angry', 'curious', 'hungry', 'playful', 'relaxed', 'sad', 'sick']
    for i, cls in enumerate(classes):
        print(f"  {i}: {cls}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export CatMood PyTorch model to ONNX')
    parser.add_argument('--input', '-i', default='catmood_model.pth',
                        help='Path to PyTorch model weights (.pth file)')
    parser.add_argument('--output', '-o', default='catmood_model.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--num-classes', '-n', type=int, default=7,
                        help='Number of mood classes')
    
    args = parser.parse_args()
    
    export_to_onnx(args.input, args.output, args.num_classes)
