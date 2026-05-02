# test.py
# Single image inference for Mobile-COD
# Usage: python test.py --input image.jpg --checkpoint model.pth

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import argparse
import warnings
import matplotlib.pyplot as plt  # ← ADDED THIS

warnings.filterwarnings("ignore")

# Import your model
from model import LightweightBoundaryAwareCOD
from config import Config


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    model = LightweightBoundaryAwareCOD()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully")
    return model


def preprocess_image(image_path, size=(256, 256)):
    """Load and preprocess input image"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    original_img = np.array(img)
    img_resized = img.resize(size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_resized).unsqueeze(0)
    return input_tensor, original_size, original_img


def predict(model, input_tensor, device, original_size):
    """Run inference and return prediction mask"""
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        pred = outputs[0]  # Main prediction
        pred_mask = torch.sigmoid(pred[0, 0]).cpu().numpy()
    
    # Resize back to original size
    pred_mask = cv2.resize(pred_mask, original_size)
    binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    return pred_mask, binary_mask


def save_result(image_path, binary_mask, output_path=None):
    """Save prediction result"""
    if output_path is None:
        base = os.path.splitext(image_path)[0]
        output_path = f"{base}_prediction.png"
    
    cv2.imwrite(output_path, binary_mask)
    print(f"✅ Prediction saved to: {output_path}")
    return output_path


def visualize_comparison(original_img, binary_mask, save_path=None):
    """Create side-by-side comparison of input and prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Input Image', fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title('Mobile-COD Prediction', fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Comparison saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Mobile-COD Inference')
    parser.add_argument('--input', '-i', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--checkpoint', '-c', type=str, 
                       default='fixed_full_model_best.pth', 
                       help='Path to model checkpoint')
    parser.add_argument('--output', '-o', type=str, default=None, 
                       help='Output path for prediction mask')
    parser.add_argument('--device', '-d', type=str, default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--compare', action='store_true', 
                       help='Save side-by-side comparison')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    print("=" * 50)
    print("🚀 Mobile-COD Inference")
    print("=" * 50)
    print(f"Input: {args.input}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # Check input exists
    if not os.path.exists(args.input):
        print(f"❌ Input image not found: {args.input}")
        return
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        print("   Please download from Zenodo: https://doi.org/10.5281/zenodo.19677006")
        return
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Preprocess image
    input_tensor, original_size, original_img = preprocess_image(args.input)
    
    # Run inference
    pred_mask, binary_mask = predict(model, input_tensor, device, original_size)
    
    # Save result
    save_result(args.input, binary_mask, args.output)
    
    # Save comparison if requested
    if args.compare:
        visualize_comparison(original_img, binary_mask)
    
    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()