# evaluate_full_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from model import LightweightBoundaryAwareCOD
from metrics import Simple_COD_Metrics, calculate_model_performance
from trainer import MobileCODTrainer
from config import config

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def load_full_model_safely():
    """Load full model with multiple fallback methods - UPDATED FOR NEW MODELS"""
    model_paths = [
        'fixed_full_model_best.pth',      
        'fixed_full_model_final.pth',     
        'simple_model_final.pth',         
        'checkpoint_epoch_50.pth',       
        'lightweight_boundary_aware_cod.pt'  
    ]
    
    for model_path in model_paths:
        if not os.path.exists(model_path):
            continue
            
        print(f"🔍 Trying to load: {model_path}")
        try:
            model = LightweightBoundaryAwareCOD()
            
            # Method 1: Try weights_only=False
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"✅ Loaded {model_path} with weights_only=False")
            except:
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"✅ Loaded {model_path} with default loading")
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("📁 Loaded from checkpoint with 'model_state_dict'")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("📁 Loaded from checkpoint with 'state_dict'")
            else:
                state_dict = checkpoint
                print("📁 Loaded raw state dict")
            
            model.load_state_dict(state_dict)
            print(f"🎯 Successfully loaded model from {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {e}")
            continue
    
    print("❌ No model files found. Available files:")
    for file in os.listdir('.'):
        if file.endswith('.pth') or file.endswith('.pt'):
            print(f"   - {file}")
    return None

def evaluate_full_model():
    """Evaluate only the full model and return metrics"""
    print("🚀 EVALUATING FULL MODEL ONLY")
    print("=" * 50)
    
    # Load data
    trainer = MobileCODTrainer(config)
    trainer.setup_mobile_data()
    test_loader = trainer.test_loader
    
    # Load model
    device = torch.device('cpu')
    model = load_full_model_safely()
    
    if model is None:
        print("❌ Could not load full model with any method")
        return None
    
    model.to(device)
    model.eval()
    
    print("📊 Calculating metrics for Full Model...")
    
    # Calculate all metrics
    metrics = Simple_COD_Metrics.calculate_all_metrics(model, test_loader, device)
    performance = calculate_model_performance(model, device)
    
    full_results = {
        'train_losses': [],  # We don't have the training losses
        'final_loss': 0.0277,  # From your original training
        'best_loss': 0.0277,
        **metrics,
        **performance
    }
    
    print("\n✅ FULL MODEL RESULTS:")
    print(f"   - F-measure: {metrics['F_measure']:.4f}")
    print(f"   - MAE: {metrics['MAE']:.4f}")
    print(f"   - E-measure: {metrics['E_measure']:.4f}")
    print(f"   - S-measure: {metrics['S_measure']:.4f}")
    print(f"   - Inference: {performance['inference_time_ms']:.2f}ms")
    print(f"   - FPS: {performance['fps']:.2f}")
    print(f"   - Parameters: {performance['parameters']:,}")
    
    return full_results

def merge_with_ablation_results(full_results):
    """Merge full model results with existing ablation results"""
    # Load existing ablation results
    try:
        with open('ablation_results.json', 'r') as f:
            ablation_results = json.load(f)
    except FileNotFoundError:
        print("❌ ablation_results.json not found. Creating new one...")
        ablation_results = {}
    
    # Convert full_results to serializable types before adding
    full_results_serializable = convert_to_serializable(full_results)
    
    # Add full model results
    ablation_results['Full Model'] = full_results_serializable
    
    # Save updated results
    with open('ablation_results_complete.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print("\n📁 Merged results saved to: ablation_results_complete.json")
    
    return ablation_results

if __name__ == "__main__":
    full_results = evaluate_full_model()
    
    if full_results is not None:
        # Merge with existing ablation results
        complete_results = merge_with_ablation_results(full_results)
        
        print("\n🎉 COMPLETE ABLATION STUDY RESULTS:")
        for model_name, results in complete_results.items():
            print(f"\n{model_name}:")
            print(f"  F-measure: {results['F_measure']:.4f}")
            print(f"  MAE: {results['MAE']:.4f}")
            print(f"  E-measure: {results['E_measure']:.4f}")
            print(f"  S-measure: {results['S_measure']:.4f}")
            print(f"  Inference: {results['inference_time_ms']:.2f}ms")
    else:
        print("\n❌ Could not evaluate full model")