# train_full_model_fixed.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LightweightBoundaryAwareCOD
from trainer import MobileCODTrainer
from config import config
import time

def debug_dataloader(train_loader):
    """Debug what the dataloader returns"""
    print("🔍 DEBUGGING DATALOADER...")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i} type: {type(batch)}")
        if isinstance(batch, (list, tuple)):
            print(f"  Length: {len(batch)}")
            for j, item in enumerate(batch):
                print(f"  Item {j}: type {type(item)}, shape {item.shape if hasattr(item, 'shape') else 'N/A'}")
        elif isinstance(batch, dict):
            print(f"  Keys: {batch.keys()}")
            for key, value in batch.items():
                print(f"  {key}: shape {value.shape if hasattr(value, 'shape') else 'N/A'}")
        break  # Only check first batch

def get_model_parameters_generic(model):
    """Get parameters without assuming specific module names"""
    # Try to identify different components
    backbone_params = []
    other_params = []
    
    for name, module in model.named_children():
        if any(keyword in name.lower() for keyword in ['backbone', 'encoder', 'features']):
            print(f"📦 Identified backbone: {name}")
            backbone_params.extend(module.parameters())
        else:
            print(f"🔧 Identified other module: {name}")
            other_params.extend(module.parameters())
    
    if backbone_params:
        return [
            {'params': backbone_params, 'lr': 1e-5},
            {'params': other_params, 'lr': 1e-4}
        ]
    else:
        # Fallback: all parameters same LR
        return model.parameters()

def train_full_model_fixed():
    print("🚀 RETRAINING FULL MODEL WITH FIXED SETTINGS")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = LightweightBoundaryAwareCOD()
    model.to(device)
    
    # Debug model structure
    print("\n🔍 Model Structure:")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
    
    # Data
    trainer = MobileCODTrainer(config)
    trainer.setup_mobile_data()
    train_loader = trainer.train_loader
    
    # Debug dataloader first
    debug_dataloader(train_loader)
    
    # Get parameters (generic approach)
    parameters = get_model_parameters_generic(model)
    
    # Optimizer
    if isinstance(parameters, list):
        optimizer = optim.AdamW(parameters, weight_decay=1e-5)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_loss = float('inf')
    train_losses = []
    
    print("\n🎯 Starting training...")
    for epoch in range(50):  # Reduced epochs for quick test
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, masks = batch[0], batch[1]
                else:
                    print(f"❌ Unexpected batch format: {len(batch)} elements")
                    continue
            elif isinstance(batch, dict):
                images, masks = batch['image'], batch['mask']
            else:
                print(f"❌ Unknown batch type: {type(batch)}")
                continue
            
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - your model returns 3 outputs
            outputs = model(images)
            # Use the first output (prediction) for loss calculation
            pred = outputs[0]  # This should be your main prediction
            
            # Calculate loss
            loss = criterion(pred, masks)
            
            # Backward with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch+1:02d} | Batch: {batch_idx:03d} | Loss: {loss.item():.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)
        
        current_lr = scheduler.get_last_lr()[0] if scheduler else 1e-4
        print(f'🚀 Epoch: {epoch+1:02d} | Average Loss: {avg_loss:.4f} | LR: {current_lr:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'fixed_full_model_best.pth')
            print(f'💾 Saved best model with loss: {best_loss:.4f}')
        
        # Early test every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("🧪 Running quick test...")
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(train_loader))
                if isinstance(test_batch, (list, tuple)) and len(test_batch) >= 2:
                    test_images, test_masks = test_batch[0], test_batch[1]
                elif isinstance(test_batch, dict):
                    test_images, test_masks = test_batch['image'], test_batch['mask']
                else:
                    continue
                    
                test_output = model(test_images.to(device))
                test_pred = test_output[0]  # Use first output
                test_loss = criterion(test_pred, test_masks.to(device))
                print(f'   Test Loss: {test_loss.item():.4f}')
            model.train()
    
    # Save final model
    torch.save(model.state_dict(), 'fixed_full_model_final.pth')
    print("✅ Training completed!")
    
    return model, train_losses

if __name__ == "__main__":
    model, losses = train_full_model_fixed()
