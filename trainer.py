# trainer.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from sklearn.metrics import precision_recall_curve
import time

from dataset import MobileCODDataset
from model import LightweightBoundaryAwareCOD
from config import config

class StableLoss(nn.Module):
    """Stable loss function using BCEWithLogitsLoss"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        # Use BCEWithLogitsLoss - it's more numerically stable
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.boundary_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targets):
        pred_mask, pred_boundary = preds  # Now these are raw logits
        gt_mask, gt_boundary = targets
        
        # Debug info
        print(f"🔍 DEBUG - Input ranges:")
        print(f"  pred_mask: [{pred_mask.min().item():.3f}, {pred_mask.max().item():.3f}]")
        print(f"  pred_boundary: [{pred_boundary.min().item():.3f}, {pred_boundary.max().item():.3f}]")
        print(f"  gt_mask: [{gt_mask.min().item():.3f}, {gt_mask.max().item():.3f}]")
        print(f"  gt_boundary: [{gt_boundary.min().item():.3f}, {gt_boundary.max().item():.3f}]")
        
        # Safety checks
        if torch.isnan(pred_mask).any():
            print("⚠️ WARNING: NaN in pred_mask, replacing with zeros")
            pred_mask = torch.nan_to_num(pred_mask, 0.0)
        
        if torch.isnan(pred_boundary).any():
            print("⚠️ WARNING: NaN in pred_boundary, replacing with zeros")
            pred_boundary = torch.nan_to_num(pred_boundary, 0.0)
        
        # Calculate losses
        seg_loss = self.seg_loss(pred_mask, gt_mask)
        boundary_loss = self.boundary_loss(pred_boundary, gt_boundary)
        
        total_loss = self.alpha * seg_loss + self.beta * boundary_loss
        
        print(f"📊 Losses - Total: {total_loss.item():.4f}, Seg: {seg_loss.item():.4f}, Boundary: {boundary_loss.item():.4f}")
        
        return total_loss, seg_loss, boundary_loss

class MobileCODTrainer:
    def __init__(self, config):
        self.config = config
        
        # Force CPU for stability
        self.device = torch.device('cpu')
        print("🔧 RUNNING ON CPU FOR STABILITY")
        
        # Initialize model
        self.model = LightweightBoundaryAwareCOD(config.backbone).to(self.device)
        self._print_mobile_stats()

        # Simple transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor(),
        ])

    def _print_mobile_stats(self):
        total_params = self.model.count_parameters()
        model_size_mb = total_params * 4 / (1024**2)
        print("=" * 60)
        print("MOBILE-OPTIMIZED BOUNDARY-AWARE COD FRAMEWORK")
        print("=" * 60)
        print(f"Model Parameters: {total_params:,}")
        print(f"Model Size: {model_size_mb:.2f} MB")
        print(f"Target Size: {self.config.target_size} MB")
        print(f"Mobile Backbone: {self.config.backbone}")
        print(f"Input Size: {self.config.input_size}x{self.config.input_size}")
        print("=" * 60)

    def debug_datasets(self):
    #"""Debug dataset loading to see what's happening"""
        print("\n" + "="*50)
        print("DATASET DEBUG INFORMATION")
        print("="*50)
    
        for dataset_name in self.config.train_datasets:
            if dataset_name in self.config.dataset_paths:
               paths = self.config.dataset_paths[dataset_name]
               img_dir = paths['train_img']
               mask_dir = paths['train_mask']
            
            print(f"\n📁 Dataset: {dataset_name}")
            print(f"   Image dir: {img_dir}")
            print(f"   Mask dir: {mask_dir}")
            print(f"   Exists: {os.path.exists(img_dir) and os.path.exists(mask_dir)}")
            
            if os.path.exists(img_dir):
                images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"   Found {len(images)} images")
                if images:
                    print(f"   Sample images: {images[:3]}")
            
            if os.path.exists(mask_dir):
                masks = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                print(f"   Found {len(masks)} masks")

        print("="*50 + "\n")      

    def setup_mobile_data(self):

        self.debug_datasets()
        train_datasets = []
        test_datasets = []

        for dataset_name in self.config.train_datasets:
            if dataset_name in self.config.dataset_paths:
                paths = self.config.dataset_paths[dataset_name]
                img_dir = paths['train_img']
                mask_dir = paths['train_mask']
                if os.path.exists(img_dir) and os.path.exists(mask_dir):
                    dataset = MobileCODDataset(
                        img_dir=img_dir,
                        mask_dir=mask_dir,
                        dataset_name=dataset_name,
                        transform=self.transform,
                        mode='train'
                    )
                    train_datasets.append(dataset)
                    print(f"✓ Added {dataset_name} training: {len(dataset)} images")

        for dataset_name in self.config.test_datasets:
            if dataset_name in self.config.dataset_paths:
                paths = self.config.dataset_paths[dataset_name]
                img_dir = paths['test_img']
                mask_dir = paths['test_mask']
                if os.path.exists(img_dir) and os.path.exists(mask_dir):
                    dataset = MobileCODDataset(
                        img_dir=img_dir,
                        mask_dir=mask_dir,
                        dataset_name=dataset_name,
                        transform=self.transform,
                        mode='test'
                    )
                    test_datasets.append(dataset)
                    print(f"✓ Added {dataset_name} testing: {len(dataset)} images")

        if train_datasets:
            self.train_dataset = ConcatDataset(train_datasets)
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
        if test_datasets:
            self.test_dataset = ConcatDataset(test_datasets)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )

        print(f"Total train samples: {len(self.train_dataset)}")
        print(f"Total test samples: {len(self.test_dataset)}")
     

    def train_mobile_model(self):
        print("Starting Mobile-Optimized Training...")
        self.setup_mobile_data()

    # Add data verification
        print(f"✅ Total training samples: {len(self.train_dataset)}")
        print(f"✅ Expected batches per epoch: {len(self.train_loader)}")
        print(f"✅ Batch size: {self.config.batch_size}")
    
    # Verify first batch
        sample_batch = next(iter(self.train_loader))
        print(f"✅ Batch keys: {sample_batch.keys()}")
        print(f"✅ Image shape: {sample_batch['image'].shape}")
        print(f"✅ Mask shape: {sample_batch['mask'].shape}")
        print(f"✅ Boundary shape: {sample_batch['boundary'].shape}")

        criterion = StableLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        history = {'train_loss': []}

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Load data
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                boundaries = batch['boundary'].to(self.device)

                # Verify data ranges occasionally
                if batch_idx % 200 == 0:
                    print(f"🔍 Batch {batch_idx} - "
                          f"Images: [{images.min().item():.3f}, {images.max().item():.3f}], "
                          f"Masks: [{masks.min().item():.3f}, {masks.max().item():.3f}]")

                # Ensure correct ranges
                masks = masks.float().clamp(0, 1)
                boundaries = boundaries.float().clamp(0, 1)

                # Forward pass
                optimizer.zero_grad()
                pred_masks, pred_boundaries = self.model(images)
                
                total_loss, seg_loss, boundary_loss = criterion(
                    (pred_masks, pred_boundaries),
                    (masks, boundaries)
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()
                batch_count += 1

                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch+1}/{self.config.epochs} | '
                          f'Batch {batch_idx}/{len(self.train_loader)} | '
                          f'Loss {total_loss.item():.4f}')
                          
            except Exception as e:
                print(f"❌ ERROR in batch {batch_idx}: {e}")
                continue

        # Calculate average loss properly
        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
        else:
            avg_loss = 0.0
            
        history['train_loss'].append(avg_loss)
        print(f'Epoch {epoch+1}: average train loss {avg_loss:.4f}')

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            self.save_checkpoint(epoch, history)

            return history

    def save_checkpoint(self, epoch, history):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': history
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}.pth')
        print(f"Checkpoint saved: checkpoint_epoch_{epoch+1}.pth")