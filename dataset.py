# dataset.py
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from config import config

class MobileCODDataset(Dataset):
    """Mobile-optimized Camouflage Object Detection Dataset"""
    def __init__(self, img_dir, mask_dir, dataset_name='camo', transform=None, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dataset_name = dataset_name
        self.transform = transform
        self.mode = mode
        
        # Get image list
        self.images = self._get_image_list()
        
        print(f"Mobile COD - Loaded {len(self.images)} images from {dataset_name} ({mode})")
    
    def _get_image_list(self):
        """Get list of images that have corresponding masks"""
        images = []
        if os.path.exists(self.img_dir):
            for f in os.listdir(self.img_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    mask_name = self._get_mask_name(f)
                    mask_path = os.path.join(self.mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        images.append(f)
                    else:
                        # Try to find mask with different extensions
                        found = False
                        for ext in ['.png', '.jpg', '.bmp']:
                            alt_mask_name = os.path.splitext(f)[0] + ext
                            alt_mask_path = os.path.join(self.mask_dir, alt_mask_name)
                            if os.path.exists(alt_mask_path):
                                images.append(f)
                                found = True
                                break
        return images
    
    def _get_mask_name(self, img_name):
        """Convert image name to mask name"""
        base_name = os.path.splitext(img_name)[0]
        return base_name + '.png'
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)

        mask_name = self._get_mask_name(img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        gt_mask = Image.open(mask_path).convert('L')

    # Resize mask using NEAREST to keep exact values
        mask_resize = transforms.Resize((config.input_size, config.input_size),
                                   interpolation=transforms.InterpolationMode.NEAREST)
        gt_mask = mask_resize(gt_mask)

    # Convert mask to numpy for processing
        gt_mask_np = np.array(gt_mask, dtype=np.uint8)

    # Generate boundary map
        boundary_map = self._generate_boundary_map(gt_mask_np)
        boundary_map = torch.from_numpy(boundary_map).float().unsqueeze(0)
        boundary_map = torch.clamp(boundary_map, 0.0, 1.0)

    # Convert mask to float32 0/1 for BCE
        gt_mask = (gt_mask_np > 128).astype(np.float32)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)

    # Transform image
        if self.transform:
           image = self.transform(image)
        else:
           image = transforms.Compose([
            transforms.Resize((config.input_size, config.input_size)),
            transforms.ToTensor()
        ])(image)

        return {
            'image': image,
            'mask': gt_mask,
            'boundary': boundary_map,
            'name': f"{self.dataset_name}_{img_name}",
            'dataset': self.dataset_name
    }

    
    def _generate_boundary_map(self, gt_mask):
        """Generate boundary map for boundary-aware learning"""
        gt_mask = (gt_mask > 128).astype(np.uint8) * 255

    # Morphology
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(gt_mask, kernel, iterations=1)
        dilated = cv2.dilate(gt_mask, kernel, iterations=1)

        boundary = dilated - eroded
        boundary = boundary.astype(np.float32) / 255.0

    # Make sure boundary is correct range
        boundary = np.clip(boundary, 0.0, 1.0)

        return boundary