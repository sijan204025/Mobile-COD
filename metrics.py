# metrics.py
import numpy as np
import torch
import cv2
from sklearn.metrics import precision_recall_curve, f1_score
import time

class COD_Metrics:
    """Comprehensive metrics for Camouflage Object Detection without skimage dependency"""
    
    @staticmethod
    def calculate_f_measure(pred, gt):
        """Calculate F-measure (F1 score)"""
        pred_binary = (pred > 0.5).astype(np.float32)
        return f1_score(gt.flatten(), pred_binary.flatten())
    
    @staticmethod
    def calculate_mae(pred, gt):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(pred - gt))
    
    @staticmethod
    def calculate_e_measure(pred, gt):
        """Calculate Enhanced Alignment Measure"""
        # Normalize predictions and ground truth
        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
        gt_norm = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-8)
        
        # Enhanced alignment calculation
        align_matrix = 1 - np.abs(pred_norm - gt_norm)
        e_measure = np.mean(align_matrix)
        return e_measure
    
    @staticmethod
    def calculate_s_measure(pred, gt):
        """Calculate Structure Measure using OpenCV"""
        s_scores = []
        
        for i in range(len(pred)):
            pred_img = (pred[i].squeeze() * 255).astype(np.uint8)
            gt_img = (gt[i].squeeze() * 255).astype(np.uint8)
            
            # Ensure same shape
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            # Calculate SSIM using OpenCV
            s_score = COD_Metrics._ssim_opencv(pred_img, gt_img)
            s_scores.append(s_score)
        
        return np.mean(s_scores)
    
    @staticmethod
    def _ssim_opencv(img1, img2):
        """Calculate SSIM using OpenCV"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)
    
    @staticmethod
    def calculate_weighted_f_measure(pred, gt, beta=0.3):
        """Calculate Weighted F-measure"""
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Calculate precision and recall
        tp = np.sum(gt * pred_binary)
        fp = np.sum((1 - gt) * pred_binary)
        fn = np.sum(gt * (1 - pred_binary))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        weighted_f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
        return weighted_f
    
    @staticmethod
    def calculate_all_metrics(model, test_loader, device):
        """Calculate all comprehensive metrics for a model"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    pred_mask = outputs[0]
                else:
                    pred_mask = outputs
                
                pred_prob = torch.sigmoid(pred_mask)
                all_predictions.extend(pred_prob.cpu().numpy())
                all_targets.extend(masks.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate all metrics
        metrics = {
            'F_measure': COD_Metrics.calculate_f_measure(predictions, targets),
            'MAE': COD_Metrics.calculate_mae(predictions, targets),
            'E_measure': COD_Metrics.calculate_e_measure(predictions, targets),
            'S_measure': COD_Metrics.calculate_s_measure(predictions, targets),
            'Weighted_F': COD_Metrics.calculate_weighted_f_measure(predictions, targets),
        }
        
        return metrics

def calculate_model_performance(model, device):
    """Calculate model size and inference speed"""
    # Parameters and size
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 ** 2)
    
    # Inference time
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Warm up
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    avg_time = (time.time() - start_time) / 100 * 1000  # ms
    
    return {
        'parameters': params,
        'model_size_mb': size_mb,
        'inference_time_ms': avg_time,
        'fps': 1000 / avg_time
    }

# Alternative simple metrics if OpenCV is also not available
class Simple_COD_Metrics:
    """Simplified metrics without any external dependencies beyond numpy"""
    
    @staticmethod
    def calculate_f_measure(pred, gt):
        """Calculate F-measure (F1 score)"""
        pred_binary = (pred > 0.5).astype(np.float32)
        return f1_score(gt.flatten(), pred_binary.flatten())
    
    @staticmethod
    def calculate_mae(pred, gt):
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(pred - gt))
    
    @staticmethod
    def calculate_e_measure(pred, gt):
        """Calculate Enhanced Alignment Measure"""
        pred_norm = (pred - np.min(pred)) / (np.max(pred) - np.min(pred) + 1e-8)
        gt_norm = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-8)
        align_matrix = 1 - np.abs(pred_norm - gt_norm)
        return np.mean(align_matrix)
    
    @staticmethod
    def calculate_s_measure(pred, gt):
        """Simple Structure Measure using gradient similarity"""
        s_scores = []
        
        for i in range(len(pred)):
            pred_img = pred[i].squeeze()
            gt_img = gt[i].squeeze()
            
            # Calculate gradients
            pred_grad = Simple_COD_Metrics._sobel_gradient(pred_img)
            gt_grad = Simple_COD_Metrics._sobel_gradient(gt_img)
            
            # Calculate gradient similarity
            grad_similarity = np.exp(-np.abs(pred_grad - gt_grad))
            s_score = np.mean(grad_similarity)
            s_scores.append(s_score)
        
        return np.mean(s_scores)
    
    @staticmethod
    def _sobel_gradient(img):
        """Calculate Sobel gradient using pure NumPy"""
        if len(img.shape) == 3:
            img = img.mean(axis=2)  # Convert to grayscale
        
        # Sobel kernels
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Pad image
        img_padded = np.pad(img, ((1, 1), (1, 1)), mode='constant')
        
        # Convolve
        grad_x = np.zeros_like(img)
        grad_y = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                grad_x[i, j] = np.sum(img_padded[i:i+3, j:j+3] * kernel_x)
                grad_y[i, j] = np.sum(img_padded[i:i+3, j:j+3] * kernel_y)
        
        return np.hypot(grad_x, grad_y)  # Gradient magnitude
    
    @staticmethod
    def calculate_all_metrics(model, test_loader, device):
        """Calculate all metrics using simple implementations"""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    pred_mask = outputs[0]
                else:
                    pred_mask = outputs
                
                pred_prob = torch.sigmoid(pred_mask)
                all_predictions.extend(pred_prob.cpu().numpy())
                all_targets.extend(masks.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        metrics = {
            'F_measure': Simple_COD_Metrics.calculate_f_measure(predictions, targets),
            'MAE': Simple_COD_Metrics.calculate_mae(predictions, targets),
            'E_measure': Simple_COD_Metrics.calculate_e_measure(predictions, targets),
            'S_measure': Simple_COD_Metrics.calculate_s_measure(predictions, targets),
        }
        
        return metrics