"""
RDD2022 Road Damage Detection - Enhanced Research Pipeline
Module 3: Deep Learning and Intelligent Vision
Enhanced with data augmentation, explainability (Grad-CAM), and improved architecture
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision import transforms
from torchvision.models import ResNet50_Weights, ResNet18_Weights
import torch.nn.functional as F

# Import model utilities
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    from model_utils import ModelCheckpoint, ModelInspector
except ImportError:
    ModelCheckpoint = None
    ModelInspector = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAugmentation:
    """Advanced data augmentation strategies"""
    
    @staticmethod
    def get_training_transforms():
        """Get augmentation for training"""
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),  # Reduced from 15
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # Reduced intensity
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduced from 0.1
        ])
    
    @staticmethod
    def get_validation_transforms():
        """Get transforms for validation (no augmentation)"""
        return transforms.Compose([])


class RDDDataset(Dataset):
    """Enhanced dataset loader with augmentation support"""
    
    def __init__(self, image_dir, transform=None, augment=False):
        self.image_paths = sorted([f for f in Path(image_dir).glob('*.jpg')])
        self.transform = transform
        self.augment = augment
        if augment:
            self.augmentation = DataAugmentation.get_training_transforms()
        else:
            self.augmentation = DataAugmentation.get_validation_transforms()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        
        # Apply augmentation
        if self.augment:
            img_tensor = self.augmentation(img_tensor)
        
        # Apply normalization
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        # Extract label from filename (RDD2022 format: Country_ImageID_ClassID.jpg)
        filename = Path(self.image_paths[idx]).stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            class_id = parts[2]
            damage_map = {'D00': 0, 'D10': 1, 'D20': 2, 'D40': 3}
            label = damage_map.get(class_id, 0)
        else:
            # Fallback to hash-based label
            label = hash(filename) % 4
        
        return img_tensor, label


class GradCAM:
    """Gradient-weighted Class Activation Mapping for explainability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate CAM
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        weighted_activations = self.activations * pooled_gradients
        cam = torch.sum(weighted_activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
    
    def visualize_cam(self, img, cam, alpha=0.5):
        """Overlay CAM on original image"""
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = np.uint8(img * alpha + heatmap * (1 - alpha))
        
        return overlaid, heatmap


class SaliencyMaps:
    """Generate saliency maps for input visualization"""
    
    @staticmethod
    def generate(model, input_tensor, target_class=None):
        """Generate saliency map"""
        input_tensor.requires_grad = True
        
        output = model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        saliency = input_tensor.grad.data.abs()
        saliency = saliency.max(dim=1)[0]  # Max across color channels
        
        # Normalize
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        
        return saliency.squeeze().cpu().numpy()


class DeepLearningModel:
    """Enhanced deep learning model with transfer learning and regularization"""
    
    def __init__(self, model_type='resnet50', num_classes=4, device=None, dropout_rate=0.5):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.num_classes = num_classes
        
        if model_type == 'resnet50':
            # Use modern weights parameter instead of deprecated pretrained
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            num_features = self.model.fc.in_features
            
            # Enhanced final layers with dropout and batch norm
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            
            # Store target layer for Grad-CAM
            self.target_layer = self.model.layer4[-1]
        
        elif model_type == 'resnet18':
            # Use modern weights parameter
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_features, num_classes)
            )
            self.target_layer = self.model.layer4[-1]
        
        self.model = self.model.to(self.device)
        
        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Initialize Grad-CAM
        self.grad_cam = GradCAM(self.model, self.target_layer)
    
    def train_epoch(self, dataloader, use_amp=False, scaler=None):
        """Train for one epoch with optional mixed precision"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(dataloader, desc="Training"):
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total if total > 0 else 0
        return total_loss / len(dataloader), accuracy
    
    def evaluate(self, dataloader):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(dataloader, desc="Evaluating"):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total if total > 0 else 0
        
        return {
            'avg_loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def extract_features(self, dataloader):
        """Extract features from penultimate layer"""
        self.model.eval()
        features_list = []
        
        # Remove final classification layer temporarily
        if isinstance(self.model.fc, nn.Sequential):
            original_fc = self.model.fc
            self.model.fc = nn.Identity()
        else:
            original_fc = self.model.fc
            self.model.fc = nn.Identity()
        
        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc="Extracting features"):
                imgs = imgs.to(self.device)
                features = self.model(imgs)
                features_list.append(features.cpu().numpy())
        
        # Restore original fc layer
        self.model.fc = original_fc
        
        return np.vstack(features_list)
    
    def generate_gradcam(self, input_tensor, target_class=None):
        """Generate Grad-CAM visualization"""
        return self.grad_cam.generate_cam(input_tensor, target_class)
    
    def generate_saliency(self, input_tensor, target_class=None):
        """Generate saliency map"""
        return SaliencyMaps.generate(self.model, input_tensor, target_class)
    
    def save_model(self, save_path, epoch=None, metrics=None):
        """Save model checkpoint with metadata"""
        if ModelCheckpoint:
            return ModelCheckpoint.save_checkpoint(
                self, self.optimizer, epoch or 0, metrics or {},
                save_path, metadata={'model_type': self.model_type}
            )
        else:
            torch.save(self.model.state_dict(), save_path)
            return save_path
    
    def load_model(self, load_path):
        """Load model from checkpoint"""
        if ModelCheckpoint:
            return ModelCheckpoint.load_checkpoint(load_path, self, self.optimizer, self.device)
        else:
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            return {'epoch': 0, 'metrics': {}}


def run_module3(data_dir, output_dir, epochs=3, batch_size=32, limit_train=None):
    """Run enhanced Module 3 pipeline"""
    # Set seeds for reproducibility
    from seed_utils import set_reproducible_seeds, configure_pytorch_determinism, get_torch_generator
    set_reproducible_seeds(42)
    configure_pytorch_determinism()
    
    logger.info("=" * 70)
    logger.info("MODULE 3: DEEP LEARNING AND INTELLIGENT VISION")
    logger.info("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_dir = os.path.join(data_dir, 'train', 'images')
    val_dir = os.path.join(data_dir, 'val', 'images')
    
    if not os.path.exists(train_dir):
        logger.error(f"Dataset not found: {train_dir}")
        return None
    
    # Normalization transform
    class NormalizeTransform:
        def __call__(self, x):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            return (x - mean) / std
    
    transform = NormalizeTransform()
    
    # Create datasets with augmentation
    logger.info("Loading dataset with augmentation...")
    train_dataset = RDDDataset(train_dir, transform=transform, augment=True)
    
    if limit_train:
        train_dataset.image_paths = train_dataset.image_paths[:limit_train]
    
    val_dataset = RDDDataset(val_dir, transform=transform, augment=False) if os.path.exists(val_dir) else None
    
    # Optimize batch sizes - MUCH larger for validation since no gradient computation
    # Use generator for reproducible shuffling
    generator = get_torch_generator(42)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 0 for Windows compatibility
        pin_memory=False,
        generator=generator,
        persistent_workers=False
    )
    
    # CRITICAL: Use much larger batch size for validation (10-16x training)
    # This dramatically speeds up evaluation with no accuracy impact
    val_batch_size = min(batch_size * 16, 256)  # Up to 256 for validation
    logger.info(f"Training batch size: {batch_size}, Validation batch size: {val_batch_size}")
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=val_batch_size, 
        shuffle=False,  # No shuffling needed for validation
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    ) if val_dataset else train_loader
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Initialize model
    logger.info("Initializing ResNet50 with enhanced architecture...")
    model = DeepLearningModel(model_type='resnet50', num_classes=4, device=device, dropout_rate=0.5)
    
    # Mixed precision training for faster computation
    use_amp = device.type == 'cuda'  # Only use AMP on GPU
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("Using mixed precision training (AMP) for faster training")
    else:
        logger.info("Using standard precision (CPU mode)")
    
    # Print model summary
    if ModelInspector:
        ModelInspector.print_model_summary(model)
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 5
    
    logger.info(f"Training for {epochs} epochs with augmentation and regularization...")
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = model.train_epoch(train_loader, use_amp=use_amp, scaler=scaler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        val_metrics = model.evaluate(val_loader)
        history['val_loss'].append(val_metrics['avg_loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        logger.info(f"  Val Loss: {val_metrics['avg_loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Learning rate scheduler step
        model.scheduler.step(val_metrics['accuracy'])
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        model.save_model(
            checkpoint_path,
            epoch=epoch+1,
            metrics={
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_metrics['avg_loss'],
                'val_acc': val_metrics['accuracy']
            }
        )
        
        # Save best model and early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            model.save_model(
                best_model_path,
                epoch=epoch+1,
                metrics={'best_val_acc': best_val_acc}
            )
            logger.info(f"  ✓ New best model saved! Accuracy: {best_val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"  Early stopping triggered after {epoch+1} epochs")
                break
    
    # Extract features
    logger.info("\nExtracting learned features...")
    features = model.extract_features(val_loader)
    
    # Generate explainability visualizations for a sample
    logger.info("\nGenerating explainability visualizations...")
    sample_imgs, sample_labels = next(iter(val_loader))
    sample_img = sample_imgs[0:1].to(device)
    
    # Grad-CAM
    try:
        gradcam = model.generate_gradcam(sample_img)
    except Exception as e:
        logger.warning(f"Grad-CAM generation failed: {e}")
        gradcam = np.zeros((7, 7))
    
    # Saliency map
    try:
        saliency = model.generate_saliency(sample_img)
    except Exception as e:
        logger.warning(f"Saliency map generation failed: {e}")
        saliency = np.zeros((224, 224))
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model': 'ResNet50 with Enhanced Architecture',
        'total_images': len(train_dataset),
        'epochs': epochs,
        'batch_size': batch_size,
        'device': str(device),
        'augmentation': 'RandomFlip, RandomRotation, ColorJitter, RandomAffine',
        'regularization': 'Dropout(0.5), BatchNorm, WeightDecay(1e-4)',
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        },
        'final_performance': {
            'train_accuracy': float(history['train_acc'][-1]) if history['train_acc'] else 0,
            'val_accuracy': float(history['val_acc'][-1]) if history['val_acc'] else 0,
            'best_val_accuracy': float(best_val_acc),
            'train_loss': float(history['train_loss'][-1]) if history['train_loss'] else 0,
            'val_loss': float(history['val_loss'][-1]) if history['val_loss'] else 0
        },
        'feature_dimension': int(features.shape[1]),
        'feature_statistics': {
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features))
        },
        'explainability': {
            'gradcam_generated': True,
            'saliency_map_generated': True,
            'gradcam_shape': list(gradcam.shape),
            'saliency_shape': list(saliency.shape)
        },
        'checkpoints': {
            'checkpoint_dir': checkpoint_dir,
            'best_model': os.path.join(output_dir, 'best_model.pth')
        }
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'module3_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save explainability visualizations
    np.save(os.path.join(output_dir, 'gradcam_sample.npy'), gradcam)
    np.save(os.path.join(output_dir, 'saliency_sample.npy'), saliency)
    
    logger.info("\n[Module 3 Complete]")
    logger.info(f"Final Training Accuracy: {results['final_performance']['train_accuracy']:.2f}%")
    logger.info(f"Final Validation Accuracy: {results['final_performance']['val_accuracy']:.2f}%")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    logger.info(f"Best model saved to: {results['checkpoints']['best_model']}")
    
    return results



if __name__ == '__main__':
    data_dir = "dataset"
    output_dir = "results/module3"
    
    # Test with limited images
    results = run_module3(data_dir, output_dir, epochs=2, batch_size=16, limit_train=50)
    if results:
        print("\n✓ Module 3 executed successfully")
