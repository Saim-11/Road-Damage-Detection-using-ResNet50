"""
Model Management Utilities for RDD2022 Research Pipeline
Handles model saving, loading, checkpointing, and architecture inspection
"""

import os
import torch
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCheckpoint:
    """Manage model checkpoints with metadata"""
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, metrics, filepath, metadata=None):
        """
        Save model checkpoint with training state and metrics
        
        Args:
            model: PyTorch model
            optimizer: Optimizer instance
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            filepath: Path to save checkpoint
            metadata: Optional additional metadata
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.model.state_dict() if hasattr(model, 'model') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
        
        # Save metadata as JSON for easy inspection
        metadata_path = filepath.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'epoch': epoch,
                'metrics': metrics,
                'timestamp': checkpoint['timestamp'],
                'metadata': metadata or {}
            }, f, indent=4)
        
        return filepath
    
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
        """
        Load model checkpoint and restore training state
        
        Args:
            filepath: Path to checkpoint file
            model: PyTorch model to load weights into
            optimizer: Optional optimizer to restore state
            device: Device to load checkpoint to
            
        Returns:
            Dictionary with epoch, metrics, and metadata
        """
        if not os.path.exists(filepath):
            logger.error(f"Checkpoint not found: {filepath}")
            return None
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model weights
        if hasattr(model, 'model'):
            model.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
        logger.info(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'metadata': checkpoint.get('metadata', {}),
            'timestamp': checkpoint.get('timestamp', 'N/A')
        }
    
    @staticmethod
    def find_latest_checkpoint(checkpoint_dir):
        """Find the most recent checkpoint in a directory"""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = sorted(checkpoint_dir.glob('*.pth'), key=lambda x: x.stat().st_mtime)
        return str(checkpoints[-1]) if checkpoints else None
    
    @staticmethod
    def list_checkpoints(checkpoint_dir):
        """List all checkpoints with their metadata"""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for ckpt_file in sorted(checkpoint_dir.glob('*.pth')):
            metadata_file = str(ckpt_file).replace('.pth', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file) as f:
                    metadata = json.load(f)
                checkpoints.append({
                    'filepath': str(ckpt_file),
                    'filename': ckpt_file.name,
                    **metadata
                })
        
        return checkpoints


class ModelInspector:
    """Inspect and analyze model architecture"""
    
    @staticmethod
    def print_model_summary(model, input_size=(3, 224, 224)):
        """Print model architecture summary"""
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        
        # Try to get the actual model if wrapped
        actual_model = model.model if hasattr(model, 'model') else model
        
        # Count parameters
        total_params = sum(p.numel() for p in actual_model.parameters())
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        
        print(f"\nModel Type: {actual_model.__class__.__name__}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        
        # Print layer structure
        print(f"\nLayer Structure:")
        print("-" * 70)
        for name, module in actual_model.named_children():
            num_params = sum(p.numel() for p in module.parameters())
            print(f"{name:30s} {module.__class__.__name__:20s} {num_params:>15,} params")
        
        print("=" * 70 + "\n")
    
    @staticmethod
    def get_model_info(model):
        """Get model information as a dictionary"""
        actual_model = model.model if hasattr(model, 'model') else model
        
        total_params = sum(p.numel() for p in actual_model.parameters())
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        
        layers = []
        for name, module in actual_model.named_children():
            layers.append({
                'name': name,
                'type': module.__class__.__name__,
                'params': sum(p.numel() for p in module.parameters())
            })
        
        return {
            'model_type': actual_model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'layers': layers
        }


class ModelExporter:
    """Export models for deployment"""
    
    @staticmethod
    def export_to_onnx(model, filepath, input_size=(1, 3, 224, 224)):
        """Export model to ONNX format"""
        try:
            import torch.onnx
            
            actual_model = model.model if hasattr(model, 'model') else model
            actual_model.eval()
            
            dummy_input = torch.randn(input_size)
            if hasattr(model, 'device'):
                dummy_input = dummy_input.to(model.device)
            
            torch.onnx.export(
                actual_model,
                dummy_input,
                filepath,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            logger.info(f"Model exported to ONNX: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    @staticmethod
    def save_model_for_inference(model, save_dir, model_name='model'):
        """Save model in a format ready for inference"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full model
        model_path = os.path.join(save_dir, f'{model_name}.pth')
        actual_model = model.model if hasattr(model, 'model') else model
        torch.save(actual_model.state_dict(), model_path)
        
        # Save model info
        info = ModelInspector.get_model_info(model)
        info_path = os.path.join(save_dir, f'{model_name}_info.json')
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        
        logger.info(f"Model saved for inference: {save_dir}")
        return model_path


class TransferLearningUtils:
    """Utilities for transfer learning"""
    
    @staticmethod
    def freeze_layers(model, freeze_until_layer=None):
        """
        Freeze model layers for transfer learning
        
        Args:
            model: PyTorch model
            freeze_until_layer: Freeze all layers before this layer name
        """
        actual_model = model.model if hasattr(model, 'model') else model
        
        if freeze_until_layer is None:
            # Freeze all layers except the last one
            for name, param in actual_model.named_parameters():
                if 'fc' not in name:  # Don't freeze the final classifier
                    param.requires_grad = False
        else:
            freeze = True
            for name, param in actual_model.named_parameters():
                if freeze_until_layer in name:
                    freeze = False
                param.requires_grad = not freeze
        
        trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in actual_model.parameters())
        
        logger.info(f"Frozen layers: {trainable:,} / {total:,} parameters trainable")
    
    @staticmethod
    def unfreeze_all(model):
        """Unfreeze all model parameters"""
        actual_model = model.model if hasattr(model, 'model') else model
        
        for param in actual_model.parameters():
            param.requires_grad = True
        
        logger.info("All layers unfrozen")


if __name__ == '__main__':
    print("Model utilities module loaded successfully")
    print("Available classes:")
    print("  - ModelCheckpoint: Save/load checkpoints")
    print("  - ModelInspector: Analyze model architecture")
    print("  - ModelExporter: Export models for deployment")
    print("  - TransferLearningUtils: Transfer learning helpers")
