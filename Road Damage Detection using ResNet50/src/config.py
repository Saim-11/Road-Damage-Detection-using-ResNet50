"""
Configuration Management for Road Damage Detection
Centralized configuration for all experiments
"""

import yaml
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'project': {
            'name': 'Road Damage Detection',
            'dataset': 'RDD2022',
            'seed': 42
        },
        
        'paths': {
            'data_dir': 'dataset',
            'output_dir': 'results',
            'checkpoints_dir': 'checkpoints',
            'reports_dir': 'reports',
            'visualizations_dir': 'visualizations'
        },
        
        'module1': {
            'limit_images': None,  # None = all images
            'noise_types': ['gaussian', 'salt_pepper', 'poisson'],
            'noise_levels': [0.01, 0.05, 0.1],
            'restoration_methods': ['gaussian_blur', 'median_blur', 'bilateral', 'nlm'],
            'edge_detectors': ['canny', 'sobel', 'prewitt', 'laplacian']
        },
        
        'module2': {
            'limit_train': None,
            'limit_val': None,
            'features': ['sift', 'hog', 'lbp', 'glcm', 'hu_moments'],
            'classifiers': ['svm', 'rf', 'knn', 'gradient_boosting'],
            'use_pca': True,
            'pca_components': 100,
            'cross_validation_folds': 5,
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        },
        
        'module3': {
            'limit_train': None,
            'epochs': 3,
            'batch_size': 16,
            'val_batch_size_multiplier': 16,  # Validation batch size = batch_size * this
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'dropout_rate': 0.5,
            'early_stopping_patience': 5,
            'use_mixed_precision': True,
            'architectures': ['resnet50', 'resnet18', 'efficientnet_b0'],
            'augmentation': {
                'horizontal_flip_prob': 0.5,
                'rotation_degrees': 10,
                'color_jitter': {
                    'brightness': 0.1,
                    'contrast': 0.1,
                    'saturation': 0.1,
                    'hue': 0.05
                },
                'translate': [0.05, 0.05]
            },
            'scheduler': {
                'type': 'ReduceLROnPlateau',
                'mode': 'max',
                'factor': 0.5,
                'patience': 2
            }
        },
        
        'evaluation': {
            'class_names': [
                'D00 (Longitudinal Crack)',
                'D10 (Transverse Crack)',
                 'D20 (Alligator Crack)',
                'D40 (Pothole)'
            ],
            'metrics': [
                'accuracy', 'precision', 'recall', 'f1',
                'confusion_matrix', 'roc_auc', 'cohen_kappa'
            ],
            'generate_plots': True,
            'plot_formats': ['png', 'pdf']
        },
        
        'ablation_studies': {
            'module1_noise_impact': True,
            'module1_restoration_comparison': True,
            'module2_feature_importance': True,
            'module2_classifier_comparison': True,
            'module3_architecture_comparison': True,
            'module3_augmentation_impact': True,
            'hybrid_fusion_methods': True
        }
    }
    
    def __init__(self, config_path=None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Deep merge with default config
            self._deep_merge(self.config, user_config)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Using default configuration")
    
    def save_to_file(self, config_path):
        """Save current configuration to YAML file"""
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {config_path}: {e}")
    
    def _deep_merge(self, base_dict, update_dict):
        """Deep merge update_dict into base_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot notation
        
        Example:
            config.get('module3.batch_size')  # Returns 16
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path, value):
        """
        Set configuration value using dot notation
        
        Example:
            config.set('module3.epochs', 5)
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def to_dict(self):
        """Return configuration as dictionary"""
        return self.config.copy()
    
    def __getitem__(self, key):
        """Allow dict-like access"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Allow dict-like setting"""
        self.config[key] = value


def create_default_config_file(output_path='config.yaml'):
    """Create default configuration file"""
    config = Config()
    config.save_to_file(output_path)
    print(f"Default configuration created: {output_path}")
    return output_path


if __name__ == '__main__':
    # Create default config file
    create_default_config_file()
    
    # Test loading
    config = Config('config.yaml')
    print(f"Batch size: {config.get('module3.batch_size')}")
    print(f"Features: {config.get('module2.features')}")
    print("✓ Configuration module working correctly")
