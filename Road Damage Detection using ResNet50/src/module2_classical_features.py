"""
RDD2022 Road Damage Detection - Enhanced Research Pipeline
Module 2: Classical Feature-Based Vision
Enhanced with SURF, FAST, GLCM, Hu Moments, and feature fusion
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import logging
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
except ImportError:
    from skimage.feature import local_binary_pattern
    # For older scikit-image versions
    from skimage.feature import graycomatrix, graycoprops
from skimage.measure import moments, moments_hu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Comprehensive feature extraction for research pipeline"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
        try:
            self.surf = cv2.xfeatures2d.SURF_create()
        except:
            self.surf = None
            logger.warning("SURF not available, using ORB as alternative")
            self.surf = cv2.ORB_create(nfeatures=500)
        self.fast = cv2.FastFeatureDetector_create()
        self.orb = cv2.ORB_create(nfeatures=500)
    
    def extract_sift(self, img):
        """Extract SIFT features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(des) == 0:
            return np.zeros(128)
        # Return mean descriptor
        return np.mean(des, axis=0)
    
    def extract_surf(self, img):
        """Extract SURF features (or ORB if SURF unavailable)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        try:
            kp, des = self.surf.detectAndCompute(gray, None)
            if des is None or len(des) == 0:
                return np.zeros(64)
            # Return mean descriptor (SURF typically 64 or 128 dim)
            mean_desc = np.mean(des, axis=0)
            # Ensure fixed size
            if len(mean_desc) > 64:
                return mean_desc[:64]
            elif len(mean_desc) < 64:
                return np.pad(mean_desc, (0, 64 - len(mean_desc)))
            return mean_desc
        except:
            return np.zeros(64)
    
    def extract_fast(self, img):
        """Extract FAST keypoint features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kp = self.fast.detect(gray, None)
        
        # Create feature vector from keypoint statistics
        if len(kp) == 0:
            return np.zeros(10)
        
        # Extract keypoint properties
        responses = [k.response for k in kp]
        sizes = [k.size for k in kp]
        angles = [k.angle for k in kp if k.angle != -1]
        
        features = [
            len(kp),  # Number of keypoints
            np.mean(responses) if responses else 0,
            np.std(responses) if responses else 0,
            np.mean(sizes) if sizes else 0,
            np.std(sizes) if sizes else 0,
            np.mean(angles) if angles else 0,
            np.std(angles) if angles else 0,
            np.min(responses) if responses else 0,
            np.max(responses) if responses else 0,
            len(angles) / len(kp) if kp else 0  # Ratio of oriented keypoints
        ]
        return np.array(features, dtype=np.float32)
    
    def extract_hog(self, img):
        """Extract HOG (Histogram of Oriented Gradients) features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        resized = cv2.resize(gray, (64, 128))
        
        # HOG parameters
        win_size = (64, 128)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        features = hog.compute(resized)
        
        # Return first 128 dimensions to keep size manageable
        return features[:128].flatten() if features is not None else np.zeros(128)
    
    def extract_lbp(self, img):
        """Extract LBP (Local Binary Patterns) texture features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        
        # Compute histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        # Return fixed-size feature (59 bins for uniform LBP with radius=3, points=24)
        # Pad or truncate to 64 dimensions
        if len(hist) < 64:
            hist = np.pad(hist, (0, 64 - len(hist)))
        else:
            hist = hist[:64]
        
        return hist.astype(np.float32)
    
    def extract_glcm(self, img):
        """Extract GLCM (Gray Level Co-occurrence Matrix) features"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Reduce gray levels for efficiency
        gray = (gray / 16).astype(np.uint8)
        
        # Compute GLCM for different angles
        distances = [1, 2]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray, distances, angles, levels=16, symmetric=True, normed=True)
        
        # Extract properties
        properties = ['contrast', 'homogeneity', 'energy', 'correlation', 'dissimilarity']
        features = []
        
        for prop in properties:
            prop_values = graycoprops(glcm, prop).flatten()
            features.extend(prop_values)
        
        # Total: 5 properties * (2 distances * 4 angles) = 40 features
        return np.array(features, dtype=np.float32)
    
    def extract_hu_moments(self, img):
        """Extract Hu invariant moments for shape analysis"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        # Compute moments
        m = cv2.moments(gray)
        hu = cv2.HuMoments(m).flatten()
        
        # Log transform for better scale
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
        
        return hu.astype(np.float32)  # 7 features
    
    def extract_color_histogram(self, img):
        """Extract color histogram features"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        resized = cv2.resize(img, (32, 32))
        hist_b = cv2.calcHist([resized], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([resized], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([resized], [2], None, [32], [0, 256])
        features = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        return features[:96]  # 96 features (32 * 3)
    
    def extract_all_features(self, img):
        """Extract all features and concatenate"""
        features = {}
        
        features['sift'] = self.extract_sift(img)  # 128
        features['surf'] = self.extract_surf(img)  # 64
        features['fast'] = self.extract_fast(img)  # 10
        features['hog'] = self.extract_hog(img)    # 128
        features['lbp'] = self.extract_lbp(img)    # 64
        features['glcm'] = self.extract_glcm(img)  # 40
        features['hu_moments'] = self.extract_hu_moments(img)  # 7
        features['color_hist'] = self.extract_color_histogram(img)  # 96
        
        return features
    
    def concatenate_features(self, feature_dict, selected_features=None):
        """Concatenate selected features into a single vector"""
        if selected_features is None:
            selected_features = ['sift', 'hog', 'lbp', 'glcm', 'hu_moments']
        
        concatenated = []
        for feat_name in selected_features:
            if feat_name in feature_dict:
                concatenated.extend(feature_dict[feat_name])
        
        return np.array(concatenated, dtype=np.float32)


class FeatureFusion:
    """Feature fusion strategies"""
    
    @staticmethod
    def simple_concatenation(features_list):
        """Simple concatenation of features"""
        return np.concatenate(features_list)
    
    @staticmethod
    def weighted_fusion(features_list, weights=None):
        """Weighted fusion of features"""
        if weights is None:
            weights = [1.0] * len(features_list)
        
        weighted = [f * w for f, w in zip(features_list, weights)]
        return np.concatenate(weighted)
    
    @staticmethod
    def pca_fusion(features_list, n_components=100):
        """PCA-based fusion"""
        concatenated = np.concatenate(features_list)
        if len(concatenated) <= n_components:
            return concatenated
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(concatenated.reshape(1, -1))
        return reduced.flatten()


class FeatureSelection:
    """Feature selection methods"""
    
    @staticmethod
    def variance_threshold(X, threshold=0.01):
        """Remove low-variance features"""
        selector = VarianceThreshold(threshold=threshold)
        return selector.fit_transform(X), selector
    
    @staticmethod
    def mutual_information(X, y, k=100):
        """Select top-k features by mutual information"""
        k = min(k, X.shape[1])
        selector = SelectKBest(mutual_info_classif, k=k)
        return selector.fit_transform(X, y), selector


class ClassicalClassifier:
    """Enhanced classical ML classifier"""
    
    def __init__(self, classifier_type='svm', n_samples=None):
        self.classifier_type = classifier_type
        self.scaler = StandardScaler()
        self.pca = None
        
        if classifier_type == 'svm':
            self.model = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)
        elif classifier_type == 'rf':
            # Adaptive hyperparameters based on dataset size
            if n_samples and n_samples < 50:
                # For very small datasets
                n_est = min(50, n_samples * 5)
                max_d = min(5, max(2, n_samples // 3))
                self.model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    max_features='sqrt'
                )
            else:
                # For larger datasets
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
        elif classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            # Adaptive k based on dataset size
            k = min(5, max(3, int(np.sqrt(n_samples)) if n_samples else 5))
            self.model = KNeighborsClassifier(
                n_neighbors=k,
                weights='distance',  # Weight by inverse distance
                metric='minkowski',
                p=2
            )
        elif classifier_type == 'gb':
            from sklearn.ensemble import GradientBoostingClassifier
            # Gradient Boosting with reasonable defaults
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            self.model = SVC()
    
    def train(self, X, y):
        """Train classifier with preprocessing"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA only for larger datasets (at least 50 samples)
        # Skip PCA for small datasets to avoid dimension mismatch
        if X_scaled.shape[0] >= 50 and X_scaled.shape[1] > 100:
            n_comp = min(100, X_scaled.shape[0] - 1, X_scaled.shape[1] - 1)
            if n_comp >= 2:
                self.pca = PCA(n_components=n_comp)
                X_scaled = self.pca.fit_transform(X_scaled)
        
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """Evaluate classifier"""
        y_pred = self.predict(X)
        return {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0))
        }
    
    def get_feature_importance(self):
        """Get feature importance (for Random Forest)"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


def extract_damage_label(image_path):
    """Extract damage class from RDD2022 filename format or augmented images"""
    filename = Path(image_path).stem
    
    # RDD2022 format: Country_ImageID_ClassID.jpg
    # Class IDs: D00, D10, D20, D40 (longitudinal, transverse, alligator, pothole)
    parts = filename.split('_')
    
    if len(parts) >= 3 and parts[0] != 'augmented':
        class_id = parts[2]
        # Map damage types to numeric labels
        damage_map = {
            'D00': 0,  # Longitudinal crack
            'D10': 1,  # Transverse crack
            'D20': 2,  # Alligator crack
            'D40': 3   # Pothole
        }
        return damage_map.get(class_id, 0)
    
    # For augmented images: augmented_image_N.jpg format
    # Binary classification: alternating 0 and 1
    if 'augmented' in filename:
        try:
            # Extract the number from augmented_image_N
            num_part = parts[-1]  # Get last part (the number)
            image_id = int(num_part)
            return image_id % 2  # Binary: 0 or 1
        except (ValueError, IndexError):
            pass
    
    # Fallback: alternating binary label
    return hash(filename) % 2


def run_module2(data_dir, output_dir, limit_train=None, limit_val=None):
    """Run Module 2 pipeline with comprehensive features"""
    # Set seeds for reproducibility
    from seed_utils import set_reproducible_seeds
    set_reproducible_seeds(42)
    
    logger.info("=" * 70)
    logger.info("MODULE 2: CLASSICAL FEATURE-BASED VISION")
    logger.info("=" * 70)
    
    # Load images
    train_dir = os.path.join(data_dir, 'train', 'images')
    val_dir = os.path.join(data_dir, 'val', 'images')
    
    if not os.path.exists(train_dir):
        logger.error(f"Dataset not found: {train_dir}")
        return None
    
    extractor = FeatureExtractor()
    
    logger.info("Extracting comprehensive features...")
    
    # Extract training features
    train_images = sorted([f for f in Path(train_dir).glob('*.jpg')])
    if limit_train:
        train_images = train_images[:limit_train]
    
    X_train_all = []
    feature_sets = {name: [] for name in ['sift', 'surf', 'fast', 'hog', 'lbp', 'glcm', 'hu_moments', 'color_hist']}
    
    for img_path in tqdm(train_images, desc="Training features"):
        img = cv2.imread(str(img_path))
        if img is not None:
            all_features = extractor.extract_all_features(img)
            
            # Store individual features for ablation study
            for feat_name, feat_vector in all_features.items():
                feature_sets[feat_name].append(feat_vector)
            
            # Concatenate main features
            combined = extractor.concatenate_features(all_features, 
                                                     ['sift', 'hog', 'lbp', 'glcm', 'hu_moments'])
            X_train_all.append(combined)
    
    X_train = np.array(X_train_all)
    # Extract actual labels from filenames instead of sequential
    y_train = np.array([extract_damage_label(img_path) for img_path in train_images[:len(X_train)]])
    
    # Extract validation features
    X_val_all = []
    y_val_list = []
    if os.path.exists(val_dir):
        val_images = sorted([f for f in Path(val_dir).glob('*.jpg')])
        if limit_val:
            val_images = val_images[:limit_val]
        
        for img_path in tqdm(val_images, desc="Validation features"):
            img = cv2.imread(str(img_path))
            if img is not None:
                all_features = extractor.extract_all_features(img)
                combined = extractor.concatenate_features(all_features,
                                                         ['sift', 'hog', 'lbp', 'glcm', 'hu_moments'])
                X_val_all.append(combined)
                y_val_list.append(extract_damage_label(img_path))
        
        X_val = np.array(X_val_all)
        y_val = np.array(y_val_list)
    else:
        split_idx = int(0.2 * len(X_train))
        X_val = X_train[:split_idx]
        y_val = y_train[:split_idx]
    
    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    logger.info(f"Feature dimension: {X_train.shape[1]}")
    logger.info(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}")
    
    # Train classifiers
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(X_train),
        'feature_dimension': int(X_train.shape[1]),
        'classifiers': {},
        'feature_statistics': {}
    }
    
    # Store feature statistics
    for feat_name, feat_data in feature_sets.items():
        if feat_data:
            feat_array = np.array(feat_data)
            results['feature_statistics'][feat_name] = {
                'dimension': int(feat_array.shape[1]) if len(feat_array.shape) > 1 else 1,
                'mean': float(np.mean(feat_array)),
                'std': float(np.std(feat_array))
            }
    
    # Train multiple classifiers
    for clf_type in ['svm', 'rf', 'knn', 'gb']:
        logger.info(f"\nTraining {clf_type.upper()} classifier...")
        classifier = ClassicalClassifier(clf_type, n_samples=len(X_train))
        classifier.train(X_train, y_train)
        
        metrics = classifier.evaluate(X_val, y_val)
        results['classifiers'][clf_type] = metrics
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1']:.4f}")
        
        # Feature importance for RF
        if clf_type == 'rf':
            importance = classifier.get_feature_importance()
            if importance is not None:
                results['classifiers'][clf_type]['top_features'] = {
                    'mean_importance': float(np.mean(importance)),
                    'max_importance': float(np.max(importance))
                }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'module2_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("\n[Module 2 Complete]")
    return results


if __name__ == '__main__':
    data_dir = "dataset"
    output_dir = "results/module2"
    
    # Test with limited images
    results = run_module2(data_dir, output_dir, limit_train=100, limit_val=20)
    if results:
        print("\n✓ Module 2 executed successfully")
