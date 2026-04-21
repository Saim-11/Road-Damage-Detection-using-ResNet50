"""
RDD2022 Road Damage Detection - Enhanced Research Pipeline
Module 1: Foundations of Vision and Image Analysis
Enhanced with noise modeling, restoration, and comprehensive analysis
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import logging
from scipy import stats
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoiseModel:
    """Noise modeling for simulating real-world degradation"""
    
    @staticmethod
    def add_gaussian_noise(img, mean=0, sigma=25):
        """Add Gaussian noise"""
        noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def add_salt_pepper_noise(img, amount=0.05):
        """Add salt and pepper (impulse) noise"""
        noisy = img.copy()
        num_salt = int(amount * img.size * 0.5)
        num_pepper = int(amount * img.size * 0.5)
        
        # Salt (white pixels)
        coords = [np.random.randint(0, i-1, num_salt) for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        
        # Pepper (black pixels)
        coords = [np.random.randint(0, i-1, num_pepper) for i in img.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        
        return noisy
    
    @staticmethod
    def add_speckle_noise(img, variance=0.1):
        """Add speckle (multiplicative) noise"""
        noise = np.random.randn(*img.shape) * variance
        noisy = img + img * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)


class ImageRestoration:
    """Image restoration and denoising techniques"""
    
    @staticmethod
    def gaussian_filter(img, kernel_size=5):
        """Gaussian blur filtering"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def median_filter(img, kernel_size=5):
        """Median filtering - effective for salt & pepper noise"""
        return cv2.medianBlur(img, kernel_size)
    
    @staticmethod
    def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
        """Bilateral filtering - edge-preserving smoothing"""
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    @staticmethod
    def nlm_denoise(img):
        """Non-local means denoising"""
        try:
            # Convert to float [0, 1] for denoise_nl_means
            img_float = img.astype(np.float32) / 255.0
            
            if len(img.shape) == 3:
                # For color images
                sigma_est = np.mean(estimate_sigma(img_float, channel_axis=-1))
                denoised = denoise_nl_means(img_float, h=1.15 * sigma_est, fast_mode=True,
                                           patch_size=5, patch_distance=6, channel_axis=-1)
            else:
                # For grayscale
                sigma_est = estimate_sigma(img_float)
                denoised = denoise_nl_means(img_float, h=1.15 * sigma_est, fast_mode=True,
                                           patch_size=5, patch_distance=6)
            
            # Convert back to [0, 255] uint8
            return (denoised * 255).astype(np.uint8)
        except Exception:
            # Fallback to bilateral filter if NLM fails (e.g., PyWavelets not installed)
            if len(img.shape) == 3:
                return cv2.bilateralFilter(img, 9, 75, 75)
            else:
                return cv2.bilateralFilter(img, 9, 75, 75)


class EdgeDetector:
    """Multiple edge detection methods"""
    
    @staticmethod
    def canny_edges(img, low=50, high=150):
        """Canny edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Canny(gray, low, high)
    
    @staticmethod
    def sobel_edges(img):
        """Sobel edge detection (gradient-based)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.uint8(np.clip(magnitude, 0, 255))
    
    @staticmethod
    def prewitt_edges(img):
        """Prewitt edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
        prewittx = cv2.filter2D(gray, cv2.CV_64F, kernelx)
        prewitty = cv2.filter2D(gray, cv2.CV_64F, kernely)
        magnitude = np.sqrt(prewittx**2 + prewitty**2)
        return np.uint8(np.clip(magnitude, 0, 255))
    
    @staticmethod
    def laplacian_edges(img):
        """Laplacian edge detection (second derivative)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.uint8(np.clip(np.abs(laplacian), 0, 255))
    
    @staticmethod
    def log_edges(img, sigma=1.0):
        """Laplacian of Gaussian edge detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        return np.uint8(np.clip(np.abs(log), 0, 255))


class ImageProcessor:
    """Enhanced image processing for research pipeline"""
    
    def __init__(self):
        self.results = {}
        self.noise_model = NoiseModel()
        self.restoration = ImageRestoration()
        self.edge_detector = EdgeDetector()
        
    def load_image(self, path, size=(256, 256)):
        """Load and resize image efficiently"""
        try:
            img = cv2.imread(str(path))
            if img is None:
                return None
            return cv2.resize(img, size)
        except:
            return None
    
    def compute_psnr(self, original, processed):
        """Calculate PSNR metric"""
        if original is None or processed is None:
            return None
        mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
        if mse == 0:
            return 100.0
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def compute_ssim(self, original, processed):
        """Calculate SSIM metric"""
        if original is None or processed is None:
            return None
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original
            processed_gray = processed
        return ssim(original_gray, processed_gray, data_range=255)
    
    def compute_mse(self, original, processed):
        """Calculate Mean Squared Error"""
        if original is None or processed is None:
            return None
        return np.mean((original.astype(float) - processed.astype(float)) ** 2)
    
    def geometric_transform(self, img, transform_type='rotation', **kwargs):
        """Apply various geometric transformations"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        if transform_type == 'rotation':
            angle = kwargs.get('angle', 15)
            scale = kwargs.get('scale', 0.9)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            return cv2.warpAffine(img, M, (w, h))
        
        elif transform_type == 'affine':
            pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
            pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
            M = cv2.getAffineTransform(pts1, pts2)
            return cv2.warpAffine(img, M, (w, h))
        
        elif transform_type == 'perspective':
            pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
            pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            return cv2.warpPerspective(img, M, (300, 300))
        
        else:
            return img
    
    def intensity_transform(self, img, method='clahe'):
        """Apply intensity transformations"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
        elif method == 'histogram':
            enhanced = cv2.equalizeHist(gray)
        elif method == 'gamma':
            gamma = 1.5
            enhanced = np.power(gray / 255.0, gamma) * 255.0
            enhanced = np.uint8(np.clip(enhanced, 0, 255))
        else:
            enhanced = gray
            
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else enhanced
    
    def process_single_image(self, img):
        """Process a single image with all techniques"""
        results = {}
        
        # Apply noise
        gaussian_noisy = self.noise_model.add_gaussian_noise(img)
        sp_noisy = self.noise_model.add_salt_pepper_noise(img)
        speckle_noisy = self.noise_model.add_speckle_noise(img)
        
        # Apply restoration on Gaussian noisy image
        restored_gaussian = self.restoration.gaussian_filter(gaussian_noisy)
        restored_median = self.restoration.median_filter(gaussian_noisy)
        restored_bilateral = self.restoration.bilateral_filter(gaussian_noisy)
        restored_nlm = self.restoration.nlm_denoise(gaussian_noisy)
        
        # Edge detection
        edges_canny = self.edge_detector.canny_edges(img)
        edges_sobel = self.edge_detector.sobel_edges(img)
        edges_prewitt = self.edge_detector.prewitt_edges(img)
        edges_laplacian = self.edge_detector.laplacian_edges(img)
        edges_log = self.edge_detector.log_edges(img)
        
        # Compute metrics for restoration
        results['restoration_metrics'] = {
            'gaussian_filter': {
                'psnr': float(self.compute_psnr(img, restored_gaussian)),
                'ssim': float(self.compute_ssim(img, restored_gaussian)),
                'mse': float(self.compute_mse(img, restored_gaussian))
            },
            'median_filter': {
                'psnr': float(self.compute_psnr(img, restored_median)),
                'ssim': float(self.compute_ssim(img, restored_median)),
                'mse': float(self.compute_mse(img, restored_median))
            },
            'bilateral_filter': {
                'psnr': float(self.compute_psnr(img, restored_bilateral)),
                'ssim': float(self.compute_ssim(img, restored_bilateral)),
                'mse': float(self.compute_mse(img, restored_bilateral))
            },
            'nlm_denoise': {
                'psnr': float(self.compute_psnr(img, restored_nlm)),
                'ssim': float(self.compute_ssim(img, restored_nlm)),
                'mse': float(self.compute_mse(img, restored_nlm))
            }
        }
        
        # Geometric transforms
        rotated = self.geometric_transform(img, 'rotation', angle=15, scale=0.9)
        results['geometric_psnr'] = float(self.compute_psnr(img, rotated))
        results['geometric_ssim'] = float(self.compute_ssim(img, rotated))
        
        # Edge detection metrics
        results['edge_pixels'] = {
            'canny': int(np.sum(edges_canny > 0)),
            'sobel': int(np.sum(edges_sobel > 0)),
            'prewitt': int(np.sum(edges_prewitt > 0)),
            'laplacian': int(np.sum(edges_laplacian > 0)),
            'log': int(np.sum(edges_log > 0))
        }
        
        return results
    
    def process_dataset(self, image_dir, limit=None):
        """Process dataset efficiently"""
        image_files = sorted([f for f in Path(image_dir).glob('*.jpg') if f.is_file()])
        if limit:
            image_files = image_files[:limit]
        
        logger.info(f"Processing {len(image_files)} images...")
        
        results = []
        for img_path in tqdm(image_files, desc="Module 1 Processing"):
            original = self.load_image(img_path)
            if original is None:
                continue
            
            img_results = self.process_single_image(original)
            img_results['image'] = str(img_path.name)
            results.append(img_results)
        
        return results
    
    def generate_report(self, results, output_dir):
        """Generate comprehensive analysis report"""
        if not results:
            return None
        
        # Aggregate statistics
        restoration_stats = {}
        for method in ['gaussian_filter', 'median_filter', 'bilateral_filter', 'nlm_denoise']:
            psnr_values = [r['restoration_metrics'][method]['psnr'] for r in results]
            ssim_values = [r['restoration_metrics'][method]['ssim'] for r in results]
            mse_values = [r['restoration_metrics'][method]['mse'] for r in results]
            
            restoration_stats[method] = {
                'psnr_mean': float(np.mean(psnr_values)),
                'psnr_std': float(np.std(psnr_values)),
                'ssim_mean': float(np.mean(ssim_values)),
                'ssim_std': float(np.std(ssim_values)),
                'mse_mean': float(np.mean(mse_values)),
                'mse_std': float(np.std(mse_values))
            }
        
        # Edge detection statistics
        edge_stats = {}
        for method in ['canny', 'sobel', 'prewitt', 'laplacian', 'log']:
            edge_pixels = [r['edge_pixels'][method] for r in results]
            edge_stats[method] = {
                'mean': float(np.mean(edge_pixels)),
                'std': float(np.std(edge_pixels)),
                'min': int(np.min(edge_pixels)),
                'max': int(np.max(edge_pixels))
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'restoration_techniques': restoration_stats,
            'edge_detection_methods': edge_stats,
            'geometric_transforms': {
                'psnr_mean': float(np.mean([r['geometric_psnr'] for r in results])),
                'ssim_mean': float(np.mean([r['geometric_ssim'] for r in results]))
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'module1_results.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info("Module 1 report generated")
        return report


def run_module1(data_dir, output_dir, limit=None):
    """Run Module 1 pipeline"""
    logger.info("=" * 70)
    logger.info("MODULE 1: FOUNDATIONS OF VISION AND IMAGE ANALYSIS")
    logger.info("=" * 70)
    
    processor = ImageProcessor()
    
    # Process training images
    train_dir = os.path.join(data_dir, 'train', 'images')
    if os.path.exists(train_dir):
        results = processor.process_dataset(train_dir, limit=limit)
        report = processor.generate_report(results, output_dir)
        
        logger.info(f"\n[Module 1 Complete]")
        logger.info(f"Processed: {len(results)} images")
        logger.info("\nRestoration Performance (PSNR):")
        for method, stats in report['restoration_techniques'].items():
            logger.info(f"  {method}: {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f} dB")
        
        logger.info("\nEdge Detection (Average Pixels):")
        for method, stats in report['edge_detection_methods'].items():
            logger.info(f"  {method}: {stats['mean']:.0f} pixels")
        
        return report
    else:
        logger.error(f"Dataset not found: {train_dir}")
        return None


if __name__ == '__main__':
    data_dir = "dataset"
    output_dir = "results/module1"
    
    # Test with limited images
    report = run_module1(data_dir, output_dir, limit=10)
    if report:
        print("\n✓ Module 1 executed successfully")
