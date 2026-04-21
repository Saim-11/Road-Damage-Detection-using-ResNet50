# Road Damage Detection - Computer Vision Project

## Project Overview

This is a complete Computer Vision research pipeline implementing progressive development across three modules following the project manual requirements:

- **Module 1**: Foundations of Vision and Image Analysis
- **Module 2**: Classical Feature-Based Vision  
- **Module 3**: Advanced Deep Learning and Intelligent Vision

## Dataset

**RDD2022 - Road Damage Dataset**
- Contains road images with 4 damage types: D00, D10, D20, D40
- Split: Train, Validation, Test sets
- Location: `dataset/` directory

## Project Structure

```
CV Final Project/
├── Computer Vision Project.pdf     # Project manual and requirements
├── requirements.txt                 # Python dependencies
├── run_simple.py                   # End-to-end pipeline execution
├── gui_application.py              # Interactive GUI application
├── src/                            # Source modules
│   ├── module1_image_processing.py      # Module 1: Image processing, noise, filtering
│   ├── module2_classical_features.py    # Module 2: Feature extraction, classifiers
│   ├── module3_deep_learning.py         # Module 3: Deep learning, transfer learning
│   ├── evaluation.py               # Metrics and evaluation
│   ├── model_utils.py              # Model utilities
│   ├── visualization.py            # Visualization tools
│   ├── seed_utils.py               # Reproducibility utilities
│   ├── config.py                   # Configuration
│   ├── utils.py                    # General utilities
│   └── __init__.py                 # Package init
├── dataset/                        # RDD2022 dataset
│   ├── train/                      # Training images
│   ├── val/                        # Validation images
│   └── test/                       # Test images
└── results/                        # Output results
    ├── module1/
    ├── module2/
    └── module3/
```

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_simple.py

# Or launch the interactive GUI
python gui_application.py
```

## Modules

### Module 1: Foundations of Vision and Image Analysis
- Noise modeling (Gaussian, Salt & Pepper, Speckle)
- Image restoration (Gaussian, Median, Bilateral, NLM filters)
- Edge detection (Canny, Sobel, Prewitt, Laplacian)
- Quality metrics (PSNR, SSIM, MSE)

### Module 2: Classical Feature-Based Vision
- Keypoint detection (SIFT, SURF, FAST, ORB)
- Descriptors (HOG, LBP, GLCM, Hu Moments)
- Multiple classifiers (SVM, Random Forest, kNN, Gradient Boosting)
- Feature selection and dimensionality reduction

### Module 3: Deep Learning and Intelligent Vision
- Transfer learning (ResNet50, ResNet18, EfficientNet)
- Data augmentation and regularization
- Grad-CAM explainability
- Multi-class damage classification (4 damage types)

## Usage

### Command Line
```bash
python run_simple.py
```
Executes all three modules and generates results in the `results/` directory.

### GUI Application
```bash
python gui_application.py
```
Interactive application with:
- Module 1: Image processing demo
- Module 2: Classical feature classification
- Module 3: Deep learning prediction
- Results & Comparison tab

## Results

Output files are saved in the `results/` directory:
- `results/module1/module1_results.json` - Image processing metrics
- `results/module2/module2_results.json` - Classifier performance
- `results/module3/module3_results.json` - Deep learning metrics

## Key Features

✓ **Complete pipeline** - All 3 modules fully implemented
✓ **Modular design** - Each module can run independently
✓ **Scientific validation** - Comprehensive evaluation metrics
✓ **Explainability** - Grad-CAM visualization for deep learning
✓ **Reproducibility** - Fixed random seeds for consistent results
✓ **GUI interface** - Easy-to-use interactive application
✓ **Proper damage classification** - 4-class RDD2022 classification

## Technologies

- **Python 3.8+**
- **OpenCV** - Image processing
- **Scikit-learn** - Classical ML classifiers
- **PyTorch** - Deep learning
- **NumPy, Pandas** - Data processing
- **Matplotlib, Seaborn** - Visualization
- **Tkinter** - GUI framework

## Citation

If using this project, please cite the RDD2022 dataset:
```
@dataset{rdd2022,
  title={Road Damage Detection 2022},
  year={2022}
}
```

## Contact

For questions or issues, please refer to the project manual: Computer Vision Project.pdf
