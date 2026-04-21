"""
Visualization Utilities for RDD2022 Research Pipeline
Provides plotting and visualization functions for all modules
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class Module1Visualizer:
    """Visualization for Module 1: Image Processing"""
    
    @staticmethod
    def plot_restoration_comparison(results_dict, output_path):
        """Plot PSNR comparison for different restoration techniques"""
        techniques = list(results_dict['restoration_techniques'].keys())
        psnr_means = [results_dict['restoration_techniques'][t]['psnr_mean'] for t in techniques]
        psnr_stds = [results_dict['restoration_techniques'][t]['psnr_std'] for t in techniques]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(techniques))
        bars = ax.bar(x, psnr_means, yerr=psnr_stds, capsize=5, 
                      color='steelblue', edgecolor='navy', linewidth=1.5)
        
        ax.set_xlabel('Restoration Technique', fontsize=12, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
        ax.set_title('Restoration Techniques Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.replace('_', ' ').title() for t in techniques], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_edge_detection_comparison(results_dict, output_path):
        """Plot edge detection methods comparison"""
        methods = list(results_dict['edge_detection_methods'].keys())
        mean_pixels = [results_dict['edge_detection_methods'][m]['mean'] for m in methods]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
        bars = ax.barh(methods, mean_pixels, color=colors, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Average Edge Pixels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Edge Detection Method', fontsize=12, fontweight='bold')
        ax.set_title('Edge Detection Methods Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_noise_restoration_pipeline(original, noisy, restored, output_path):
        """Visualize noise and restoration pipeline"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        images = [original, noisy, restored]
        titles = ['Original', 'Noisy (Gaussian)', 'Restored (Bilateral)']
        
        for ax, img, title in zip(axes, images, titles):
            if len(img.shape) == 3:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class Module2Visualizer:
    """Visualization for Module 2: Classical Features"""
    
    @staticmethod
    def plot_feature_distribution(features_dict, output_path):
        """Plot feature dimension distribution"""
        feat_names = list(features_dict.keys())
        dimensions = [features_dict[f]['dimension'] for f in feat_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(feat_names)))
        wedges, texts, autotexts = ax.pie(dimensions, labels=[f.upper() for f in feat_names], 
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        ax.set_title('Feature Dimension Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_classifier_comparison(classifiers_dict, output_path):
        """Compare different classifiers"""
        clf_names = list(classifiers_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, clf_name in enumerate(clf_names):
            values = [classifiers_dict[clf_name][m] for m in metrics]
            ax.bar(x + i * width, values, width, label=clf_name.upper(), alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Classifier Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class Module3Visualizer:
    """Visualization for Module 3: Deep Learning"""
    
    @staticmethod
    def plot_training_history(history, output_path):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def visualize_gradcam(original_img, cam, output_path, alpha=0.5):
        """Visualize Grad-CAM heatmap overlay"""
        # Resize CAM to match image
        if isinstance(cam, np.ndarray):
            cam_resized = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
        else:
            cam_resized = cam
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        if len(original_img.shape) == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlaid = np.uint8(original_img * alpha + heatmap * (1 - alpha))
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlaid)
        axes[2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class DashboardGenerator:
    """Generate comprehensive HTML dashboard"""
    
    @staticmethod
    def generate_html_dashboard(results_dir, output_path):
        """Generate interactive HTML dashboard from results"""
        
        # Load results
        m1_results = None
        m2_results = None
        m3_results = None
        
        m1_path = os.path.join(results_dir, 'module1', 'module1_results.json')
        m2_path = os.path.join(results_dir, 'module2', 'module2_results.json')
        m3_path = os.path.join(results_dir, 'module3', 'module3_results.json')
        
        if os.path.exists(m1_path):
            with open(m1_path) as f:
                m1_results = json.load(f)
        
        if os.path.exists(m2_path):
            with open(m2_path) as f:
                m2_results = json.load(f)
        
        if os.path.exists(m3_path):
            with open(m3_path) as f:
                m3_results = json.load(f)
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RDD2022 Research Pipeline - Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            text-align: center;
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        .module-section {{
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #f8f9ff 0%, #e8eaff 100%);
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }}
        .module-section h2 {{
            color: #2d3748;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }}
        .stat-card p {{
            color: #2d3748;
            font-size: 1.8em;
            font-weight: bold;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            margin-top: 30px;
            font-size: 0.9em;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 RDD2022 Research Pipeline</h1>
        <p class="subtitle">Comprehensive Analysis Dashboard</p>
"""
        
        # Module 1 Section
        if m1_results:
            html_content += f"""
        <div class="module-section">
            <h2>📊 Module 1: Image Processing & Analysis</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Images</h3>
                    <p>{m1_results.get('total_images', 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Best PSNR</h3>
                    <p>{max([v['psnr_mean'] for v in m1_results.get('restoration_techniques', {}).values()]):.2f} dB</p>
                </div>
                <div class="stat-card">
                    <h3>Restoration Methods</h3>
                    <p>{len(m1_results.get('restoration_techniques', {}))}</p>
                </div>
                <div class="stat-card">
                    <h3>Edge Detectors</h3>
                    <p>{len(m1_results.get('edge_detection_methods', {}))}</p>
                </div>
            </div>
        </div>
"""
        
        # Module 2 Section
        if m2_results:
            html_content += f"""
        <div class="module-section">
            <h2>🎯 Module 2: Classical Feature-Based Vision</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Total Images</h3>
                    <p>{m2_results.get('total_images', 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Feature Dimension</h3>
                    <p>{m2_results.get('feature_dimension', 0)}</p>
                </div>
                <div class="stat-card">
                    <h3>Best Accuracy</h3>
                    <p>{max([v['accuracy'] for v in m2_results.get('classifiers', {}).values()]) * 100:.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Feature Types</h3>
                    <p>{len(m2_results.get('feature_statistics', {}))}</p>
                </div>
            </div>
        </div>
"""
        
        # Module 3 Section
        if m3_results:
            html_content += f"""
        <div class="module-section">
            <h2>🧠 Module 3: Deep Learning & Intelligent Vision</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>Model</h3>
                    <p style="font-size: 1.2em;">{m3_results.get('model', 'N/A')}</p>
                </div>
                <div class="stat-card">
                    <h3>Training Accuracy</h3>
                    <p>{m3_results.get('final_performance', {}).get('train_accuracy', 0):.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Validation Accuracy</h3>
                    <p>{m3_results.get('final_performance', {}).get('val_accuracy', 0):.1f}%</p>
                </div>
                <div class="stat-card">
                    <h3>Epochs Trained</h3>
                    <p>{m3_results.get('epochs', 0)}</p>
                </div>
            </div>
        </div>
"""
        
        html_content += f"""
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path


def generate_all_visualizations(results_dir, output_dir):
    """Generate all visualizations from results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Module 1 visualizations
    m1_path = os.path.join(results_dir, 'module1', 'module1_results.json')
    if os.path.exists(m1_path):
        with open(m1_path) as f:
            m1_results = json.load(f)
        
        Module1Visualizer.plot_restoration_comparison(
            m1_results, 
            os.path.join(output_dir, 'module1_restoration.png')
        )
        Module1Visualizer.plot_edge_detection_comparison(
            m1_results, 
            os.path.join(output_dir, 'module1_edges.png')
        )
    
    # Module 2 visualizations
    m2_path = os.path.join(results_dir, 'module2', 'module2_results.json')
    if os.path.exists(m2_path):
        with open(m2_path) as f:
            m2_results = json.load(f)
        
        Module2Visualizer.plot_feature_distribution(
            m2_results['feature_statistics'],
            os.path.join(output_dir, 'module2_features.png')
        )
        Module2Visualizer.plot_classifier_comparison(
            m2_results['classifiers'],
            os.path.join(output_dir, 'module2_classifiers.png')
        )
    
    # Module 3 visualizations
    m3_path = os.path.join(results_dir, 'module3', 'module3_results.json')
    if os.path.exists(m3_path):
        with open(m3_path) as f:
            m3_results = json.load(f)
        
        if 'training_history' in m3_results:
            Module3Visualizer.plot_training_history(
                m3_results['training_history'],
                os.path.join(output_dir, 'module3_training.png')
            )
    
    # Generate HTML dashboard
    dashboard_path = DashboardGenerator.generate_html_dashboard(
        results_dir,
        os.path.join(output_dir, 'dashboard.html')
    )
    
    return dashboard_path


if __name__ == '__main__':
    # Test visualization generation
    results_dir = 'results'
    output_dir = 'results/visualizations'
    
    if os.path.exists(results_dir):
        dashboard = generate_all_visualizations(results_dir, output_dir)
        print(f"✓ Visualizations generated in: {output_dir}")
        print(f"✓ Dashboard available at: {dashboard}")
