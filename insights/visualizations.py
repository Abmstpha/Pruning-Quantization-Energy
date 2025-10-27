"""Visualization utilities for experiment results analysis."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


class ResultsVisualizer:
    """Creates visualizations for pruning and quantization experiment results."""
    
    def __init__(self, results_df: pd.DataFrame, save_dir: Path = None):
        """Initialize visualizer with results DataFrame.
        
        Args:
            results_df: DataFrame containing experiment results
            save_dir: Directory to save plots (optional)
        """
        self.results_df = results_df.copy()
        self.save_dir = save_dir or Path("insights")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_model_size_comparison(self, save: bool = True) -> None:
        """Plot model size comparison across experiments."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        experiments = (
            self.results_df['Experiment'].astype(str) + 
            " (" + self.results_df['Backbone'].astype(str) + ")"
        )
        sizes = self.results_df['Model Size (MB)'].astype(float)
        
        # Color by quantization status
        colors = [
            '#2E86AB' if q in ('No', 'Fake-8') else '#A23B72' 
            for q in self.results_df.get('Quantized', ['No'] * len(self.results_df))
        ]
        
        bars = ax.bar(experiments, sizes, color=colors, alpha=0.85, 
                     edgecolor='black', linewidth=1.1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Model Size (MB)', fontsize=12)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_title('Model Size Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Legend
        legend_elements = [
            Patch(facecolor='#2E86AB', label='Not-Quantized / Fake-8'),
            Patch(facecolor='#A23B72', label='Quantized (PTQ)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "model_size_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_accuracy_comparison(self, save: bool = True) -> None:
        """Plot accuracy comparison with color-coded performance drops."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data
        experiments = (
            self.results_df['Experiment'].astype(str) + 
            " (" + self.results_df['Backbone'].astype(str) + ")"
        )
        accuracies = self.results_df['Accuracy (%)'].astype(float)
        
        # Get baseline accuracies for comparison
        baseline_map = {}
        for backbone in self.results_df['Backbone'].unique():
            baseline_rows = self.results_df[
                (self.results_df['Backbone'] == backbone) & 
                (self.results_df['Experiment'].str.contains('Baseline', na=False))
            ]
            if not baseline_rows.empty:
                baseline_map[backbone] = baseline_rows['Accuracy (%)'].iloc[0]
        
        # Color by accuracy drop
        colors = []
        for _, row in self.results_df.iterrows():
            backbone = row['Backbone']
            accuracy = row['Accuracy (%)']
            baseline = baseline_map.get(backbone, accuracy)
            drop = baseline - accuracy
            
            if drop <= 1:
                colors.append('#27AE60')  # Green: minimal drop
            elif drop <= 3:
                colors.append('#F39C12')  # Orange: moderate drop
            else:
                colors.append('#E74C3C')  # Red: significant drop
        
        bars = ax.bar(experiments, accuracies, color=colors, alpha=0.85,
                     edgecolor='black', linewidth=1.1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_title('Accuracy Comparison Across Experiments', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Legend
        legend_elements = [
            Patch(facecolor='#27AE60', label='Drop ≤ 1%'),
            Patch(facecolor='#F39C12', label='Drop 1-3%'),
            Patch(facecolor='#E74C3C', label='Drop > 3%')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "accuracy_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_energy_consumption(self, save: bool = True) -> None:
        """Plot energy consumption if data is available."""
        energy_data = self.results_df[
            self.results_df['Energy (kWh)'].fillna(0) > 0
        ]
        
        if energy_data.empty:
            print("⚠️ No energy data available for plotting")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        experiments = (
            energy_data['Experiment'].astype(str) + 
            " (" + energy_data['Backbone'].astype(str) + ")"
        )
        energy_values = energy_data['Energy (kWh)'].astype(float)
        
        bars = ax.bar(experiments, energy_values, color='#16A085', alpha=0.85,
                     edgecolor='black', linewidth=1.1)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.6f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
        ax.set_xlabel('Experiment', fontsize=12)
        ax.set_title('Energy Consumption Across Experiments', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "energy_consumption.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_efficiency_scatter(self, save: bool = True) -> None:
        """Plot accuracy vs model size scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        accuracies = self.results_df['Accuracy (%)'].astype(float)
        sizes = self.results_df['Model Size (MB)'].astype(float)
        backbones = self.results_df['Backbone'].astype(str)
        
        # Color by backbone
        backbone_colors = {'resnet18': '#1f77b4', 'mixer_b16_224': '#ff7f0e'}
        colors = [backbone_colors.get(bb, '#2ca02c') for bb in backbones]
        
        scatter = ax.scatter(sizes, accuracies, c=colors, alpha=0.7, s=100, edgecolors='black')
        
        # Add labels for each point
        for i, (size, acc, exp, bb) in enumerate(zip(sizes, accuracies, 
                                                    self.results_df['Experiment'], backbones)):
            ax.annotate(f"{exp[:15]}...", (size, acc), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Model Size (MB)', fontsize=12)
        ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        ax.set_title('Accuracy vs Model Size Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend
        legend_elements = [
            plt.scatter([], [], c=color, label=backbone, s=100, edgecolors='black')
            for backbone, color in backbone_colors.items()
        ]
        ax.legend(handles=legend_elements, title='Backbone')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / "efficiency_scatter.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, save: bool = True) -> Dict:
        """Create a comprehensive summary report."""
        summary = {}
        
        for backbone in self.results_df['Backbone'].unique():
            backbone_data = self.results_df[self.results_df['Backbone'] == backbone]
            
            if backbone_data.empty:
                continue
            
            # Best accuracy
            best_acc_row = backbone_data.nlargest(1, 'Accuracy (%)').iloc[0]
            
            # Smallest model
            smallest_row = backbone_data.nsmallest(1, 'Model Size (MB)').iloc[0]
            
            # Best efficiency (accuracy/size ratio)
            backbone_data_copy = backbone_data.copy()
            backbone_data_copy['efficiency'] = (
                backbone_data_copy['Accuracy (%)'] / backbone_data_copy['Model Size (MB)']
            )
            best_efficiency_row = backbone_data_copy.nlargest(1, 'efficiency').iloc[0]
            
            summary[backbone] = {
                'best_accuracy': {
                    'experiment': best_acc_row['Experiment'],
                    'accuracy': best_acc_row['Accuracy (%)'],
                    'size_mb': best_acc_row['Model Size (MB)']
                },
                'smallest_model': {
                    'experiment': smallest_row['Experiment'],
                    'accuracy': smallest_row['Accuracy (%)'],
                    'size_mb': smallest_row['Model Size (MB)']
                },
                'best_efficiency': {
                    'experiment': best_efficiency_row['Experiment'],
                    'accuracy': best_efficiency_row['Accuracy (%)'],
                    'size_mb': best_efficiency_row['Model Size (MB)'],
                    'efficiency_score': best_efficiency_row['efficiency']
                }
            }
        
        if save:
            # Save summary as text report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.save_dir / f"summary_report_{timestamp}.txt"
            
            with open(report_path, 'w') as f:
                f.write("PRUNING & QUANTIZATION EXPERIMENT SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                
                for backbone, data in summary.items():
                    f.write(f"[{backbone.upper()}]\n")
                    f.write(f"  Best Accuracy: {data['best_accuracy']['experiment']} "
                           f"({data['best_accuracy']['accuracy']:.2f}%)\n")
                    f.write(f"  Smallest Model: {data['smallest_model']['experiment']} "
                           f"({data['smallest_model']['size_mb']:.2f} MB)\n")
                    f.write(f"  Best Efficiency: {data['best_efficiency']['experiment']} "
                           f"(Score: {data['best_efficiency']['efficiency_score']:.2f})\n\n")
        
        return summary
    
    def generate_all_plots(self) -> None:
        """Generate all visualization plots."""
        print("Generating visualization plots...")
        
        self.plot_model_size_comparison()
        self.plot_accuracy_comparison()
        self.plot_energy_consumption()
        self.plot_efficiency_scatter()
        
        summary = self.create_summary_report()
        
        print(f"✅ All plots saved to: {self.save_dir}")
        return summary
