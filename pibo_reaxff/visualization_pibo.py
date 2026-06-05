"""
Visualization and Analysis Module for PIBO Results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import os


class PIBOVisualizer:
    """Visualization tools for PIBO results"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.blocks = ['bond', 'angle', 'torsion', 'vdw_coulomb']
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def plot_convergence_all_blocks(self):
        """Plot convergence curves for all blocks"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, block in enumerate(self.blocks):
            ax = axes[idx]
            
            block_dir = os.path.join(self.results_dir, f'{block}_optimization')
            history_file = os.path.join(block_dir, 'parameter_history.csv')
            
            if os.path.exists(history_file):
                df = pd.read_csv(history_file)
                
                if 'loss' in df.columns:
                    # Plot convergence
                    ax.plot(df['loss'].values, alpha=0.7, linewidth=1.5)
                    
                    # Add running minimum
                    running_min = np.minimum.accumulate(df['loss'].values)
                    ax.plot(running_min, 'r-', linewidth=2, label='Best so far')
                    
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Loss')
                    ax.set_title(f'{block.upper()} Convergence')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Log scale if range is large
                    if df['loss'].max() / df['loss'].min() > 100:
                        ax.set_yscale('log')
                        
        plt.suptitle('Optimization Convergence for All Blocks', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'convergence_all_blocks.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Convergence plot saved: {output_file}")
        
    def plot_dft_comparison_summary(self):
        """Create summary of DFT vs ReaxFF comparisons"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        all_metrics = []
        
        for idx, block in enumerate(self.blocks):
            ax = axes[idx]
            
            comp_file = os.path.join(self.results_dir, f'{block}_optimization', 'dft_comparison.csv')
            
            if os.path.exists(comp_file):
                df = pd.read_csv(comp_file)
                
                if all(['DFT_Energy' in df.columns, 'ReaxFF_Energy' in df.columns]):
                    dft = df['DFT_Energy'].values
                    reaxff = df['ReaxFF_Energy'].values
                    
                    # Parity plot
                    ax.scatter(dft, reaxff, alpha=0.6, s=30)
                    
                    # Perfect agreement line
                    min_val = min(dft.min(), reaxff.min())
                    max_val = max(dft.max(), reaxff.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                    
                    # Calculate metrics
                    rmse = np.sqrt(np.mean((reaxff - dft)**2))
                    mae = np.mean(np.abs(reaxff - dft))
                    r2 = stats.pearsonr(dft, reaxff)[0]**2
                    
                    # Add text
                    text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nR²: {r2:.3f}'
                    ax.text(0.05, 0.95, text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='wheat', alpha=0.5))
                    
                    ax.set_xlabel('DFT Energy (eV)')
                    ax.set_ylabel('ReaxFF Energy (eV)')
                    ax.set_title(f'{block.upper()}')
                    ax.grid(True, alpha=0.3)
                    
                    # Store metrics
                    all_metrics.append({
                        'block': block,
                        'RMSE': rmse,
                        'MAE': mae,
                        'R2': r2,
                        'n_points': len(dft)
                    })
                    
        plt.suptitle('DFT vs ReaxFF Comparison', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'dft_comparison_summary.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DFT comparison plot saved: {output_file}")
        
        # Save metrics table
        if all_metrics:
            df_metrics = pd.DataFrame(all_metrics)
            metrics_file = os.path.join(self.results_dir, 'dft_metrics_summary.csv')
            df_metrics.to_csv(metrics_file, index=False)
            print(f"Metrics saved: {metrics_file}")
            
    def plot_parameter_evolution(self):
        """Plot how parameters evolve during optimization"""
        
        for block in self.blocks:
            block_dir = os.path.join(self.results_dir, f'{block}_optimization')
            history_file = os.path.join(block_dir, 'parameter_history.csv')
            
            if os.path.exists(history_file):
                df = pd.read_csv(history_file)
                
                # Get parameter columns (exclude 'loss')
                param_cols = [col for col in df.columns if col != 'loss']
                
                if len(param_cols) > 0:
                    n_params = len(param_cols)
                    n_cols = min(3, n_params)
                    n_rows = (n_params + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                    
                    if n_params == 1:
                        axes = [axes]
                    else:
                        axes = axes.flatten()
                    
                    for idx, param in enumerate(param_cols):
                        if idx < len(axes):
                            ax = axes[idx]
                            
                            # Plot parameter evolution
                            ax.plot(df[param].values, alpha=0.7)
                            
                            # Add running average
                            window = min(20, len(df)//5)
                            if window > 1:
                                rolling_mean = pd.Series(df[param]).rolling(window).mean()
                                ax.plot(rolling_mean, 'r-', linewidth=2, alpha=0.7)
                            
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Value')
                            ax.set_title(param)
                            ax.grid(True, alpha=0.3)
                            
                    # Hide extra subplots
                    for idx in range(n_params, len(axes)):
                        axes[idx].axis('off')
                        
                    plt.suptitle(f'{block.upper()} Parameter Evolution', fontsize=14)
                    plt.tight_layout()
                    
                    output_file = os.path.join(block_dir, 'parameter_evolution.png')
                    plt.savefig(output_file, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Parameter evolution plot saved ({block}): {output_file}")
                    
    def plot_physics_score_history(self):
        """Plot physics score evolution during optimization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, block in enumerate(self.blocks):
            ax = axes[idx]
            
            # Load physics scores if available
            score_file = os.path.join(self.results_dir, f'{block}_optimization', 'physics_scores.csv')
            
            if os.path.exists(score_file):
                df = pd.read_csv(score_file)
                
                if 'physics_score' in df.columns:
                    scores = df['physics_score'].values
                    
                    # Plot scores
                    ax.plot(scores, alpha=0.7, color='blue', label='Physics Score')
                    
                    # Add stages markers
                    stage_transitions = [30, 80, 110]  # Example stage transitions
                    for trans in stage_transitions:
                        if trans < len(scores):
                            ax.axvline(x=trans, color='red', linestyle='--', alpha=0.5)
                            
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Physics Score')
                    ax.set_title(f'{block.upper()} Physics Score Evolution')
                    ax.set_ylim([0, 1.1])
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
        plt.suptitle('Physics Score History', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(self.results_dir, 'physics_scores.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Physics score plot saved: {output_file}")
        
    def create_final_report(self):
        """Generate comprehensive HTML report"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PIBO Optimization Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; margin: 20px 0; }
                .metric { background-color: #e7f3ff; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Physics-Informed Bayesian Optimization Report</h1>
        """
        
        # Load optimized parameters
        param_file = os.path.join(self.results_dir, 'optimized_parameters.json')
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                params = json.load(f)
                
            html_content += "<h2>Optimized Parameters</h2>"
            html_content += "<table>"
            html_content += "<tr><th>Parameter</th><th>Value</th></tr>"
            
            for key, value in params.items():
                if key != 'loss':
                    html_content += f"<tr><td>{key}</td><td>{value:.6f}</td></tr>"
                    
            html_content += "</table>"
            
        # Add plots
        html_content += "<h2>Convergence Analysis</h2>"
        if os.path.exists(os.path.join(self.results_dir, 'convergence_all_blocks.png')):
            html_content += '<img src="convergence_all_blocks.png" alt="Convergence">'
            
        html_content += "<h2>DFT Comparison</h2>"
        if os.path.exists(os.path.join(self.results_dir, 'dft_comparison_summary.png')):
            html_content += '<img src="dft_comparison_summary.png" alt="DFT Comparison">'
            
        # Load metrics
        metrics_file = os.path.join(self.results_dir, 'dft_metrics_summary.csv')
        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)
            
            html_content += "<h2>Performance Metrics</h2>"
            html_content += df_metrics.to_html(index=False, float_format='%.4f')
            
        html_content += """
        </body>
        </html>
        """
        
        report_file = os.path.join(self.results_dir, 'report.html')
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        print(f"HTML report generated: {report_file}")
        
    def generate_all_visualizations(self):
        """Generate all visualizations and reports"""
        
        print("\nGenerating visualizations...")
        
        # Generate plots
        self.plot_convergence_all_blocks()
        self.plot_dft_comparison_summary()
        self.plot_parameter_evolution()
        self.plot_physics_score_history()
        
        # Generate report
        self.create_final_report()
        
        print("\nAll visualizations complete!")


def visualize_results(results_dir):
    """Main function to visualize PIBO results"""
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
        
    visualizer = PIBOVisualizer(results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find most recent results directory
        import glob
        result_dirs = glob.glob('results_*')
        if result_dirs:
            results_dir = sorted(result_dirs)[-1]
            print(f"Using most recent results: {results_dir}")
        else:
            print("No results directory found")
            sys.exit(1)
            
    visualize_results(results_dir)
