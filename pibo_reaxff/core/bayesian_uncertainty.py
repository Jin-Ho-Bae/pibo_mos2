"""
Bayesian Uncertainty Quantification Module for PIBO ReaxFF
Provides parameter posterior distribution and model prediction uncertainty
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal
import json
import os
from typing import Dict, List, Tuple, Optional
import pickle


class BayesianUncertaintyQuantifier:
    """
    Handles uncertainty quantification for Bayesian optimization results
    """
    
    def __init__(self, gp_model=None, X_observed=None, y_observed=None, param_bounds=None, param_names=None):
        """
        Initialize uncertainty quantifier
        
        Args:
            gp_model: Trained Gaussian Process model
            X_observed: Observed parameter values
            y_observed: Observed objective function values
            param_bounds: Parameter bounds
            param_names: Parameter names
        """
        self.gp_model = gp_model
        self.X_observed = X_observed
        self.y_observed = y_observed
        self.param_bounds = param_bounds
        self.param_names = param_names
        
        # Storage for uncertainty metrics
        self.posterior_samples = None
        self.posterior_mean = None
        self.posterior_std = None
        self.acquisition_values = None
        self.prediction_uncertainties = None
        
    def compute_posterior_distribution(self, n_grid=50):
        """
        Compute posterior distribution over parameter space
        
        Args:
            n_grid: Number of grid points per dimension for visualization
            
        Returns:
            Dictionary containing posterior statistics
        """
        if self.gp_model is None:
            raise ValueError("GP model not available")
            
        n_params = len(self.param_bounds)
        
        # Create grid for each parameter
        param_grids = []
        for (lb, ub) in self.param_bounds:
            param_grids.append(np.linspace(lb, ub, n_grid))
            
        # For high-dimensional spaces, use sampling instead of full grid
        if n_params <= 3:
            # Full grid for low dimensions
            grids = np.meshgrid(*param_grids)
            X_grid = np.column_stack([g.ravel() for g in grids])
        else:
            # Latin hypercube sampling for high dimensions
            from pyDOE import lhs
            n_samples = min(10000, n_grid ** min(n_params, 3))
            X_unit = lhs(n_params, samples=n_samples)
            
            # Scale to bounds
            lb = np.array([b[0] for b in self.param_bounds])
            ub = np.array([b[1] for b in self.param_bounds])
            X_grid = lb + X_unit * (ub - lb)
            
        # Compute posterior mean and standard deviation
        posterior_mean, posterior_std = self.gp_model.predict(X_grid, return_std=True)
        
        # Store results
        self.posterior_mean = posterior_mean
        self.posterior_std = posterior_std
        
        posterior_stats = {
            'mean': posterior_mean,
            'std': posterior_std,
            'lower_95': posterior_mean - 1.96 * posterior_std,
            'upper_95': posterior_mean + 1.96 * posterior_std,
            'X_grid': X_grid,
            'param_names': self.param_names
        }
        
        return posterior_stats
        
    def sample_posterior_parameters(self, n_samples=1000, around_best=True):
        """
        Sample parameters from the posterior distribution
        
        Args:
            n_samples: Number of samples to generate
            around_best: If True, sample around the best observed point
            
        Returns:
            Array of sampled parameters
        """
        if self.X_observed is None or self.y_observed is None:
            raise ValueError("No observed data available")
            
        # Find best observed point
        best_idx = np.argmax(self.y_observed)
        best_params = self.X_observed[best_idx]
        
        if around_best:
            # Sample around the best point using local GP approximation
            # Estimate local covariance from nearby points
            distances = np.linalg.norm(self.X_observed - best_params, axis=1)
            nearby_idx = np.argsort(distances)[:min(20, len(distances))]
            X_nearby = self.X_observed[nearby_idx]
            
            # Compute empirical covariance
            if len(X_nearby) > 1:
                cov_matrix = np.cov(X_nearby.T)
                # Add small diagonal for numerical stability
                cov_matrix += np.eye(len(self.param_bounds)) * 1e-6
                
                # Sample from multivariate normal
                samples = multivariate_normal.rvs(
                    mean=best_params,
                    cov=cov_matrix * 0.1,  # Scale down covariance
                    size=n_samples
                )
            else:
                # Fallback to independent sampling
                samples = self._sample_independent(n_samples, best_params)
        else:
            # Sample from full parameter space
            samples = self._sample_from_gp_posterior(n_samples)
            
        # Clip to bounds
        samples = self._clip_to_bounds(samples)
        
        self.posterior_samples = samples
        return samples
        
    def _sample_independent(self, n_samples, center_point):
        """Sample independently around a center point"""
        n_params = len(self.param_bounds)
        samples = np.zeros((n_samples, n_params))
        
        for i, (lb, ub) in enumerate(self.param_bounds):
            # Use truncated normal distribution
            std = (ub - lb) * 0.1  # 10% of range as std
            samples[:, i] = stats.truncnorm.rvs(
                (lb - center_point[i]) / std,
                (ub - center_point[i]) / std,
                loc=center_point[i],
                scale=std,
                size=n_samples
            )
            
        return samples
        
    def _sample_from_gp_posterior(self, n_samples):
        """Sample from the full GP posterior"""
        from pyDOE import lhs
        
        n_params = len(self.param_bounds)
        
        # Generate candidate points
        n_candidates = n_samples * 10
        X_candidates = lhs(n_params, samples=n_candidates)
        
        # Scale to bounds
        lb = np.array([b[0] for b in self.param_bounds])
        ub = np.array([b[1] for b in self.param_bounds])
        X_candidates = lb + X_candidates * (ub - lb)
        
        # Get posterior mean and std
        mean, std = self.gp_model.predict(X_candidates, return_std=True)
        
        # Sample from posterior using Thompson sampling
        samples_idx = []
        for _ in range(n_samples):
            # Sample function values
            f_samples = mean + std * np.random.randn(len(mean))
            # Select best
            best_idx = np.argmax(f_samples)
            samples_idx.append(best_idx)
            
        samples = X_candidates[samples_idx]
        return samples
        
    def _clip_to_bounds(self, samples):
        """Clip samples to parameter bounds"""
        clipped = samples.copy()
        for i, (lb, ub) in enumerate(self.param_bounds):
            clipped[:, i] = np.clip(clipped[:, i], lb, ub)
        return clipped
        
    def compute_prediction_uncertainty(self, X_test=None, n_test=1000):
        """
        Compute prediction uncertainty at test points
        
        Args:
            X_test: Test points (if None, generate random test points)
            n_test: Number of test points to generate
            
        Returns:
            Dictionary with prediction statistics
        """
        if X_test is None:
            # Generate test points
            from pyDOE import lhs
            n_params = len(self.param_bounds)
            X_unit = lhs(n_params, samples=n_test)
            
            lb = np.array([b[0] for b in self.param_bounds])
            ub = np.array([b[1] for b in self.param_bounds])
            X_test = lb + X_unit * (ub - lb)
            
        # Get predictions with uncertainty
        mean, std = self.gp_model.predict(X_test, return_std=True)
        
        # Compute confidence intervals
        lower_95 = mean - 1.96 * std
        upper_95 = mean + 1.96 * std
        lower_68 = mean - std
        upper_68 = mean + std
        
        # Compute coefficient of variation
        cv = std / np.abs(mean + 1e-10)
        
        prediction_stats = {
            'X_test': X_test,
            'mean': mean,
            'std': std,
            'lower_95': lower_95,
            'upper_95': upper_95,
            'lower_68': lower_68,
            'upper_68': upper_68,
            'coefficient_variation': cv
        }
        
        self.prediction_uncertainties = prediction_stats
        return prediction_stats
        
    def compute_acquisition_landscape(self, best_y, n_test=5000):
        """
        Compute acquisition function values across parameter space
        
        Args:
            best_y: Best observed objective value
            n_test: Number of test points
            
        Returns:
            Dictionary with acquisition function values
        """
        from pyDOE import lhs
        
        n_params = len(self.param_bounds)
        
        # Generate test points
        X_test = lhs(n_params, samples=n_test)
        lb = np.array([b[0] for b in self.param_bounds])
        ub = np.array([b[1] for b in self.param_bounds])
        X_test = lb + X_test * (ub - lb)
        
        # Predict
        mean, std = self.gp_model.predict(X_test, return_std=True)
        
        # Compute Expected Improvement
        ei = self._expected_improvement(mean, std, best_y)
        
        # Compute Probability of Improvement
        poi = self._probability_of_improvement(mean, std, best_y)
        
        # Compute Upper Confidence Bound
        ucb = self._upper_confidence_bound(mean, std, kappa=2.0)
        
        acquisition_values = {
            'X_test': X_test,
            'expected_improvement': ei,
            'probability_improvement': poi,
            'upper_confidence_bound': ucb,
            'posterior_mean': mean,
            'posterior_std': std
        }
        
        self.acquisition_values = acquisition_values
        return acquisition_values
        
    def _expected_improvement(self, mean, std, best_y, xi=0.01):
        """Calculate Expected Improvement"""
        improvement = mean - best_y - xi
        Z = improvement / (std + 1e-10)
        ei = improvement * stats.norm.cdf(Z) + std * stats.norm.pdf(Z)
        ei[std == 0] = 0
        return ei
        
    def _probability_of_improvement(self, mean, std, best_y, xi=0.01):
        """Calculate Probability of Improvement"""
        Z = (mean - best_y - xi) / (std + 1e-10)
        poi = stats.norm.cdf(Z)
        return poi
        
    def _upper_confidence_bound(self, mean, std, kappa=2.0):
        """Calculate Upper Confidence Bound"""
        return mean + kappa * std
        
    def save_uncertainty_data(self, output_dir, block_name):
        """
        Save all uncertainty quantification data
        
        Args:
            output_dir: Directory to save results
            block_name: Name of the optimization block
        """
        uncertainty_dir = os.path.join(output_dir, 'uncertainty_quantification')
        os.makedirs(uncertainty_dir, exist_ok=True)
        
        # Save posterior samples
        if self.posterior_samples is not None:
            df_samples = pd.DataFrame(self.posterior_samples, columns=self.param_names)
            df_samples.to_csv(
                os.path.join(uncertainty_dir, f'{block_name}_posterior_samples.csv'),
                index=False
            )
            
        # Save posterior statistics
        if self.posterior_mean is not None:
            posterior_stats = {
                'mean': self.posterior_mean.tolist() if isinstance(self.posterior_mean, np.ndarray) else self.posterior_mean,
                'std': self.posterior_std.tolist() if isinstance(self.posterior_std, np.ndarray) else self.posterior_std,
                'param_names': self.param_names,
                'param_bounds': self.param_bounds
            }
            
            with open(os.path.join(uncertainty_dir, f'{block_name}_posterior_stats.json'), 'w') as f:
                json.dump(posterior_stats, f, indent=2)
                
        # Save prediction uncertainties
        if self.prediction_uncertainties is not None:
            df_pred = pd.DataFrame({
                'mean': self.prediction_uncertainties['mean'],
                'std': self.prediction_uncertainties['std'],
                'cv': self.prediction_uncertainties['coefficient_variation']
            })
            df_pred.to_csv(
                os.path.join(uncertainty_dir, f'{block_name}_prediction_uncertainty.csv'),
                index=False
            )
            
        # Save acquisition values
        if self.acquisition_values is not None:
            df_acq = pd.DataFrame({
                'ei': self.acquisition_values['expected_improvement'],
                'poi': self.acquisition_values['probability_improvement'],
                'ucb': self.acquisition_values['upper_confidence_bound']
            })
            df_acq.to_csv(
                os.path.join(uncertainty_dir, f'{block_name}_acquisition_values.csv'),
                index=False
            )
            
        # Save GP model
        if self.gp_model is not None:
            model_file = os.path.join(uncertainty_dir, f'{block_name}_gp_model.pkl')
            with open(model_file, 'wb') as f:
                pickle.dump(self.gp_model, f)
                
        print(f"Uncertainty data saved in: {uncertainty_dir}")
        
    def plot_uncertainty_diagnostics(self, output_dir, block_name):
        """
        Create diagnostic plots for uncertainty quantification
        
        Args:
            output_dir: Directory to save plots
            block_name: Name of the optimization block
        """
        plot_dir = os.path.join(output_dir, 'uncertainty_plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Parameter posterior distributions
        if self.posterior_samples is not None:
            self._plot_posterior_distributions(plot_dir, block_name)
            
        # 2. Prediction uncertainty
        if self.prediction_uncertainties is not None:
            self._plot_prediction_uncertainty(plot_dir, block_name)
            
        # 3. Acquisition function landscape
        if self.acquisition_values is not None:
            self._plot_acquisition_landscape(plot_dir, block_name)
            
        # 4. Correlation matrix
        if self.posterior_samples is not None:
            self._plot_correlation_matrix(plot_dir, block_name)
            
    def _plot_posterior_distributions(self, plot_dir, block_name):
        """Plot posterior distributions for each parameter"""
        n_params = self.posterior_samples.shape[1]
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Prepare data for CSV export
        csv_data = {}
        kde_data = {}
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
            
        for i, param_name in enumerate(self.param_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Get histogram data
            counts, bins, _ = ax.hist(self.posterior_samples[:, i], bins=30, density=True, 
                                     alpha=0.7, edgecolor='black')
            
            # Calculate KDE
            kde = stats.gaussian_kde(self.posterior_samples[:, i])
            x_range = np.linspace(self.posterior_samples[:, i].min(),
                                self.posterior_samples[:, i].max(), 100)
            kde_values = kde(x_range)
            ax.plot(x_range, kde_values, 'r-', linewidth=2, label='KDE')
            
            # Add best value
            best_idx = np.argmax(self.y_observed)
            best_val = self.X_observed[best_idx, i]
            ax.axvline(best_val, color='green', linestyle='--', 
                      linewidth=2, label='Best')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Posterior: {param_name}')
            ax.legend()
            
            # Store data for CSV
            csv_data[f'{param_name}_samples'] = self.posterior_samples[:, i]
            csv_data[f'{param_name}_best'] = best_val
            kde_data[f'{param_name}_kde_x'] = x_range
            kde_data[f'{param_name}_kde_y'] = kde_values
            csv_data[f'{param_name}_hist_bins'] = bins[:-1]  # Exclude the last edge
            csv_data[f'{param_name}_hist_counts'] = counts
            
        # Remove empty subplots
        if n_params < n_rows * n_cols:
            for i in range(n_params, n_rows * n_cols):
                row = i // n_cols
                col = i % n_cols
                fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
                
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{block_name}_posterior_distributions.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        # Save samples data
        samples_df = pd.DataFrame(self.posterior_samples, columns=self.param_names)
        samples_df['best_idx'] = 0
        samples_df.loc[0, 'best_idx'] = 1  # Mark the best sample
        for i, param_name in enumerate(self.param_names):
            samples_df[f'{param_name}_best_value'] = csv_data[f'{param_name}_best']
        samples_df.to_csv(os.path.join(plot_dir, f'{block_name}_posterior_distributions.csv'), index=False)
        
        # Save KDE data separately
        kde_df = pd.DataFrame(kde_data)
        kde_df.to_csv(os.path.join(plot_dir, f'{block_name}_posterior_kde.csv'), index=False)
        
    def _plot_prediction_uncertainty(self, plot_dir, block_name):
        """Plot prediction uncertainty diagnostics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean vs Std
        ax = axes[0, 0]
        scatter = ax.scatter(self.prediction_uncertainties['mean'],
                           self.prediction_uncertainties['std'],
                           c=self.prediction_uncertainties['coefficient_variation'],
                           cmap='viridis', alpha=0.6)
        ax.set_xlabel('Predicted Mean')
        ax.set_ylabel('Predicted Std')
        ax.set_title('Prediction Uncertainty')
        plt.colorbar(scatter, ax=ax, label='CV')
        
        # Confidence intervals
        ax = axes[0, 1]
        sorted_idx = np.argsort(self.prediction_uncertainties['mean'])
        mean_sorted = self.prediction_uncertainties['mean'][sorted_idx]
        lower_95 = self.prediction_uncertainties['lower_95'][sorted_idx]
        upper_95 = self.prediction_uncertainties['upper_95'][sorted_idx]
        lower_68 = self.prediction_uncertainties['lower_68'][sorted_idx]
        upper_68 = self.prediction_uncertainties['upper_68'][sorted_idx]
        
        ax.plot(range(len(mean_sorted)), mean_sorted, 'b-', label='Mean')
        ax.fill_between(range(len(mean_sorted)), lower_95, upper_95,
                        alpha=0.3, label='95% CI')
        ax.set_xlabel('Sample Index (sorted)')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Confidence Intervals')
        ax.legend()
        
        # Histogram of uncertainties
        ax = axes[1, 0]
        std_counts, std_bins, _ = ax.hist(self.prediction_uncertainties['std'], bins=30,
                                         edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Std')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Uncertainties')
        
        # CV distribution
        ax = axes[1, 1]
        cv_counts, cv_bins, _ = ax.hist(self.prediction_uncertainties['coefficient_variation'],
                                       bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Coefficient of Variation')
        ax.set_ylabel('Frequency')
        ax.set_title('Relative Uncertainty Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{block_name}_prediction_uncertainty.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        # Main uncertainty data
        uncertainty_df = pd.DataFrame({
            'mean': self.prediction_uncertainties['mean'].flatten(),
            'std': self.prediction_uncertainties['std'].flatten(),
            'coefficient_variation': self.prediction_uncertainties['coefficient_variation'].flatten(),
            'lower_95': self.prediction_uncertainties['lower_95'].flatten(),
            'upper_95': self.prediction_uncertainties['upper_95'].flatten(),
            'lower_68': self.prediction_uncertainties['lower_68'].flatten(),
            'upper_68': self.prediction_uncertainties['upper_68'].flatten()
        })
        
        # Add X_test data if available and has compatible dimensions
        if self.prediction_uncertainties['X_test'].shape[0] == len(uncertainty_df):
            for i in range(self.prediction_uncertainties['X_test'].shape[1]):
                uncertainty_df[f'X_test_{i}'] = self.prediction_uncertainties['X_test'][:, i]
        
        uncertainty_df.to_csv(os.path.join(plot_dir, f'{block_name}_prediction_uncertainty.csv'), index=False)
        
        # Save confidence intervals sorted data
        ci_df = pd.DataFrame({
            'sorted_index': range(len(mean_sorted)),
            'mean_sorted': mean_sorted.flatten(),
            'lower_95_sorted': lower_95.flatten(),
            'upper_95_sorted': upper_95.flatten(),
            'lower_68_sorted': lower_68.flatten(),
            'upper_68_sorted': upper_68.flatten()
        })
        ci_df.to_csv(os.path.join(plot_dir, f'{block_name}_confidence_intervals.csv'), index=False)
        
        # Save histogram data
        hist_df = pd.DataFrame({
            'std_bins': std_bins[:-1],
            'std_counts': std_counts,
            'cv_bins': cv_bins[:-1],
            'cv_counts': cv_counts
        })
        hist_df.to_csv(os.path.join(plot_dir, f'{block_name}_uncertainty_histograms.csv'), index=False)
        
    def _plot_acquisition_landscape(self, plot_dir, block_name):
        """Plot acquisition function landscapes"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Expected Improvement
        ax = axes[0, 0]
        scatter = ax.scatter(self.acquisition_values['posterior_mean'],
                           self.acquisition_values['expected_improvement'],
                           c=self.acquisition_values['posterior_std'],
                           cmap='coolwarm', alpha=0.5, s=10)
        ax.set_xlabel('Posterior Mean')
        ax.set_ylabel('Expected Improvement')
        ax.set_title('EI Landscape')
        plt.colorbar(scatter, ax=ax, label='Std')
        
        # Probability of Improvement
        ax = axes[0, 1]
        ax.scatter(self.acquisition_values['posterior_mean'],
                  self.acquisition_values['probability_improvement'],
                  alpha=0.5, s=10)
        ax.set_xlabel('Posterior Mean')
        ax.set_ylabel('Probability of Improvement')
        ax.set_title('POI Landscape')
        
        # Upper Confidence Bound
        ax = axes[1, 0]
        ax.scatter(self.acquisition_values['posterior_mean'],
                  self.acquisition_values['upper_confidence_bound'],
                  alpha=0.5, s=10)
        ax.set_xlabel('Posterior Mean')
        ax.set_ylabel('Upper Confidence Bound')
        ax.set_title('UCB Landscape')
        
        # Acquisition values distribution
        ax = axes[1, 1]
        ei_counts, ei_bins, _ = ax.hist(self.acquisition_values['expected_improvement'],
                                       bins=30, label='EI', alpha=0.7)
        poi_counts, poi_bins, _ = ax.hist(self.acquisition_values['probability_improvement'],
                                         bins=30, label='POI', alpha=0.7)
        ax.set_xlabel('Acquisition Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Acquisition Values Distribution')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{block_name}_acquisition_landscape.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save data to CSV
        # Main acquisition data
        acq_df = pd.DataFrame({
            'posterior_mean': self.acquisition_values['posterior_mean'].flatten(),
            'posterior_std': self.acquisition_values['posterior_std'].flatten(),
            'expected_improvement': self.acquisition_values['expected_improvement'].flatten(),
            'probability_improvement': self.acquisition_values['probability_improvement'].flatten(),
            'upper_confidence_bound': self.acquisition_values['upper_confidence_bound'].flatten()
        })
        
        # Add X_test data if available
        if self.acquisition_values['X_test'].shape[0] == len(acq_df):
            for i in range(self.acquisition_values['X_test'].shape[1]):
                acq_df[f'X_test_{i}'] = self.acquisition_values['X_test'][:, i]
        
        acq_df.to_csv(os.path.join(plot_dir, f'{block_name}_acquisition_landscape.csv'), index=False)
        
        # Save histogram data
        # Pad arrays with NaN if they have different lengths
        max_len = max(len(ei_counts), len(poi_counts))
        ei_counts_padded = np.pad(ei_counts, (0, max_len - len(ei_counts)), constant_values=np.nan)
        poi_counts_padded = np.pad(poi_counts, (0, max_len - len(poi_counts)), constant_values=np.nan)
        ei_bins_padded = np.pad(ei_bins[:-1], (0, max_len - len(ei_bins[:-1])), constant_values=np.nan)
        poi_bins_padded = np.pad(poi_bins[:-1], (0, max_len - len(poi_bins[:-1])), constant_values=np.nan)
        
        hist_df = pd.DataFrame({
            'ei_bins': ei_bins_padded,
            'ei_counts': ei_counts_padded,
            'poi_bins': poi_bins_padded,
            'poi_counts': poi_counts_padded
        })
        hist_df.to_csv(os.path.join(plot_dir, f'{block_name}_acquisition_histograms.csv'), index=False)
        
    def _plot_correlation_matrix(self, plot_dir, block_name):
        """Plot parameter correlation matrix"""
        if len(self.param_names) < 2:
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Compute correlation matrix
        df_samples = pd.DataFrame(self.posterior_samples, columns=self.param_names)
        corr_matrix = df_samples.corr()
        
        # Compute covariance matrix as well
        cov_matrix = df_samples.cov()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        
        ax.set_title(f'Parameter Correlation Matrix - {block_name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{block_name}_correlation_matrix.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix to CSV
        corr_matrix.to_csv(os.path.join(plot_dir, f'{block_name}_correlation_matrix.csv'))
        
        # Save covariance matrix to CSV
        cov_matrix.to_csv(os.path.join(plot_dir, f'{block_name}_covariance_matrix.csv'))
        
        # Save summary statistics
        stats_df = pd.DataFrame({
            'parameter': self.param_names,
            'mean': df_samples.mean().values,
            'std': df_samples.std().values,
            'min': df_samples.min().values,
            'max': df_samples.max().values,
            'median': df_samples.median().values,
            'q25': df_samples.quantile(0.25).values,
            'q75': df_samples.quantile(0.75).values,
            'skewness': df_samples.skew().values,
            'kurtosis': df_samples.kurtosis().values
        })
        stats_df.to_csv(os.path.join(plot_dir, f'{block_name}_parameter_statistics.csv'), index=False)
