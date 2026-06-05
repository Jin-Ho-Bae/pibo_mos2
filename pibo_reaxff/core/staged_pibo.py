"""
Staged Physics-Informed Bayesian Optimization for ReaxFF
Implements multi-stage optimization with block search and physics-aware acquisition
"""

import numpy as np
from scipy.stats import norm
from pyDOE import lhs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os
from datetime import datetime


class PhysicsInformedBO:
    """Physics-Informed Bayesian Optimization with staged strategy"""
    
    def __init__(self, block_type, param_bounds, param_names, config=None):
        self.block_type = block_type
        self.param_bounds = np.array(param_bounds)
        self.param_names = param_names
        self.n_params = len(param_bounds)
        
        # Configuration
        self.config = config or {}
        self.stage = 0  # Current optimization stage
        self.iteration = 0
        
        # Data storage
        self.X_observed = []
        self.y_observed = []
        self.physics_scores = []
        self.energy_components = {}
        
        # Stage parameters
        self.stage_config = {
            0: {'name': 'warm_start', 'n_init': 500, 'physics_weight': 0.3},
            1: {'name': 'coarse', 'physics_weight': 0.7, 'precision': 'low'},
            2: {'name': 'refine', 'physics_weight': 0.5, 'precision': 'high'},
            3: {'name': 'polish', 'physics_weight': 0.2, 'precision': 'highest'}
        }
        
    def physics_score(self, params):
        """Calculate physics-based score for parameter set"""
        score = 1.0
        
        # Bond parameters physics constraints
        if self.block_type == 'bond':
            # Check bond dissociation energy ordering
            if 'De_MoS' in self.param_names:
                idx = self.param_names.index('De_MoS')
                De_MoS = params[idx]
                # Mo-S should be strong
                if De_MoS < 50 or De_MoS > 80:
                    score *= 0.5
                    
        # Angle parameters physics constraints  
        elif self.block_type == 'angle':
            # Check equilibrium angles
            if 'Theta0_SMoS' in self.param_names:
                idx = self.param_names.index('Theta0_SMoS')
                theta = params[idx]
                # Should be close to tetrahedral/octahedral
                if abs(theta - 90) > 20:
                    score *= 0.7
                    
        # Charge-related constraints
        if any('charge' in name for name in self.param_names):
            # Ensure charge neutrality
            total_charge = sum(params[i] for i, name in enumerate(self.param_names) 
                             if 'charge' in name)
            if abs(total_charge) > 0.1:
                score *= 0.3
                
        return score
    
    def acquisition_physics_aware(self, mean, std, best_y, params_batch):
        """Physics-aware acquisition function"""
        # Standard EI
        z = (mean - best_y - 0.01) / (std + 1e-9)
        ei = (mean - best_y - 0.01) * norm.cdf(z) + std * norm.pdf(z)
        
        # Physics weighting
        physics_weight = self.stage_config[self.stage]['physics_weight']
        physics_scores = np.array([self.physics_score(p) for p in params_batch])
        
        # Combined acquisition
        ei_physics = ei * (1 - physics_weight + physics_weight * physics_scores)
        
        return ei_physics, physics_scores
    
    def advance_stage(self):
        """Advance to next optimization stage"""
        if self.stage < 3:
            self.stage += 1
            print(f"Advanced to stage {self.stage}: {self.stage_config[self.stage]['name']}")
            return True
        return False
    
    def get_stage_samples(self):
        """Get number of samples for current stage"""
        if self.stage == 0:
            return self.stage_config[0]['n_init']
        elif self.stage == 1:
            return 1000  # 증가: 50 -> 1000
        elif self.stage == 2:
            return 500   # 증가: 30 -> 500
        else:
            return 250   # 증가: 20 -> 250
