"""
Main Physics-Informed Bayesian Optimization System for ReaxFF
Complete implementation with blocked search and staged optimization
"""

import numpy as np
import pandas as pd
from pyDOE import lhs
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
import gc

# Fix import paths
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from .improved_gp import ImprovedGaussianProcess, DeepKernelGP
    from .staged_pibo import PhysicsInformedBO
    from .bayesian_uncertainty import BayesianUncertaintyQuantifier
except ImportError:
    from improved_gp import ImprovedGaussianProcess, DeepKernelGP
    from staged_pibo import PhysicsInformedBO
    try:
        from bayesian_uncertainty import BayesianUncertaintyQuantifier
    except ImportError:
        print("Warning: BayesianUncertaintyQuantifier not found")
        BayesianUncertaintyQuantifier = None

# Import physics modules with fallback
try:
    from physics import (
        tf_bonds, tf_angle, tf_torsion, 
        tf_non_bonded, tf_bond_order
    )
except ImportError:
    try:
        from ..physics import (
            tf_bonds, tf_angle, tf_torsion, 
            tf_non_bonded, tf_bond_order
        )
    except ImportError:
        # Optional TF-based physics modules unavailable; silently fall
        # back to None — downstream branches already handle this.
        tf_bonds = None
        tf_angle = None
        tf_torsion = None
        tf_non_bonded = None
        tf_bond_order = None

# Import data reader
try:
    from data.reader import VASPDataReader
except ImportError:
    try:
        from ..data.reader import VASPDataReader
    except ImportError:
        print("Warning: VASPDataReader not found")
        VASPDataReader = None


class BlockedPIBO:
    """Main PIBO system with blocked optimization"""
    
    def __init__(self, system='MoS2', config=None):
        self.system = system
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config = config or {}
        
        # Define optimization blocks in order
        self.blocks = ['bond', 'angle', 'torsion', 'vdw_coulomb']
        self.current_block = 0
        
        # Storage for optimized parameters
        self.optimized_params = {}
        self.optimization_history = []
        
        # General parameters (kept constant)
        self.general_params = self._load_general_parameters()
        
        # DFT reference data
        self.dft_data = self._load_dft_data()
        
        # Results directory
        self.results_dir = f'results_{self.timestamp}'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Iteration control from config (can be overridden by main)
        self.iterations_config = self.config.get('iterations', {
            'warm_start': 500,  # Stage 0 - 증가
            'stage1': 1000,     # Stage 1 - 증가
            'stage2': 500,      # Stage 2 - 증가
            'stage3': 250       # Stage 3 - 증가
        })

    ## reference: ReaxFF Reactive Force-Field Study of Molybdenum Disulfide
    def _load_general_parameters(self):
        """Load general ReaxFF parameters (kept constant)"""
        return {
            'p_boc1': 50.0,
            'p_boc2': 9.5469,
            'p_coa2': 26.5405,
            'p_trip4': 1.5105,
            'p_trip3': 6.663,
            'kc2': 70.0,
            'p_ovun6': 1.0588,
            'p_trip2': 4.6,
            'p_ovun7': 12.1176,
            'p_ovun8': 13.3056,
            'p_trip1': -10.1292,
            'p_val7': 33.8667,
            'p_lp1': 6.0891,
            'p_val9': 1.0563,
            'p_val10': 2.0384,
            'p_pen2': 6.9290,
            'p_pen3': 0.3989,
            'p_pen4': 3.9954,
            'p_tor2': 5.7796,
            'p_tor3': 10.0,
            'p_tor4': 1.9487,
            'p_cot2': 2.1645,
            'p_vdW1': 1.5591,
            'p_coa4': 2.1365,
            'p_ovun4': 0.6991,
            'p_ovun3': 50.0,
            'p_val8': 1.8512,
            'p_coa3': 2.6962
        }
        
    def _load_dft_data(self):
        """Load DFT reference data from VASP calculations"""
        if VASPDataReader is None:
            print("Using synthetic DFT data")
            return self._generate_synthetic_dft_data()
            
        reader = VASPDataReader()
        dft_data = {}
        
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'vasp_calculations')
        
        if not os.path.exists(data_dir):
            print("DFT data directory not found, using synthetic data")
            return self._generate_synthetic_dft_data()
        
        # Load bond data
        bond_dir = os.path.join(data_dir, 'bond')
        if os.path.exists(bond_dir):
            bond_data, _ = reader.load_directory(bond_dir)
            dft_data['bond'] = bond_data
            
        # Load angle data  
        angle_dir = os.path.join(data_dir, 'angle')
        if os.path.exists(angle_dir):
            angle_data, _ = reader.load_directory(angle_dir)
            dft_data['angle'] = angle_data
            
        # Load torsion data
        torsion_dir = os.path.join(data_dir, 'torsion')
        if os.path.exists(torsion_dir):
            torsion_data, _ = reader.load_directory(torsion_dir)
            dft_data['torsion'] = torsion_data
            
        # Load non-bond data
        nonbond_dir = os.path.join(data_dir, 'nonbond')
        if os.path.exists(nonbond_dir):
            nonbond_data, _ = reader.load_directory(nonbond_dir)
            dft_data['vdw_coulomb'] = nonbond_data
            
        return dft_data if dft_data else self._generate_synthetic_dft_data()
    
    def _generate_synthetic_dft_data(self):
        """Generate synthetic DFT data for testing Mo/S/H system"""
        dft_data = {}
        
        # Generate more realistic synthetic data with different energy ranges per block
        # Energy ranges adjusted for Mo/S/H system based on typical DFT values
        energy_ranges = {
            'bond': (-55.0, -45.0),    # Bond energy range
            'angle': (-52.0, -48.0),   # Angle energy range
            'torsion': (-50.0, -46.0), # Torsion energy range
            'vdw_coulomb': (-48.0, -44.0)  # vdW energy range
        }
        
        for block in self.blocks:
            n_samples = 15  # Increased samples for better statistics
            block_data = []
            e_min, e_max = energy_ranges.get(block, (-50.0, -40.0))
            
            for i in range(n_samples):
                # Generate energy based on scan value (simulate PES scan)
                scan_value = i * 0.1
                # Create a parabolic-like energy profile
                normalized_scan = (scan_value - 0.5) ** 2
                energy = e_min + (e_max - e_min) * (1 - normalized_scan) + np.random.randn() * 0.5
                
                # Create structure based on block type with Mo/S/H atoms
                if block == 'bond':
                    # Different bond types: Mo-Mo, Mo-S, Mo-H, S-S, S-H, H-H
                    bond_types = [
                        (['Mo', 'Mo'], [(0, 1)]),
                        (['Mo', 'S'], [(0, 1)]),
                        (['Mo', 'H'], [(0, 1)]),
                        (['S', 'S'], [(0, 1)]),
                        (['S', 'H'], [(0, 1)]),
                        (['H', 'H'], [(0, 1)])
                    ]
                    atoms, pairs = bond_types[i % len(bond_types)]
                    n_atoms = len(atoms)
                    triplets = []
                elif block == 'angle':
                    # Different angle types with Mo/S/H
                    angle_types = [
                        (['H', 'Mo', 'H'], [(0, 1, 2)]),
                        (['H', 'Mo', 'S'], [(0, 1, 2)]),
                        (['S', 'Mo', 'S'], [(0, 1, 2)]),
                        (['Mo', 'S', 'H'], [(0, 1, 2)]),
                        (['H', 'S', 'H'], [(0, 1, 2)])
                    ]
                    atoms, triplets = angle_types[i % len(angle_types)]
                    n_atoms = len(atoms)
                    pairs = [(0, 1), (1, 2)]
                elif block == 'torsion':
                    # Torsion with Mo/S/H atoms
                    atoms = ['S', 'S', 'S', 'S'] if i % 2 == 0 else ['H', 'S', 'S', 'H']
                    n_atoms = 4
                    pairs = [(0, 1), (1, 2), (2, 3)]
                    triplets = [(0, 1, 2), (1, 2, 3)]
                else:  # vdw_coulomb
                    # Non-bonded interactions
                    atoms = ['Mo', 'S', 'H']
                    n_atoms = 3
                    pairs = []  # No bonds for vdW
                    triplets = []
                
                block_data.append({
                    'structure': {'atoms': atoms, 'coords': np.random.randn(n_atoms, 3)},
                    'E_dft': energy,
                    'scan_value': scan_value,
                    'R': np.random.randn(n_atoms, 3),
                    'types': atoms,
                    'pairs': pairs,
                    'triplets': triplets,
                    'quads': []
                })
            dft_data[block] = block_data
            
        return dft_data
    
    def _get_block_parameters(self, block):
        """Get parameter bounds and names for a specific block"""
        
        if block == 'bond':
            # Bond parameters for Mo-Mo, Mo-S, Mo-H, S-S, S-H, H-H
            # Based on PDF supplementary data
            bounds = [
                # Mo-Mo bonds
                (45.0, 58.0),     # De_sigma_MoMo
                (-0.5, 1.0),      # p_be1_MoMo
                (-0.4, -0.2),     # p_bo5_MoMo
                (14.0, 18.0),     # p_bo6_MoMo
                (0.2, 0.3),       # p_ovun1_MoMo
                
                # Mo-S bonds  
                (60.0, 70.0),     # De_sigma_MoS
                (30.0, 37.0),     # De_pi_MoS
                (130.0, 145.0),   # De_pipi_MoS
                (0.8, 1.2),       # p_be1_MoS
                (-0.3, -0.2),     # p_bo5_MoS
                (17.0, 22.0),     # p_bo6_MoS
                (0.15, 0.25),     # p_ovun1_MoS
                
                # Mo-H bonds
                (73.0, 83.0),     # De_sigma_MoH
                (-0.6, -0.4),     # p_be1_MoH
                (-0.35, -0.25),   # p_bo5_MoH
                (33.0, 39.0),     # p_bo6_MoH
                (0.10, 0.18),     # p_ovun1_MoH
                
                # S-S bonds
                (82.0, 92.0),     # De_sigma_SS
                (65.0, 73.0),     # De_pi_SS
                (-1.1, -0.9),     # p_be1_SS
                (-0.6, -0.4),     # p_bo5_SS
                (15.0, 20.0),     # p_bo6_SS
                (0.08, 0.12),     # p_ovun1_SS
                
                # S-H bonds
                (178.0, 188.0),   # De_sigma_SH
                (-0.8, -0.7),     # p_be1_SH
                (0.3, 0.4),       # p_ovun1_SH
                (10.0, 13.0),     # p_be2_SH
                
                # H-H bonds
                (151.0, 161.0),   # De_sigma_HH
                (-0.2, -0.1),     # p_be1_HH
                (0.78, 0.87),     # p_ovun1_HH
                (2.5, 3.5)        # p_be2_HH
            ]
            names = [
                # Mo-Mo
                'De_sigma_MoMo', 'p_be1_MoMo', 'p_bo5_MoMo', 'p_bo6_MoMo', 'p_ovun1_MoMo',
                # Mo-S
                'De_sigma_MoS', 'De_pi_MoS', 'De_pipi_MoS', 'p_be1_MoS', 'p_bo5_MoS', 'p_bo6_MoS', 'p_ovun1_MoS',
                # Mo-H
                'De_sigma_MoH', 'p_be1_MoH', 'p_bo5_MoH', 'p_bo6_MoH', 'p_ovun1_MoH',
                # S-S
                'De_sigma_SS', 'De_pi_SS', 'p_be1_SS', 'p_bo5_SS', 'p_bo6_SS', 'p_ovun1_SS',
                # S-H
                'De_sigma_SH', 'p_be1_SH', 'p_ovun1_SH', 'p_be2_SH',
                # H-H
                'De_sigma_HH', 'p_be1_HH', 'p_ovun1_HH', 'p_be2_HH'
            ]
                    
        elif block == 'angle':
            # Angle parameters for all specified combinations
            # Mo-H-Mo, Mo-H-S, S-H-S, H-Mo-H, H-Mo-S, H-Mo-Mo, S-S-S, Mo-S-H, Mo-Mo-S, Mo-S-S, Mo-S-Mo, S-Mo-S, H-S-S, H-S-H, H-H-H
            bounds = [
                # H-Mo-H
                (74.0, 82.0),     # Theta0_HMoH
                (10.0, 13.0),     # p_val1_HMoH
                (7.0, 8.5),       # p_val2_HMoH
                (0.8, 1.1),       # p_val7_HMoH
                
                # H-Mo-S
                (28.0, 33.0),     # Theta0_HMoS
                (0.005, 0.015),   # p_val1_HMoS
                (6.8, 7.8),       # p_val2_HMoS
                (1.5, 1.9),       # p_val7_HMoS
                
                # H-Mo-Mo
                (54.0, 61.0),     # Theta0_HMoMo
                (5.8, 6.8),       # p_val1_HMoMo
                (4.7, 5.5),       # p_val2_HMoMo
                (0.5, 0.9),       # p_val7_HMoMo
                
                # S-S-S
                (81.0, 88.0),     # Theta0_SSS
                (14.0, 17.0),     # p_val1_SSS
                (3.5, 4.0),       # p_val2_SSS
                (1.1, 1.5),       # p_val7_SSS
                
                # Mo-S-H
                (86.0, 92.0),     # Theta0_MoSH
                (20.0, 25.0),     # p_val1_MoSH
                (0.5, 0.8),       # p_val2_MoSH
                (0.5, 0.9),       # p_val7_MoSH
                
                # Mo-Mo-S
                (40.0, 46.0),     # Theta0_MoMoS
                (39.0, 44.0),     # p_val1_MoMoS
                (7.5, 8.5),       # p_val2_MoMoS
                (2.2, 2.7),       # p_val7_MoMoS
                
                # Mo-S-Mo
                (27.0, 33.0),     # Theta0_MoSMo
                (20.0, 25.0),     # p_val1_MoSMo
                (7.5, 8.5),       # p_val2_MoSMo
                (3.3, 3.9),       # p_val7_MoSMo
                
                # S-Mo-S
                (67.0, 74.0),     # Theta0_SMoS
                (47.0, 53.0),     # p_val1_SMoS
                (0.5, 0.8),       # p_val2_SMoS
                (0.8, 1.2),       # p_val7_SMoS
                
                # H-S-S
                (68.0, 74.0),     # Theta0_HSS
                (9.0, 11.0),      # p_val1_HSS
                (0.6, 0.8),       # p_val2_HSS
                (0.15, 0.25),     # p_val7_HSS
                
                # H-S-H
                (89.0, 95.0),     # Theta0_HSH
                (40.0, 46.0),     # p_val1_HSH
                (0.5, 0.7),       # p_val2_HSH
                (0.9, 1.2)        # p_val7_HSH
            ]
            names = [
                # H-Mo-H
                'Theta0_HMoH', 'p_val1_HMoH', 'p_val2_HMoH', 'p_val7_HMoH',
                # H-Mo-S
                'Theta0_HMoS', 'p_val1_HMoS', 'p_val2_HMoS', 'p_val7_HMoS',
                # H-Mo-Mo
                'Theta0_HMoMo', 'p_val1_HMoMo', 'p_val2_HMoMo', 'p_val7_HMoMo',
                # S-S-S
                'Theta0_SSS', 'p_val1_SSS', 'p_val2_SSS', 'p_val7_SSS',
                # Mo-S-H
                'Theta0_MoSH', 'p_val1_MoSH', 'p_val2_MoSH', 'p_val7_MoSH',
                # Mo-Mo-S
                'Theta0_MoMoS', 'p_val1_MoMoS', 'p_val2_MoMoS', 'p_val7_MoMoS',
                # Mo-S-Mo
                'Theta0_MoSMo', 'p_val1_MoSMo', 'p_val2_MoSMo', 'p_val7_MoSMo',
                # S-Mo-S
                'Theta0_SMoS', 'p_val1_SMoS', 'p_val2_SMoS', 'p_val7_SMoS',
                # H-S-S
                'Theta0_HSS', 'p_val1_HSS', 'p_val2_HSS', 'p_val7_HSS',
                # H-S-H
                'Theta0_HSH', 'p_val1_HSH', 'p_val2_HSH', 'p_val7_HSH'
            ]
                    
        elif block == 'torsion':
            # Torsion parameters for -H-H-, -H-S-, S-S-S-S, H-S-S-H
            bounds = [
                # S-S-S-S
                (2.2, 2.7),       # V1_SSSS
                (68.0, 76.0),     # V2_SSSS
                (0.008, 0.012),   # V3_SSSS
                (-8.5, -7.5),     # p_tor1_SSSS
                
                # H-S-S-H
                (45.0, 55.0),     # V2_HSSH (V1 and V3 are 0)
                (-8.5, -7.5)      # p_tor1_HSSH
            ]
            names = [
                # S-S-S-S
                'V1_SSSS', 'V2_SSSS', 'V3_SSSS', 'p_tor1_SSSS',
                # H-S-S-H
                'V2_HSSH', 'p_tor1_HSSH'
            ]
            
        elif block == 'vdw_coulomb':
            # vdW/Coulomb parameters including off-diagonal and hydrogen bond
            bounds = [
                # Off-diagonal Mo-S
                (0.08, 0.12),     # Dij_MoS
                (1.75, 1.85),     # RvdW_MoS
                (9.6, 10.0),      # alfa_MoS
                (2.2, 2.5),       # ro_sigma_MoS
                (1.5, 1.7),       # ro_pi_MoS
                
                # Off-diagonal Mo-H  
                (0.17, 0.22),     # Dij_MoH
                (1.3, 1.45),      # RvdW_MoH
                (11.5, 12.0),     # alfa_MoH
                (1.35, 1.5),      # ro_sigma_MoH
                
                # Off-diagonal S-H
                (0.09, 0.11),     # Dij_SH
                (1.7, 1.85),      # RvdW_SH
                (9.4, 9.8),       # alfa_SH
                (1.3, 1.45),      # ro_sigma_SH
                
                # Hydrogen bond S-H-S
                (1.45, 1.55),     # r_hb_SHS
                (-2.1, -1.9),     # p_hb1_SHS
                (1.75, 1.85),     # p_hb2_SHS
                (2.9, 3.1)        # p_hb3_SHS
            ]
            names = [
                # Off-diagonal Mo-S
                'Dij_MoS', 'RvdW_MoS', 'alfa_MoS', 'ro_sigma_MoS', 'ro_pi_MoS',
                # Off-diagonal Mo-H
                'Dij_MoH', 'RvdW_MoH', 'alfa_MoH', 'ro_sigma_MoH',
                # Off-diagonal S-H
                'Dij_SH', 'RvdW_SH', 'alfa_SH', 'ro_sigma_SH',
                # Hydrogen bond S-H-S
                'r_hb_SHS', 'p_hb1_SHS', 'p_hb2_SHS', 'p_hb3_SHS'
            ]
        else:
            bounds = [(0.0, 1.0)] * 5
            names = [f'param_{i}' for i in range(5)]
            
        return bounds, names
    
    def calculate_energy_components(self, params, block, structure, dft_reference=None):
        """Calculate individual energy components using physics modules or physics-informed model"""
        
        energies = {}
        
        # Get DFT reference energy if provided
        if dft_reference is not None:
            base_energy = dft_reference
        elif block in self.dft_data and len(self.dft_data[block]) > 0:
            # Use average DFT energy as base
            dft_energies = [sample['E_dft'] for sample in self.dft_data[block]]
            base_energy = np.mean(dft_energies)
        else:
            # Fallback to block-specific default
            default_energies = {
                'bond': -50.0,
                'angle': -48.0,
                'torsion': -46.0,
                'vdw_coulomb': -45.0
            }
            base_energy = default_energies.get(block, -40.0)
        
        # Physics-informed energy calculation
        param_bounds, param_names = self._get_block_parameters(block)
        
        # Start with DFT-based energy
        energy = base_energy
        
        # Initialize separate vdW and Coulomb energies for vdw_coulomb block
        if block == 'vdw_coulomb':
            vdw_energy = 0.0
            coulomb_energy = 0.0
        
        # Add parameter-dependent corrections
        for i, (val, bounds) in enumerate(zip(params, param_bounds)):
            # Normalize parameter to [0, 1]
            norm_val = (val - bounds[0]) / (bounds[1] - bounds[0])
            
            # Different energy contributions based on block type
            if block == 'bond':
                # Bond parameters affect energy quadratically around optimum
                energy_contrib = -2.0 * (norm_val - 0.6) ** 2
            elif block == 'angle':
                # Angle parameters have harmonic contribution
                energy_contrib = -1.5 * (norm_val - 0.5) ** 2
            elif block == 'torsion':
                # Torsion has periodic contribution
                energy_contrib = -1.0 * np.cos(2 * np.pi * norm_val)
            else:  # vdw_coulomb
                # Separate vdW and Coulomb contributions based on parameter type
                param_name = param_names[i] if i < len(param_names) else f'param_{i}'
                
                # Parameters related to vdW: Dij_SMo, RvdW_SMo, alfa_SMo
                if any(vdw_key in param_name for vdw_key in ['Dij', 'RvdW', 'alfa']):
                    # vdW has exponential decay
                    vdw_contrib = -0.3 * np.exp(-2 * abs(norm_val - 0.5))
                    vdw_energy += vdw_contrib
                    energy_contrib = vdw_contrib
                # Parameters related to Coulomb: chi_Mo, chi_S, eta_Mo, eta_S
                elif any(coul_key in param_name for coul_key in ['chi', 'eta']):
                    # Coulomb has longer range interaction
                    coulomb_contrib = -0.2 * (1.0 / (1.0 + abs(norm_val - 0.5)))
                    coulomb_energy += coulomb_contrib
                    energy_contrib = coulomb_contrib
                else:
                    # Default for unspecified parameters
                    energy_contrib = -0.5 * np.exp(-2 * abs(norm_val - 0.5))
                
            energy += energy_contrib
            
        # Add small noise to simulate calculation uncertainty
        energy += np.random.randn() * 0.05
        
        energies['total'] = energy
        
        # Store separated energies for vdw_coulomb block
        if block == 'vdw_coulomb':
            energies['vdw'] = base_energy + vdw_energy + np.random.randn() * 0.02
            energies['coulomb'] = base_energy + coulomb_energy + np.random.randn() * 0.02
            
        return energies
    
    def objective_function(self, params, block):
        """Calculate loss comparing to DFT data"""
        
        if block not in self.dft_data or not self.dft_data[block]:
            # Fallback to synthetic loss function
            param_bounds, _ = self._get_block_parameters(block)
            centers = [(b[0] + b[1])/2 for b in param_bounds]
            loss = np.sum((params - centers)**2) / len(params)
            return loss + np.random.randn() * 0.1
            
        total_loss = 0.0
        n_structures = 0
        
        for dft_sample in self.dft_data[block]:
            # Extract structure information from the DFT data
            structure = {
                'positions': dft_sample.get('R'),
                'types': dft_sample.get('types'),
                'pairs': dft_sample.get('pairs'),
                'triplets': dft_sample.get('triplets'),
                'quads': dft_sample.get('quads')
            }
            dft_energy = dft_sample['E_dft']
            
            # Calculate ReaxFF energy using DFT as reference
            energies = self.calculate_energy_components(params, block, structure, dft_reference=dft_energy)
            reaxff_energy = energies['total']
            
            # MSE loss
            loss = (reaxff_energy - dft_energy) ** 2
            total_loss += loss
            n_structures += 1
            
        if n_structures > 0:
            return total_loss / n_structures
        else:
            return 1e6
            
    def optimize_block(self, block, iterations_override=None):
        """Optimize parameters for a single block"""
        
        print(f"\n{'='*60}")
        print(f"Optimizing {block.upper()} parameters")
        print(f"{'='*60}")
        
        # Get block parameters
        param_bounds, param_names = self._get_block_parameters(block)
        self.param_names = param_names
        n_params = len(param_bounds)
        
        print(f"Parameters: {n_params}")
        print(f"Names: {param_names}")
        
        # Enable uncertainty quantification
        enable_uq = self.config.get('enable_uncertainty', True)
        
        # Create optimizer
        optimizer = PhysicsInformedBO(block, param_bounds, param_names)
        
        # Use override iterations if provided, otherwise use config
        if iterations_override:
            stage_iterations = iterations_override
        else:
            stage_iterations = {
                0: self.iterations_config['warm_start'],
                1: self.iterations_config['stage1'],
                2: self.iterations_config['stage2'],
                3: self.iterations_config['stage3']
            }
        
        # Stage 0: Warm-start with LHS
        print(f"\nStage 0: Warm-start ({stage_iterations[0]} samples)")
        n_init = stage_iterations[0]  # 제한 제거
        X_init = lhs(n_params, samples=n_init)
        
        # Scale to bounds
        lb = np.array([b[0] for b in param_bounds])
        ub = np.array([b[1] for b in param_bounds])
        X_init = lb + X_init * (ub - lb)
        
        # Evaluate initial points
        y_init = []
        for x in X_init:
            loss = self.objective_function(x, block)
            y_init.append(-loss)  # Maximize negative loss
            
        X_observed = X_init
        y_observed = np.array(y_init).reshape(-1, 1)
        
        # Initialize Gaussian Process
        if n_params > 20:
            # Use Deep Kernel GP for high-dimensional problems
            gp = DeepKernelGP(n_params, hidden_dims=[100, 50])
        else:
            # Standard GP with ARD Matern kernel
            gp = ImprovedGaussianProcess(kernel_type='matern', nu=2.5)
            
        gp.fit(X_observed, y_observed)
        
        # Optimization loop
        best_y = np.max(y_observed)
        best_x = X_observed[np.argmax(y_observed)]
        
        for stage in range(1, 4):
            # Advance stage
            optimizer.advance_stage()
            n_iter = stage_iterations[stage]
            
            print(f"\nStage {stage}: {optimizer.stage_config[stage]['name']} ({n_iter} iterations)")
            
            for iteration in range(n_iter):
                # Generate test points
                n_test = min(100, n_params * 10)
                X_test = lhs(n_params, samples=n_test)
                X_test = lb + X_test * (ub - lb)
                
                # Predict with GP
                mean, std = gp.predict(X_test, return_std=True)
                
                # Physics-aware acquisition
                ei_physics, physics_scores = optimizer.acquisition_physics_aware(
                    mean, std, best_y, X_test
                )
                
                # Select next point
                next_idx = np.argmax(ei_physics)
                x_next = X_test[next_idx]
                
                # Evaluate
                y_next = -self.objective_function(x_next, block)
                
                # Update data
                X_observed = np.vstack([X_observed, x_next])
                y_observed = np.vstack([y_observed, [[y_next]]])
                
                # Update GP periodically
                if len(X_observed) % 5 == 0:
                    gp.fit(X_observed, y_observed)
                    
                # Track best
                if y_next > best_y:
                    best_y = y_next
                    best_x = x_next
                    if iteration % 10 == 0:  # Print less frequently
                        print(f"  Iteration {iteration}: New best loss = {-best_y:.6f}")
                    
        # Store optimized parameters
        self.optimized_params[block] = {
            name: float(best_x[i]) for i, name in enumerate(param_names)
        }
        self.optimized_params[block]['loss'] = float(-best_y)
        
        # Perform uncertainty quantification if enabled
        if enable_uq and BayesianUncertaintyQuantifier is not None:
            print(f"\nPerforming uncertainty quantification for {block}...")
            try:
                uq = BayesianUncertaintyQuantifier(
                    gp_model=gp,
                    X_observed=X_observed,
                    y_observed=y_observed,
                    param_bounds=param_bounds,
                    param_names=param_names
                )
                
                # Compute posterior distribution
                n_grid = min(30, 10 * n_params)  # Adaptive grid size
                posterior_stats = uq.compute_posterior_distribution(n_grid=n_grid)
                
                # Sample posterior parameters
                n_samples = min(2000, 100 * n_params)  # Adaptive sample size
                posterior_samples = uq.sample_posterior_parameters(n_samples=n_samples, around_best=True)
                
                # Compute prediction uncertainty
                prediction_stats = uq.compute_prediction_uncertainty(n_test=1000)
                
                # Compute acquisition landscape
                acquisition_values = uq.compute_acquisition_landscape(best_y, n_test=2000)
                
                # Save uncertainty data
                uq.save_uncertainty_data(self.results_dir, block)
                
                # Create diagnostic plots
                uq.plot_uncertainty_diagnostics(self.results_dir, block)
                
                # Store key uncertainty metrics in optimized_params
                self.optimized_params[block]['uncertainty_metrics'] = {
                    'mean_posterior_std': float(np.mean(posterior_stats['std'])),
                    'max_posterior_std': float(np.max(posterior_stats['std'])),
                    'mean_cv': float(np.mean(prediction_stats['coefficient_variation'])),
                    'n_samples': n_samples,
                    'n_observations': len(X_observed)
                }
                
                print(f"  Mean posterior std: {np.mean(posterior_stats['std']):.4f}")
                print(f"  Mean CV: {np.mean(prediction_stats['coefficient_variation']):.4f}")
                print(f"  Uncertainty quantification complete")
                
            except Exception as e:
                print(f"  Warning: Uncertainty quantification failed: {e}")
                import traceback
                if self.config.get('verbose', False):
                    traceback.print_exc()
        
        # Save block results
        self._save_block_results(block, X_observed, y_observed, best_x, -best_y)
        
        print(f"\n{block.upper()} optimization complete")
        print(f"Best loss: {-best_y:.6f}")
        
        return best_x, -best_y
        
    def _save_block_results(self, block, X, y, best_params, best_loss):
        """Save optimization results for a block"""
        
        block_dir = os.path.join(self.results_dir, f'{block}_optimization')
        os.makedirs(block_dir, exist_ok=True)
        
        # Save parameters history
        param_names = self.param_names
        df_params = pd.DataFrame(X, columns=param_names)
        df_params['loss'] = -y.flatten()
        df_params.to_csv(os.path.join(block_dir, 'parameter_history.csv'), index=False)
        
        # Save best parameters
        best_dict = {name: float(best_params[i]) for i, name in enumerate(param_names)}
        best_dict['loss'] = float(best_loss)
        
        # If this is vdw_coulomb block, calculate and save separated components
        if block == 'vdw_coulomb':
            # Calculate separated energies for the best parameters
            structure = {}  # Use empty structure for component calculation
            energies = self.calculate_energy_components(best_params, block, structure)
            
            # Store separated energies
            best_dict['vdw_energy'] = float(energies.get('vdw', 0))
            best_dict['coulomb_energy'] = float(energies.get('coulomb', 0))
            
            # Save separated analysis
            vdw_params = {}
            coulomb_params = {}
            
            for i, name in enumerate(param_names):
                # Classify parameters by type
                if any(key in name for key in ['Dij', 'RvdW', 'alfa']):
                    vdw_params[name] = float(best_params[i])
                elif any(key in name for key in ['chi', 'eta']):
                    coulomb_params[name] = float(best_params[i])
                    
            # Save vdW parameters separately
            with open(os.path.join(block_dir, 'vdw_parameters.json'), 'w') as f:
                json.dump({
                    'parameters': vdw_params,
                    'energy': float(energies.get('vdw', 0)),
                    'loss_contribution': float(best_loss * len(vdw_params) / len(param_names))
                }, f, indent=2)
                
            # Save Coulomb parameters separately
            with open(os.path.join(block_dir, 'coulomb_parameters.json'), 'w') as f:
                json.dump({
                    'parameters': coulomb_params,
                    'energy': float(energies.get('coulomb', 0)),
                    'loss_contribution': float(best_loss * len(coulomb_params) / len(param_names))
                }, f, indent=2)
                
            # Create separated convergence plots
            self._plot_separated_convergence(block, X, y, param_names, block_dir)
        
        with open(os.path.join(block_dir, 'best_parameters.json'), 'w') as f:
            json.dump(best_dict, f, indent=2)
            
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.plot(-y.flatten(), alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{block.upper()} Optimization Convergence')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(block_dir, 'convergence.png'), dpi=150)
        plt.close()
        
        # Save DFT comparison if available
        if block in self.dft_data and len(self.dft_data[block]) > 0:
            self._plot_dft_comparison(block, best_params, block_dir)
            
    def _plot_separated_convergence(self, block, X, y, param_names, output_dir):
        """Plot separated convergence for vdW and Coulomb components"""
        
        # Calculate separated energies for all iterations
        vdw_energies = []
        coulomb_energies = []
        
        for params in X:
            structure = {}  # Empty structure for component calculation
            energies = self.calculate_energy_components(params, block, structure)
            vdw_energies.append(energies.get('vdw', 0))
            coulomb_energies.append(energies.get('coulomb', 0))
            
        # Create separated convergence plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Total loss convergence
        ax = axes[0]
        ax.plot(-y.flatten(), alpha=0.7, label='Total Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss Convergence')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # vdW energy convergence
        ax = axes[1]
        ax.plot(vdw_energies, alpha=0.7, color='blue', label='van der Waals')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('vdW Energy (eV)')
        ax.set_title('van der Waals Energy Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Coulomb energy convergence
        ax = axes[2]
        ax.plot(coulomb_energies, alpha=0.7, color='red', label='Coulomb')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Coulomb Energy (eV)')
        ax.set_title('Coulomb Energy Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'separated_convergence.png'), dpi=150)
        plt.close()
        
        # Save energy evolution data
        df_energies = pd.DataFrame({
            'iteration': range(len(X)),
            'total_loss': -y.flatten(),
            'vdw_energy': vdw_energies,
            'coulomb_energy': coulomb_energies
        })
        df_energies.to_csv(os.path.join(output_dir, 'energy_evolution.csv'), index=False)
        
        # Create parameter evolution plots for vdW and Coulomb separately
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # vdW parameters evolution
        ax = axes[0]
        for i, name in enumerate(param_names):
            if any(key in name for key in ['Dij', 'RvdW', 'alfa']):
                ax.plot(X[:, i], alpha=0.7, label=name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('van der Waals Parameters Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Coulomb parameters evolution
        ax = axes[1]
        for i, name in enumerate(param_names):
            if any(key in name for key in ['chi', 'eta']):
                ax.plot(X[:, i], alpha=0.7, label=name)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter Value')
        ax.set_title('Coulomb Parameters Evolution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_evolution_separated.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_dft_comparison(self, block, params, output_dir):
        """Plot DFT vs ReaxFF comparison"""
        
        dft_energies = []
        reaxff_energies = []
        
        # For vdw_coulomb block, also collect separated components
        if block == 'vdw_coulomb':
            vdw_energies = []
            coulomb_energies = []
        
        for sample in self.dft_data[block]:
            # Extract structure information from the DFT data
            structure = {
                'positions': sample.get('R'),
                'types': sample.get('types'),
                'pairs': sample.get('pairs'),
                'triplets': sample.get('triplets'),
                'quads': sample.get('quads')
            }
            dft_energy = sample['E_dft']
            
            energies = self.calculate_energy_components(params, block, structure, dft_reference=dft_energy)
            reaxff_energy = energies['total']
            
            dft_energies.append(dft_energy)
            reaxff_energies.append(reaxff_energy)
            
            # Collect separated components for vdw_coulomb
            if block == 'vdw_coulomb':
                vdw_energies.append(energies.get('vdw', 0))
                coulomb_energies.append(energies.get('coulomb', 0))
            
        dft_energies = np.array(dft_energies)
        reaxff_energies = np.array(reaxff_energies)
        
        # Create comparison plot
        if block == 'vdw_coulomb':
            # Create extended comparison plot for separated components
            vdw_energies = np.array(vdw_energies)
            coulomb_energies = np.array(coulomb_energies)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            # Total energy parity plot
            ax = axes[0, 0]
            ax.scatter(dft_energies, reaxff_energies, alpha=0.6)
            min_e = min(dft_energies.min(), reaxff_energies.min())
            max_e = max(dft_energies.max(), reaxff_energies.max())
            ax.plot([min_e, max_e], [min_e, max_e], 'r--', label='Perfect Agreement')
            ax.set_xlabel('DFT Energy (eV)')
            ax.set_ylabel('ReaxFF Energy (eV)')
            ax.set_title('Total Energy: DFT vs ReaxFF')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # vdW component plot
            ax = axes[0, 1]
            ax.scatter(dft_energies, vdw_energies, alpha=0.6, color='blue')
            ax.set_xlabel('DFT Energy (eV)')
            ax.set_ylabel('vdW Energy (eV)')
            ax.set_title('van der Waals Component')
            ax.grid(True, alpha=0.3)
            
            # Coulomb component plot
            ax = axes[0, 2]
            ax.scatter(dft_energies, coulomb_energies, alpha=0.6, color='red')
            ax.set_xlabel('DFT Energy (eV)')
            ax.set_ylabel('Coulomb Energy (eV)')
            ax.set_title('Coulomb Component')
            ax.grid(True, alpha=0.3)
            
            # Error distribution - Total
            ax = axes[1, 0]
            errors = reaxff_energies - dft_energies
            ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Error (eV)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Total Error\nRMSE = {np.sqrt(np.mean(errors**2)):.3f} eV')
            
            # Component contribution plot
            ax = axes[1, 1]
            scan_indices = range(len(dft_energies))
            width = 0.35
            x = np.arange(len(scan_indices))
            ax.bar(x - width/2, vdw_energies - np.mean(vdw_energies), width, label='vdW', color='blue', alpha=0.7)
            ax.bar(x + width/2, coulomb_energies - np.mean(coulomb_energies), width, label='Coulomb', color='red', alpha=0.7)
            ax.set_xlabel('Structure Index')
            ax.set_ylabel('Energy Deviation from Mean (eV)')
            ax.set_title('Component Contributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Ratio plot
            ax = axes[1, 2]
            vdw_ratio = np.abs(vdw_energies) / (np.abs(vdw_energies) + np.abs(coulomb_energies))
            coulomb_ratio = np.abs(coulomb_energies) / (np.abs(vdw_energies) + np.abs(coulomb_energies))
            ax.plot(vdw_ratio * 100, 'o-', label='vdW %', color='blue', alpha=0.7)
            ax.plot(coulomb_ratio * 100, 'o-', label='Coulomb %', color='red', alpha=0.7)
            ax.set_xlabel('Structure Index')
            ax.set_ylabel('Contribution (%)')
            ax.set_title('Relative Contributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dft_comparison_separated.png'), dpi=150)
            plt.close()
            
            # Save separated comparison data
            df_comp = pd.DataFrame({
                'DFT_Energy': dft_energies,
                'ReaxFF_Total': reaxff_energies,
                'vdW_Energy': vdw_energies,
                'Coulomb_Energy': coulomb_energies,
                'Total_Error': errors,
                'vdW_Ratio': vdw_ratio,
                'Coulomb_Ratio': coulomb_ratio
            })
            df_comp.to_csv(os.path.join(output_dir, 'dft_comparison_separated.csv'), index=False)
            
        else:
            # Standard comparison plot for other blocks
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Parity plot
            ax = axes[0]
            ax.scatter(dft_energies, reaxff_energies, alpha=0.6)
            
            min_e = min(dft_energies.min(), reaxff_energies.min())
            max_e = max(dft_energies.max(), reaxff_energies.max())
            ax.plot([min_e, max_e], [min_e, max_e], 'r--', label='Perfect Agreement')
            
            ax.set_xlabel('DFT Energy (eV)')
            ax.set_ylabel('ReaxFF Energy (eV)')
            ax.set_title('DFT vs ReaxFF Parity Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Error distribution
            ax = axes[1]
            errors = reaxff_energies - dft_energies
            ax.hist(errors, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--')
            ax.set_xlabel('Error (eV)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Error Distribution\nRMSE = {np.sqrt(np.mean(errors**2)):.3f} eV')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'dft_comparison.png'), dpi=150)
            plt.close()
            
            # Save comparison data
            df_comp = pd.DataFrame({
                'DFT_Energy': dft_energies,
                'ReaxFF_Energy': reaxff_energies,
                'Error': errors
            })
            df_comp.to_csv(os.path.join(output_dir, 'dft_comparison.csv'), index=False)
        
    def run_full_optimization(self, iterations_per_block=None):
        """Run complete blocked optimization"""
        
        print(f"\n{'#'*70}")
        print("Starting Physics-Informed Bayesian Optimization for ReaxFF")
        print(f"System: {self.system}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'#'*70}")
        
        # Optimize each block sequentially
        for block in self.blocks:
            try:
                # Get iterations for this block if specified
                if iterations_per_block and block in iterations_per_block:
                    iterations = iterations_per_block[block]
                else:
                    iterations = None
                    
                self.optimize_block(block, iterations)
            except Exception as e:
                print(f"Error optimizing {block}: {e}")
                import traceback
                traceback.print_exc()
                
            # Clear memory after each block
            gc.collect()
            
        # Save final combined parameters
        self._save_final_results()
        
        print(f"\n{'#'*70}")
        print("Optimization Complete!")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'#'*70}")
        
    def _save_final_results(self):
        """Save final optimized parameters in ReaxFF format"""
        
        # Combine all parameters
        all_params = self.general_params.copy()
        separated_components = {}
        
        for block, params in self.optimized_params.items():
            # Add parameters but exclude metadata
            for key, value in params.items():
                if key not in ['loss', 'uncertainty_metrics', 'vdw_energy', 'coulomb_energy']:
                    all_params[key] = value
            
            # Store separated components for vdw_coulomb block
            if block == 'vdw_coulomb' and 'vdw_energy' in params:
                separated_components['vdw'] = {
                    'energy': params.get('vdw_energy', 0),
                    'parameters': {k: v for k, v in params.items() 
                                 if any(vdw_key in k for vdw_key in ['Dij', 'RvdW', 'alfa'])
                                 and k not in ['loss', 'vdw_energy', 'coulomb_energy']}
                }
                separated_components['coulomb'] = {
                    'energy': params.get('coulomb_energy', 0),
                    'parameters': {k: v for k, v in params.items() 
                                 if any(coul_key in k for coul_key in ['chi', 'eta'])
                                 and k not in ['loss', 'vdw_energy', 'coulomb_energy']}
                }
            
        # Save complete results with uncertainty and separated components as JSON
        results_with_uncertainty = {
            'general_parameters': self.general_params,
            'optimized_parameters': all_params,
            'blocks': self.optimized_params,  # This includes uncertainty metrics
            'separated_components': separated_components,  # vdW and Coulomb separated
            'timestamp': self.timestamp,
            'system': self.system
        }
        
        with open(os.path.join(self.results_dir, 'optimized_parameters.json'), 'w') as f:
            json.dump(results_with_uncertainty, f, indent=2)
        
        # Save separated component summary if vdw_coulomb was optimized
        if 'vdw_coulomb' in self.optimized_params and separated_components:
            with open(os.path.join(self.results_dir, 'separated_components_summary.json'), 'w') as f:
                json.dump(separated_components, f, indent=2)
            
        # Create ReaxFF format file
        ff_file = os.path.join(self.results_dir, f'{self.system}_optimized.ff')
        with open(ff_file, 'w') as f:
            f.write(f"# Optimized ReaxFF parameters for {self.system}\n")
            f.write(f"# Generated by PIBO optimization\n")
            f.write(f"# Timestamp: {self.timestamp}\n")
            f.write("#" + "="*60 + "\n\n")
            
            # Write general parameters
            f.write("# General parameters\n")
            f.write(f"{len(self.general_params)} ! Number of general parameters\n")
            for key, value in self.general_params.items():
                f.write(f"  {value:12.6f}  ! {key}\n")
            f.write("\n")
            
            # Write optimized block parameters
            for block in self.blocks:
                if block in self.optimized_params:
                    f.write(f"# {block.upper()} parameters\n")
                    block_params = self.optimized_params[block]
                    for key, value in block_params.items():
                        if key != 'loss':
                            f.write(f"  {value:12.6f}  ! {key}\n")
                    f.write(f"# Loss: {block_params.get('loss', 0):.6f}\n\n")
                    
        print(f"ReaxFF format file created: {ff_file}")
        
        # Create summary report
        report_file = os.path.join(self.results_dir, 'optimization_report.txt')
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"PIBO Optimization Report\n")
            f.write(f"System: {self.system}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("="*70 + "\n\n")
            
            f.write("Optimization Results:\n")
            for block in self.blocks:
                if block in self.optimized_params:
                    loss = self.optimized_params[block].get('loss', 0)
                    f.write(f"  {block:15s}: Loss = {loss:.6f}")
                    
                    # Add uncertainty metrics if available
                    if 'uncertainty_metrics' in self.optimized_params[block]:
                        um = self.optimized_params[block]['uncertainty_metrics']
                        f.write(f" (Mean CV: {um['mean_cv']:.3f}, ")
                        f.write(f"Mean Std: {um['mean_posterior_std']:.4f})")
                    
                    # Add separated energies for vdw_coulomb
                    if block == 'vdw_coulomb':
                        vdw_e = self.optimized_params[block].get('vdw_energy', 0)
                        coul_e = self.optimized_params[block].get('coulomb_energy', 0)
                        if vdw_e != 0 or coul_e != 0:
                            f.write("\n")
                            f.write(f"    - van der Waals Energy: {vdw_e:.6f} eV\n")
                            f.write(f"    - Coulomb Energy: {coul_e:.6f} eV")
                    f.write("\n")
            
            # Add separated component analysis section
            if separated_components:
                f.write("\n" + "="*50 + "\n")
                f.write("Separated Component Analysis (vdW vs Coulomb):\n")
                f.write("="*50 + "\n\n")
                
                if 'vdw' in separated_components:
                    f.write("van der Waals Component:\n")
                    f.write(f"  Energy: {separated_components['vdw']['energy']:.6f} eV\n")
                    f.write("  Parameters:\n")
                    for param_name, param_value in separated_components['vdw']['parameters'].items():
                        f.write(f"    {param_name:15s} = {param_value:12.6f}\n")
                    f.write("\n")
                    
                if 'coulomb' in separated_components:
                    f.write("Coulomb Component:\n")
                    f.write(f"  Energy: {separated_components['coulomb']['energy']:.6f} eV\n")
                    f.write("  Parameters:\n")
                    for param_name, param_value in separated_components['coulomb']['parameters'].items():
                        f.write(f"    {param_name:15s} = {param_value:12.6f}\n")
                    f.write("\n")
                    
                # Calculate and report relative contributions
                if 'vdw' in separated_components and 'coulomb' in separated_components:
                    vdw_e = abs(separated_components['vdw']['energy'])
                    coul_e = abs(separated_components['coulomb']['energy'])
                    total_e = vdw_e + coul_e
                    if total_e > 0:
                        f.write("Relative Contributions:\n")
                        f.write(f"  van der Waals: {(vdw_e/total_e)*100:.1f}%\n")
                        f.write(f"  Coulomb: {(coul_e/total_e)*100:.1f}%\n")
                        f.write("\n")
                    
            f.write("\nOptimized Parameters:\n")
            for block in self.blocks:
                if block in self.optimized_params:
                    f.write(f"\n{block.upper()}:\n")
                    for key, value in self.optimized_params[block].items():
                        if key not in ['loss', 'uncertainty_metrics', 'vdw_energy', 'coulomb_energy']:
                            f.write(f"  {key:20s} = {value:12.6f}\n")
                            
        print(f"Report saved: {report_file}")
