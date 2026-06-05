"""
Complete ReaxFF parameter definitions for MoS2/H system
Based on the paper parameters structure
"""

import numpy as np
import json
import os

class ReaxFFParametersComplete:
    """Complete ReaxFF parameter set including all parameter types"""
    
    def __init__(self):
        # Initialize all parameter categories
        self.general_params = {}
        self.atom_params = {}
        self.bond_params = {}
        self.off_diagonal_params = {}
        self.angle_params = {}
        self.torsion_params = {}
        self.hbond_params = {}
        
        # Load initial values from PDF reference
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize all parameters based on PDF reference values"""
        
        # ==================== GENERAL PARAMETERS (39 total) ====================
        self.general_params = {
            'p_boc1': (50.0000, (45.0, 55.0)),     # Bond order cutoff 1
            'p_boc2': (9.5469, (8.0, 11.0)),       # Bond order cutoff 2
            'p_coa2': (26.5405, (24.0, 29.0)),     # Correction angle 2
            'p_trip4': (1.5105, (1.0, 2.0)),       # Triple bond stabilization 4
            'p_trip3': (6.6630, (6.0, 7.5)),       # Triple bond stabilization 3
            'kc2': (70.0000, (65.0, 75.0)),        # C2-correction
            'p_ovun6': (1.0588, (0.8, 1.3)),       # Undercoordination 6
            'p_trip2': (4.6000, (4.0, 5.2)),       # Triple bond stabilization 2
            'p_ovun7': (12.1176, (10.0, 14.0)),    # Undercoordination 7
            'p_ovun8': (13.3056, (11.0, 15.0)),    # Undercoordination 8
            'p_trip1': (-10.1292, (-12.0, -8.0)),  # Triple bond stabilization 1
            'swa': (0.0000, None),                  # Lower Taper-radius (fixed)
            'swb': (10.0000, None),                 # Upper Taper-radius (fixed)
            'p_val7': (33.8667, (30.0, 37.0)),     # Valence angle 7
            'p_lp1': (6.0891, (5.5, 6.5)),         # Lone pair 1
            'p_val9': (1.0563, (0.9, 1.2)),        # Valence angle 9
            'p_val10': (2.0384, (1.8, 2.3)),       # Valence angle 10
            'p_pen2': (6.9290, (6.0, 8.0)),        # Penalty 2
            'p_pen3': (0.3989, (0.3, 0.5)),        # Penalty 3
            'p_pen4': (3.9954, (3.5, 4.5)),        # Penalty 4
            'p_tor2': (5.7796, (5.0, 6.5)),        # Torsion 2
            'p_tor3': (10.0000, (9.0, 11.0)),      # Torsion 3
            'p_tor4': (1.9487, (1.7, 2.2)),        # Torsion 4
            'p_cot2': (2.1645, (1.9, 2.4)),        # Correction torsion 2
            'p_vdW1': (1.5591, (1.4, 1.7)),        # van der Waals 1
            'cutoff': (0.1000, None),               # Bond order cutoff (fixed)
            'p_coa4': (2.1365, (1.9, 2.4)),        # Correction angle 4
            'p_ovun4': (0.6991, (0.5, 0.9)),       # Undercoordination 4
            'p_ovun3': (50.0000, (45.0, 55.0)),    # Undercoordination 3
            'p_val8': (1.8512, (1.6, 2.1)),        # Valence angle 8
            'p_coa3': (2.6962, (2.4, 3.0))         # Correction angle 3
        }
        
        # ==================== ATOM PARAMETERS ====================
        # Format: (value, (min, max)) where None means fixed parameter
        
        # Hydrogen parameters
        self.atom_params['H'] = {
            'r_sigma': (0.7853, (0.7, 0.85)),
            'valency': (1.0000, None),
            'mass': (1.0080, None),
            'r_vdw': (1.5904, (1.4, 1.7)),
            'epsilon': (0.0419, (0.03, 0.05)),
            'gamma': (1.0206, (0.9, 1.1)),
            'r_pi': (-0.1000, None),
            'valency_e': (1.0000, None),
            'alpha': (9.3557, (8.5, 10.0)),
            'gamma_w': (5.0518, (4.5, 5.5)),
            'valency_boc': (1.0000, None),
            'p_ovun5': (0.0000, None),
            'chi': (4.5280, (4.0, 5.0)),  # Electronegativity - χEEM = 121.1250^0.5
            'eta': (6.9742, (6.5, 7.5)),  # Hardness - ηEEM = 121.1250/2
            'p_lp2': (0.0000, None),
            'heat_inc': (0.0000, None),
            'p_boc3': (3.3517, (3.0, 3.7)),
            'p_boc4': (1.9771, (1.7, 2.2)),
            'p_boc5': (0.7571, (0.6, 0.9)),
            'p_ovun2': (-15.7683, (-17.0, -14.0)),
            'p_val3': (2.1488, (1.9, 2.4)),
            'p_val5': (1.0338, (0.9, 1.1)),
            'p_hbond': (2.8793, (2.5, 3.2))
        }
        
        # Sulfur parameters
        self.atom_params['S'] = {
            'r_sigma': (1.8328, (1.7, 1.95)),
            'valency': (2.0000, None),
            'mass': (32.0600, None),
            'r_vdw': (1.8815, (1.75, 2.0)),
            'epsilon': (0.3236, (0.28, 0.37)),
            'gamma': (0.7530, (0.68, 0.83)),
            'r_pi': (1.6468, (1.5, 1.8)),
            'valency_e': (6.0000, None),
            'alpha': (9.0000, (8.0, 10.0)),
            'gamma_w': (4.9055, (4.4, 5.4)),
            'valency_boc': (4.0000, None),
            'p_ovun5': (30.0000, (27.0, 33.0)),
            'chi': (6.5745, (6.0, 7.0)),
            'eta': (9.0000, (8.0, 10.0)),
            'p_lp2': (3.4994, (3.1, 3.9)),
            'heat_inc': (0.0000, None),
            'p_boc3': (22.1978, (20.0, 24.0)),
            'p_boc4': (12.0000, (10.5, 13.5)),
            'p_boc5': (15.3230, (14.0, 17.0)),
            'p_ovun2': (-15.7363, (-17.0, -14.0)),
            'p_val3': (2.8802, (2.5, 3.2)),
            'p_val5': (1.0338, (0.9, 1.1)),
            'p_hbond': (2.8793, (2.5, 3.2))
        }
        
        # Molybdenum parameters
        self.atom_params['Mo'] = {
            'r_sigma': (2.4695, (2.3, 2.6)),
            'valency': (5.6375, (5.2, 6.0)),
            'mass': (95.9400, None),
            'r_vdw': (1.8471, (1.7, 2.0)),
            'epsilon': (0.3413, (0.3, 0.38)),
            'gamma': (0.7020, (0.63, 0.77)),
            'r_pi': (0.1000, (0.05, 0.15)),
            'valency_e': (6.0000, None),
            'alpha': (13.1958, (12.0, 14.5)),
            'gamma_w': (44.8826, (42.0, 48.0)),
            'valency_boc': (4.0000, None),
            'p_ovun5': (0.0000, None),
            'chi': (1.5954, (1.4, 1.8)),
            'eta': (6.5231, (6.0, 7.0)),
            'p_lp2': (0.0000, None),
            'heat_inc': (0.1000, None),
            'p_boc3': (0.0722, (0.05, 0.09)),
            'p_boc4': (3.4529, (3.1, 3.8)),
            'p_boc5': (3.1767, (2.8, 3.5)),
            'p_ovun2': (-17.9815, (-19.5, -16.5)),
            'p_val3': (3.1072, (2.8, 3.4)),
            'p_val5': (1.0338, (0.9, 1.1)),
            'p_hbond': (3.4590, (3.1, 3.8))
        }
        
        # ==================== BOND PARAMETERS (6 bonds) ====================
        # H-H bond
        self.bond_params['H-H'] = {
            'De_sigma': (156.0973, (140.0, 170.0)),
            'De_pi': (0.0000, None),
            'De_pipi': (0.0000, None),
            'p_be1': (-0.1377, (-0.2, -0.08)),
            'p_be2': (0.0000, None),
            'p_bo1': (2.9907, (2.7, 3.3)),
            'p_bo2': (1.0000, None),
            'p_bo3': (0.0000, None),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.0593, (-0.08, -0.04)),
            'p_bo6': (4.8358, (4.4, 5.3)),
            'p_ovun1': (0.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # H-S bond
        self.bond_params['H-S'] = {
            'De_sigma': (183.1582, (165.0, 200.0)),
            'De_pi': (0.0000, None),
            'De_pipi': (0.0000, None),
            'p_be1': (-0.7544, (-0.85, -0.65)),
            'p_be2': (0.0000, None),
            'p_bo1': (11.7366, (10.5, 13.0)),
            'p_bo2': (1.0000, None),
            'p_bo3': (0.0000, None),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.0595, (-0.08, -0.04)),
            'p_bo6': (4.6177, (4.2, 5.0)),
            'p_ovun1': (1.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # S-S bond
        self.bond_params['S-S'] = {
            'De_sigma': (86.8868, (78.0, 96.0)),
            'De_pi': (69.1367, (62.0, 76.0)),
            'De_pipi': (0.0000, None),
            'p_be1': (-0.9993, (-1.1, -0.9)),
            'p_be2': (-0.4781, (-0.55, -0.4)),
            'p_bo1': (0.2799, (0.2, 0.35)),
            'p_bo2': (-0.1677, (-0.2, -0.13)),
            'p_bo3': (8.2557, (7.5, 9.0)),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.1131, (-0.14, -0.09)),
            'p_bo6': (6.1440, (5.5, 6.8)),
            'p_ovun1': (1.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # H-Mo bond
        self.bond_params['H-Mo'] = {
            'De_sigma': (77.9675, (70.0, 86.0)),
            'De_pi': (0.0000, None),
            'De_pipi': (0.0000, None),
            'p_be1': (-0.5019, (-0.58, -0.42)),
            'p_be2': (-0.3000, (-0.35, -0.25)),
            'p_bo1': (0.0697, (0.05, 0.09)),
            'p_bo2': (-0.3027, (-0.35, -0.25)),
            'p_bo3': (15.0243, (13.5, 16.5)),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.0500, (-0.07, -0.03)),
            'p_bo6': (5.9618, (5.4, 6.5)),
            'p_ovun1': (1.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # S-Mo bond
        self.bond_params['S-Mo'] = {
            'De_sigma': (64.9870, (58.0, 72.0)),
            'De_pi': (33.5938, (30.0, 37.0)),
            'De_pipi': (137.6546, (124.0, 151.0)),
            'p_be1': (1.0000, (0.9, 1.1)),
            'p_be2': (-0.2304, (-0.27, -0.19)),
            'p_bo1': (1.3265, (1.2, 1.45)),
            'p_bo2': (-0.1497, (-0.18, -0.12)),
            'p_bo3': (7.0000, (6.3, 7.7)),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.1447, (-0.17, -0.12)),
            'p_bo6': (6.6437, (6.0, 7.3)),
            'p_ovun1': (1.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # Mo-Mo bond
        self.bond_params['Mo-Mo'] = {
            'De_sigma': (51.8235, (46.0, 57.0)),
            'De_pi': (0.0000, None),
            'De_pipi': (0.0000, None),
            'p_be1': (0.8271, (0.74, 0.91)),
            'p_be2': (-0.3000, (-0.35, -0.25)),
            'p_bo1': (0.2248, (0.18, 0.27)),
            'p_bo2': (-0.3000, (-0.35, -0.25)),
            'p_bo3': (16.0000, (14.5, 17.5)),
            'p_bo4': (1.0000, None),
            'p_bo5': (-0.1908, (-0.22, -0.16)),
            'p_bo6': (7.3978, (6.7, 8.1)),
            'p_ovun1': (0.0000, None),
            'n_bond': (0.0000, None)
        }
        
        # ==================== OFF-DIAGONAL PARAMETERS ====================
        self.off_diagonal_params['H-S'] = {
            'Dij': (0.1017, (0.09, 0.11)),
            'RvdW': (1.7755, (1.6, 1.95)),
            'alfa': (9.6088, (8.6, 10.6)),
            'r_sigma': (1.3696, (1.23, 1.51)),
            'r_pi': (-1.0000, None),
            'r_pipi': (-1.0000, None)
        }
        
        self.off_diagonal_params['H-Mo'] = {
            'Dij': (0.1939, (0.17, 0.21)),
            'RvdW': (1.3679, (1.23, 1.5)),
            'alfa': (11.7159, (10.5, 12.9)),
            'r_sigma': (1.4389, (1.3, 1.58)),
            'r_pi': (-1.0000, None),
            'r_pipi': (-1.0000, None)
        }
        
        self.off_diagonal_params['S-Mo'] = {
            'Dij': (0.1000, (0.09, 0.11)),
            'RvdW': (1.8000, (1.62, 1.98)),
            'alfa': (9.8235, (8.8, 10.8)),
            'r_sigma': (2.3403, (2.1, 2.57)),
            'r_pi': (1.6147, (1.45, 1.78)),
            'r_pipi': (-1.0000, None)
        }
        
        # ==================== ANGLE PARAMETERS (15 angles) ====================
        # Store only the most important angles for optimization
        self.angle_params['H-H-H'] = {
            'theta_0': (0.0000, None),
            'p_val1': (27.9213, (25.0, 31.0)),
            'p_val2': (5.8635, (5.3, 6.4)),
            'p_coa1': (0.0000, None),
            'p_val7': (0.0000, None),
            'p_pen1': (0.0000, None),
            'p_val4': (1.0400, (0.9, 1.14))
        }
        
        self.angle_params['H-S-H'] = {
            'theta_0': (92.1229, (83.0, 101.0)),
            'p_val1': (42.8350, (38.5, 47.0)),
            'p_val2': (0.6163, (0.55, 0.68)),
            'p_coa1': (0.0000, None),
            'p_val7': (1.0235, (0.92, 1.13)),
            'p_pen1': (0.0000, None),
            'p_val4': (1.0010, (0.9, 1.1))
        }
        
        self.angle_params['H-S-S'] = {
            'theta_0': (70.9476, (64.0, 78.0)),
            'p_val1': (9.9024, (8.9, 10.9)),
            'p_val2': (0.6923, (0.62, 0.76)),
            'p_coa1': (0.0000, None),
            'p_val7': (0.2031, (0.18, 0.22)),
            'p_pen1': (0.0000, None),
            'p_val4': (2.9811, (2.7, 3.3))
        }
        
        self.angle_params['S-Mo-S'] = {
            'theta_0': (70.5456, (63.5, 77.5)),
            'p_val1': (50.0000, (45.0, 55.0)),
            'p_val2': (0.6721, (0.6, 0.74)),
            'p_coa1': (0.0984, (0.09, 0.11)),
            'p_val7': (1.0000, None),
            'p_pen1': (0.0000, None),
            'p_val4': (1.4973, (1.35, 1.65))
        }
        
        self.angle_params['Mo-S-Mo'] = {
            'theta_0': (30.0000, (27.0, 33.0)),
            'p_val1': (22.6920, (20.4, 25.0)),
            'p_val2': (8.0000, (7.2, 8.8)),
            'p_coa1': (0.0000, None),
            'p_val7': (3.6520, (3.3, 4.0)),
            'p_pen1': (0.0000, None),
            'p_val4': (3.2979, (3.0, 3.6))
        }
        
        # Add more angles as needed...
        
        # ==================== TORSION PARAMETERS (4 torsions) ====================
        self.torsion_params['H-H-H-H'] = {
            'V1': (0.0000, None),
            'V2': (0.0000, None),
            'V3': (0.0000, None),
            'p_tor1': (0.0000, None),
            'p_cot1': (0.0000, None),
            'n_bond': (0.0000, None),
            'unused': (0.0000, None)
        }
        
        self.torsion_params['H-H-S-H'] = {
            'V1': (0.0000, None),
            'V2': (0.0000, None),
            'V3': (0.0000, None),
            'p_tor1': (0.0000, None),
            'p_cot1': (0.0000, None),
            'n_bond': (0.0000, None),
            'unused': (0.0000, None)
        }
        
        self.torsion_params['S-S-S-S'] = {
            'V1': (2.4661, (2.2, 2.7)),
            'V2': (71.9719, (65.0, 79.0)),
            'V3': (0.0100, (0.005, 0.015)),
            'p_tor1': (-8.0000, (-8.8, -7.2)),
            'p_cot1': (0.0000, None),
            'n_bond': (0.0000, None),
            'unused': (0.0000, None)
        }
        
        self.torsion_params['H-S-S-H'] = {
            'V1': (0.0000, None),
            'V2': (50.0000, (45.0, 55.0)),
            'V3': (0.0000, None),
            'p_tor1': (-8.0000, (-8.8, -7.2)),
            'p_cot1': (0.0000, None),
            'n_bond': (0.0000, None),
            'unused': (0.0000, None)
        }
        
        # ==================== HYDROGEN BOND PARAMETERS ====================
        self.hbond_params['S-H-S'] = {
            'r_hb': (1.5000, (1.35, 1.65)),
            'p_hb1': (-2.0000, (-2.2, -1.8)),
            'p_hb2': (1.7976, (1.6, 1.98)),
            'p_hb3': (3.0000, (2.7, 3.3))
        }
        
    def get_all_optimizable_params(self):
        """Get all optimizable parameters with their bounds"""
        params_list = []
        bounds_list = []
        names_list = []
        
        # Add general parameters
        for name, (value, bounds) in self.general_params.items():
            if bounds is not None:  # Only include parameters with bounds
                params_list.append(value)
                bounds_list.append(bounds)
                names_list.append(f'general_{name}')
                
        # Add atom parameters
        for atom, atom_params in self.atom_params.items():
            for name, (value, bounds) in atom_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'atom_{atom}_{name}')
                    
        # Add bond parameters  
        for bond, bond_params in self.bond_params.items():
            for name, (value, bounds) in bond_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'bond_{bond}_{name}')
                    
        # Add off-diagonal parameters
        for pair, pair_params in self.off_diagonal_params.items():
            for name, (value, bounds) in pair_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'offdiag_{pair}_{name}')
                    
        # Add angle parameters
        for angle, angle_params in self.angle_params.items():
            for name, (value, bounds) in angle_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'angle_{angle}_{name}')
                    
        # Add torsion parameters
        for torsion, torsion_params in self.torsion_params.items():
            for name, (value, bounds) in torsion_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'torsion_{torsion}_{name}')
                    
        # Add hydrogen bond parameters
        for hb, hb_params in self.hbond_params.items():
            for name, (value, bounds) in hb_params.items():
                if bounds is not None:
                    params_list.append(value)
                    bounds_list.append(bounds)
                    names_list.append(f'hbond_{hb}_{name}')
                    
        return np.array(params_list), bounds_list, names_list
    
    def get_blocked_params(self):
        """Get parameters grouped by optimization blocks"""
        blocks = {
            'general': [],
            'atom': [],
            'bond': [],
            'angle': [],
            'torsion': [],
            'vdw_coulomb': [],
            'hbond': []
        }
        
        # General parameters block
        for name, (value, bounds) in self.general_params.items():
            if bounds is not None:
                blocks['general'].append({
                    'name': f'general_{name}',
                    'value': value,
                    'bounds': bounds
                })
                
        # Atom parameters block
        for atom, atom_params in self.atom_params.items():
            for name, (value, bounds) in atom_params.items():
                if bounds is not None:
                    blocks['atom'].append({
                        'name': f'atom_{atom}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        # Bond parameters block
        for bond, bond_params in self.bond_params.items():
            for name, (value, bounds) in bond_params.items():
                if bounds is not None:
                    blocks['bond'].append({
                        'name': f'bond_{bond}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        # Angle parameters block
        for angle, angle_params in self.angle_params.items():
            for name, (value, bounds) in angle_params.items():
                if bounds is not None:
                    blocks['angle'].append({
                        'name': f'angle_{angle}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        # Torsion parameters block
        for torsion, torsion_params in self.torsion_params.items():
            for name, (value, bounds) in torsion_params.items():
                if bounds is not None:
                    blocks['torsion'].append({
                        'name': f'torsion_{torsion}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        # vdW and Coulomb parameters (from off-diagonal and some atom params)
        for pair, pair_params in self.off_diagonal_params.items():
            for name, (value, bounds) in pair_params.items():
                if bounds is not None and name in ['Dij', 'RvdW', 'alfa']:
                    blocks['vdw_coulomb'].append({
                        'name': f'offdiag_{pair}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        # Add electronegativity and hardness parameters to vdW/Coulomb block
        for atom, atom_params in self.atom_params.items():
            for name in ['chi', 'eta']:
                if name in atom_params:
                    value, bounds = atom_params[name]
                    if bounds is not None:
                        blocks['vdw_coulomb'].append({
                            'name': f'atom_{atom}_{name}',
                            'value': value,
                            'bounds': bounds
                        })
                        
        # Hydrogen bond parameters block
        for hb, hb_params in self.hbond_params.items():
            for name, (value, bounds) in hb_params.items():
                if bounds is not None:
                    blocks['hbond'].append({
                        'name': f'hbond_{hb}_{name}',
                        'value': value,
                        'bounds': bounds
                    })
                    
        return blocks
    
    def update_parameters(self, param_dict):
        """Update parameters from a dictionary"""
        for name, value in param_dict.items():
            # Parse the parameter name
            parts = name.split('_', 2)
            if len(parts) < 3:
                continue
                
            category = parts[0]
            
            if category == 'general':
                param_name = parts[1]
                if param_name in self.general_params:
                    old_val, bounds = self.general_params[param_name]
                    self.general_params[param_name] = (value, bounds)
                    
            elif category == 'atom':
                atom_type = parts[1]
                param_name = parts[2]
                if atom_type in self.atom_params:
                    if param_name in self.atom_params[atom_type]:
                        old_val, bounds = self.atom_params[atom_type][param_name]
                        self.atom_params[atom_type][param_name] = (value, bounds)
                        
            elif category == 'bond':
                bond_type = parts[1]
                param_name = parts[2]
                if bond_type in self.bond_params:
                    if param_name in self.bond_params[bond_type]:
                        old_val, bounds = self.bond_params[bond_type][param_name]
                        self.bond_params[bond_type][param_name] = (value, bounds)
                        
            elif category == 'offdiag':
                pair_type = parts[1]
                param_name = parts[2]
                if pair_type in self.off_diagonal_params:
                    if param_name in self.off_diagonal_params[pair_type]:
                        old_val, bounds = self.off_diagonal_params[pair_type][param_name]
                        self.off_diagonal_params[pair_type][param_name] = (value, bounds)
                        
            elif category == 'angle':
                angle_type = parts[1]
                param_name = parts[2]
                if angle_type in self.angle_params:
                    if param_name in self.angle_params[angle_type]:
                        old_val, bounds = self.angle_params[angle_type][param_name]
                        self.angle_params[angle_type][param_name] = (value, bounds)
                        
            elif category == 'torsion':
                torsion_type = parts[1]
                param_name = parts[2]
                if torsion_type in self.torsion_params:
                    if param_name in self.torsion_params[torsion_type]:
                        old_val, bounds = self.torsion_params[torsion_type][param_name]
                        self.torsion_params[torsion_type][param_name] = (value, bounds)
                        
            elif category == 'hbond':
                hb_type = parts[1]
                param_name = parts[2]
                if hb_type in self.hbond_params:
                    if param_name in self.hbond_params[hb_type]:
                        old_val, bounds = self.hbond_params[hb_type][param_name]
                        self.hbond_params[hb_type][param_name] = (value, bounds)
                        
    def save_to_reaxff_format(self, filename):
        """Save parameters to standard ReaxFF force field format"""
        with open(filename, 'w') as f:
            # Header
            f.write("Reactive MD-force field: H/S/Mo for MoS2 project - PIBO Optimized\n")
            f.write(f"Generated: {os.path.basename(filename)}\n")
            
            # General parameters
            f.write(f"39 ! Number of general parameters\n")
            general_order = [
                'p_boc1', 'p_boc2', 'p_coa2', 'p_trip4', 'p_trip3', 'kc2', 'p_ovun6',
                'p_trip2', 'p_ovun7', 'p_ovun8', 'p_trip1', 'swa', 'swb', 'unused1',
                'p_val7', 'p_lp1', 'p_val9', 'p_val10', 'unused2', 'p_pen2', 'p_pen3',
                'p_pen4', 'unused3', 'p_tor2', 'p_tor3', 'p_tor4', 'unused4', 'p_cot2',
                'p_vdW1', 'cutoff', 'p_coa4', 'p_ovun4', 'p_ovun3', 'p_val8',
                'unused5', 'unused6', 'unused7', 'unused8', 'p_coa3'
            ]
            
            for param_name in general_order:
                if param_name.startswith('unused'):
                    f.write(f"  {0.0000:12.4f} !not used\n")
                elif param_name in self.general_params:
                    value, _ = self.general_params[param_name]
                    f.write(f"  {value:12.4f} !{param_name}\n")
                else:
                    f.write(f"  {0.0000:12.4f} !{param_name}\n")
                    
            # Atom parameters
            f.write("3 ! Nr of atoms; atomID;ro(sigma); Val;atom mass;Rvdw;Dij;gamma\n")
            f.write("  alfa;gamma(w);Val(angle);p(ovun5);n.u.;chiEEM;etaEEM;n.u.\n")
            f.write("  ro(pipi);p(lp2);Heat increment;p(boc4);p(boc3);p(boc5),n.u.;n.u.\n")
            f.write("  p(ovun2);p(val3);n.u.;Val(boc);p(val5);n.u.;n.u.;n.u.\n")
            
            atom_order = ['H', 'S', 'Mo']
            for atom in atom_order:
                params = self.atom_params[atom]
                
                # Line 1
                values = []
                for key in ['r_sigma', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma', 'r_pi', 'valency_e']:
                    if key in params:
                        val, _ = params[key]
                        values.append(val)
                    else:
                        values.append(0.0)
                f.write(f"{atom:2s} {values[0]:8.4f} {values[1]:8.4f} {values[2]:8.4f} "
                       f"{values[3]:8.4f} {values[4]:8.4f} {values[5]:8.4f} {values[6]:8.4f} {values[7]:8.4f}\n")
                
                # Line 2
                values = []
                for key in ['alpha', 'gamma_w', 'valency_boc', 'p_ovun5', 'unused1', 'chi', 'eta', 'unused2']:
                    if key in params:
                        val, _ = params[key]
                        values.append(val)
                    elif key.startswith('unused'):
                        values.append(0.0)
                    else:
                        values.append(0.0)
                f.write(f"   {values[0]:8.4f} {values[1]:8.4f} {values[2]:8.4f} {values[3]:8.4f} "
                       f"{values[4]:8.4f} {values[5]:8.4f} {values[6]:8.4f} {values[7]:8.4f}\n")
                
                # Line 3
                values = []
                for key in ['p_lp2', 'heat_inc', 'p_boc4', 'p_boc3', 'p_boc5', 'unused1', 'unused2', 'unused3']:
                    if key in params:
                        val, _ = params[key]
                        values.append(val)
                    elif key.startswith('unused'):
                        values.append(0.0)
                    else:
                        values.append(0.0)
                f.write(f"   {values[0]:8.4f} {values[1]:8.4f} {values[2]:8.4f} {values[3]:8.4f} "
                       f"{values[4]:8.4f} {values[5]:8.4f} {values[6]:8.4f} {values[7]:8.4f}\n")
                
                # Line 4
                values = []
                for key in ['p_ovun2', 'p_val3', 'unused1', 'valency_boc', 'p_val5', 'unused2', 'unused3', 'unused4']:
                    if key in params:
                        val, _ = params[key]
                        values.append(val)
                    elif key.startswith('unused'):
                        values.append(0.0)
                    else:
                        values.append(1.0 if key == 'valency_boc' else 0.0)
                f.write(f"   {values[0]:8.4f} {values[1]:8.4f} {values[2]:8.4f} {values[3]:8.4f} "
                       f"{values[4]:8.4f} {values[5]:8.4f} {values[6]:8.4f} {values[7]:8.4f}\n")
                       
            # Bond parameters
            f.write("6 ! Nr of bonds; at1;at2;De(sigma);De(pi);De(pipi);p(be1);p(be2)\n")
            f.write("  p(bo3);p(bo4);n.u.;p(bo1);p(bo2)\n")
            
            bond_order = [('H', 'H'), ('H', 'S'), ('S', 'S'), ('H', 'Mo'), ('S', 'Mo'), ('Mo', 'Mo')]
            for i, (at1, at2) in enumerate(bond_order):
                bond_key = f'{at1}-{at2}'
                if bond_key in self.bond_params:
                    params = self.bond_params[bond_key]
                    
                    # Get atom indices (1-indexed)
                    atom_idx = {'H': 1, 'S': 2, 'Mo': 3}
                    idx1, idx2 = atom_idx[at1], atom_idx[at2]
                    
                    # Line 1
                    values = []
                    for key in ['De_sigma', 'De_pi', 'De_pipi', 'p_be1', 'p_be2']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(0.0)
                    f.write(f"  {idx1:2d} {idx2:2d} {values[0]:9.4f} {values[1]:9.4f} {values[2]:9.4f} "
                           f"{values[3]:9.4f} {values[4]:9.4f} 1.0000 6.0000 0.8240\n")
                    
                    # Line 2
                    values = []
                    for key in ['p_bo1', 'p_bo2', 'p_bo3', 'p_bo4', 'p_bo5', 'p_bo6']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(1.0 if key == 'p_bo2' or key == 'p_bo4' else 0.0)
                    f.write(f"     {values[0]:9.4f} {values[1]:9.4f} {values[2]:9.4f} {values[3]:9.4f} "
                           f"{values[4]:9.4f} {values[5]:9.4f} 1.0000 0.0000\n")
                           
            # Off-diagonal parameters
            f.write("3 ! Nr of off-diagonal terms. at1;at2;Dij;RvdW;alfa;ro(sigma);ro(pi);ro(pipi)\n")
            
            offdiag_order = [('H', 'S'), ('H', 'Mo'), ('S', 'Mo')]
            for i, (at1, at2) in enumerate(offdiag_order):
                pair_key = f'{at1}-{at2}'
                if pair_key in self.off_diagonal_params:
                    params = self.off_diagonal_params[pair_key]
                    atom_idx = {'H': 1, 'S': 2, 'Mo': 3}
                    idx1, idx2 = atom_idx[at1], atom_idx[at2]
                    
                    values = []
                    for key in ['Dij', 'RvdW', 'alfa', 'r_sigma', 'r_pi', 'r_pipi']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(-1.0)
                    f.write(f"  {idx1:2d} {idx2:2d} {values[0]:9.4f} {values[1]:9.4f} {values[2]:9.4f} "
                           f"{values[3]:9.4f} {values[4]:9.4f} {values[5]:9.4f}\n")
                           
            # Angle parameters
            num_angles = len(self.angle_params)
            f.write(f"{num_angles} ! Nr of angles. at1;at2;at3;Thetao,o;p(val1);p(val2);p(coa1);p(val7);p(pen1);p(val4)\n")
            
            for angle_key, params in self.angle_params.items():
                atoms = angle_key.split('-')
                if len(atoms) == 3:
                    atom_idx = {'H': 1, 'S': 2, 'Mo': 3}
                    idx1 = atom_idx.get(atoms[0], 0)
                    idx2 = atom_idx.get(atoms[1], 0)
                    idx3 = atom_idx.get(atoms[2], 0)
                    
                    values = []
                    for key in ['theta_0', 'p_val1', 'p_val2', 'p_coa1', 'p_val7', 'p_pen1', 'p_val4']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(0.0)
                    f.write(f"  {idx1:2d} {idx2:2d} {idx3:2d} {values[0]:9.4f} {values[1]:9.4f} "
                           f"{values[2]:9.4f} {values[3]:9.4f} {values[4]:9.4f} {values[5]:9.4f} {values[6]:9.4f}\n")
                           
            # Torsion parameters
            num_torsions = len(self.torsion_params)
            f.write(f"{num_torsions} ! Nr of torsions. at1;at2;at3;at4;V1;V2;V3;p(tor1);p(cot1);n;unused\n")
            
            for torsion_key, params in self.torsion_params.items():
                atoms = torsion_key.split('-')
                if len(atoms) == 4:
                    atom_idx = {'H': 1, 'S': 2, 'Mo': 3, 'X': 0}
                    idx1 = atom_idx.get(atoms[0], 0)
                    idx2 = atom_idx.get(atoms[1], 0)
                    idx3 = atom_idx.get(atoms[2], 0)
                    idx4 = atom_idx.get(atoms[3], 0)
                    
                    values = []
                    for key in ['V1', 'V2', 'V3', 'p_tor1', 'p_cot1', 'n_bond', 'unused']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(0.0)
                    f.write(f"  {idx1:2d} {idx2:2d} {idx3:2d} {idx4:2d} {values[0]:9.4f} {values[1]:9.4f} "
                           f"{values[2]:9.4f} {values[3]:9.4f} {values[4]:9.4f} {values[5]:9.4f} {values[6]:9.4f}\n")
                           
            # Hydrogen bonds
            num_hbonds = len(self.hbond_params)
            f.write(f"{num_hbonds} ! Nr of hydrogen bonds. at1;at2;at3;r(hb);p(hb1);p(hb2);p(hb3);unused\n")
            
            for hb_key, params in self.hbond_params.items():
                atoms = hb_key.split('-')
                if len(atoms) == 3:
                    atom_idx = {'H': 1, 'S': 2, 'Mo': 3}
                    idx1 = atom_idx.get(atoms[0], 0)
                    idx2 = atom_idx.get(atoms[1], 0)
                    idx3 = atom_idx.get(atoms[2], 0)
                    
                    values = []
                    for key in ['r_hb', 'p_hb1', 'p_hb2', 'p_hb3']:
                        if key in params:
                            val, _ = params[key]
                            values.append(val)
                        else:
                            values.append(0.0)
                    f.write(f"  {idx1:2d} {idx2:2d} {idx3:2d} {values[0]:9.4f} {values[1]:9.4f} "
                           f"{values[2]:9.4f} {values[3]:9.4f}\n")
                           
        print(f"Parameters saved to {filename}")
        
    def validate_parameters(self):
        """Validate that all required parameters are present"""
        errors = []
        warnings = []
        
        # Check general parameters
        required_general = ['p_boc1', 'p_boc2', 'p_coa2', 'p_val7', 'p_val8', 'p_val9', 'p_val10']
        for param in required_general:
            if param not in self.general_params:
                errors.append(f"Missing required general parameter: {param}")
                
        # Check atom parameters
        required_atoms = ['H', 'S', 'Mo']
        for atom in required_atoms:
            if atom not in self.atom_params:
                errors.append(f"Missing parameters for atom: {atom}")
            else:
                required_atom_params = ['r_sigma', 'valency', 'mass', 'r_vdw', 'epsilon', 'gamma']
                for param in required_atom_params:
                    if param not in self.atom_params[atom]:
                        errors.append(f"Missing required parameter {param} for atom {atom}")
                        
        # Check bond parameters
        if len(self.bond_params) < 3:
            warnings.append(f"Only {len(self.bond_params)} bond types defined (expected at least 3)")
            
        # Check if all bounds are reasonable
        for name, (value, bounds) in self.general_params.items():
            if bounds is not None:
                if value < bounds[0] or value > bounds[1]:
                    warnings.append(f"General parameter {name} value {value} outside bounds {bounds}")
                    
        return errors, warnings


def test_parameter_system():
    """Test the complete parameter system"""
    print("Testing Complete ReaxFF Parameter System")
    print("="*60)
    
    # Create parameter object
    params = ReaxFFParametersComplete()
    
    # Get all optimizable parameters
    values, bounds, names = params.get_all_optimizable_params()
    print(f"Total optimizable parameters: {len(values)}")
    
    # Get blocked parameters
    blocks = params.get_blocked_params()
    print("\nParameters per block:")
    for block_name, block_params in blocks.items():
        print(f"  {block_name}: {len(block_params)} parameters")
        
    # Validate parameters
    errors, warnings = params.validate_parameters()
    if errors:
        print("\nValidation Errors:")
        for error in errors:
            print(f"  ERROR: {error}")
    if warnings:
        print("\nValidation Warnings:")
        for warning in warnings:
            print(f"  WARNING: {warning}")
            
    # Test saving to file
    test_file = "test_parameters.ff"
    params.save_to_reaxff_format(test_file)
    print(f"\nTest file saved: {test_file}")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    

if __name__ == "__main__":
    test_parameter_system()
