"""
Utilities for creating publication-quality DFT vs ReaxFF comparison plots
Similar to the style in scientific papers
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from typing import Dict, List, Tuple, Optional

# Set publication quality defaults
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['xtick.minor.width'] = 1.0
mpl.rcParams['ytick.minor.width'] = 1.0
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 8


class DFTComparisonPlotter:
    """Create publication-quality DFT vs ReaxFF comparison plots"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_energy_scan(self, 
                         scan_values: np.ndarray,
                         dft_energies: np.ndarray, 
                         reaxff_energies: np.ndarray,
                         scan_type: str,
                         molecule_name: str = "MoS₂",
                         save_prefix: str = "comparison",
                         units: Optional[Dict] = None):
        """
        Create energy vs scan parameter plot in publication style
        
        Args:
            scan_values: Array of scan parameter values (bond length, angle, etc.)
            dft_energies: DFT/QM reference energies
            reaxff_energies: ReaxFF calculated energies  
            scan_type: Type of scan ('bond', 'angle', 'torsion')
            molecule_name: Name of molecule for title
            save_prefix: Prefix for saved files
            units: Dictionary with unit specifications
        """
        
        # Set default units based on scan type
        if units is None:
            units = self._get_default_units(scan_type)
        
        # Shift energies to minimum
        dft_min = np.min(dft_energies)
        reaxff_min = np.min(reaxff_energies)
        
        # Option 1: Shift both to their own minimum
        dft_shifted = dft_energies - dft_min
        reaxff_shifted = reaxff_energies - reaxff_min
        
        # Option 2: Shift both to DFT minimum (to see absolute error)
        # dft_shifted = dft_energies - dft_min
        # reaxff_shifted = reaxff_energies - dft_min
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot data
        ax.plot(scan_values, reaxff_shifted, 'o-', color='red', 
                label='ReaxFF', markerfacecolor='red', markeredgecolor='darkred',
                markeredgewidth=0.5, markersize=6)
        
        ax.plot(scan_values, dft_shifted, 's-', color='black',
                label='QM', markerfacecolor='black', markeredgecolor='black',
                markeredgewidth=0.5, markersize=5)
        
        # Labels and formatting
        ax.set_xlabel(units['xlabel'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy (Kcal/mol)', fontsize=12, fontweight='bold')
        
        # Title
        title = f"{molecule_name}"
        ax.text(0.5, 1.05, title, transform=ax.transAxes,
                ha='center', fontsize=12, fontweight='bold')
        
        # Legend
        ax.legend(loc='best', frameon=True, fancybox=False,
                 edgecolor='black', fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Axis formatting
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Set axis limits with some padding
        x_range = np.max(scan_values) - np.min(scan_values)
        y_max = max(np.max(dft_shifted), np.max(reaxff_shifted))
        
        ax.set_xlim(np.min(scan_values) - 0.05*x_range, 
                   np.max(scan_values) + 0.05*x_range)
        ax.set_ylim(-0.05*y_max, y_max*1.1)
        
        # Add minor ticks
        ax.minorticks_on()
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, f'{save_prefix}_{scan_type}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        # Save data for Origin
        self._save_data_for_origin(scan_values, dft_energies, reaxff_energies,
                                   dft_shifted, reaxff_shifted,
                                   scan_type, save_prefix)
        
        plt.close()
        
        return fig_path
    
    def plot_multiple_scans(self, data_dict: Dict, molecule_name: str = "MoS₂"):
        """
        Create subplot figure with multiple scan types
        
        Args:
            data_dict: Dictionary with scan data for each type
            molecule_name: Name of molecule
        """
        
        n_plots = len(data_dict)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        
        for idx, (scan_type, data) in enumerate(data_dict.items()):
            ax = axes[idx]
            
            scan_values = data['scan_values']
            dft_energies = data['dft_energies']
            reaxff_energies = data['reaxff_energies']
            
            # Shift energies
            dft_min = np.min(dft_energies)
            reaxff_min = np.min(reaxff_energies)
            dft_shifted = dft_energies - dft_min
            reaxff_shifted = reaxff_energies - reaxff_min
            
            # Plot
            ax.plot(scan_values, reaxff_shifted, 'o-', color='red',
                   label='ReaxFF', markerfacecolor='red', markeredgecolor='darkred',
                   markeredgewidth=0.5, markersize=6)
            
            ax.plot(scan_values, dft_shifted, 's-', color='black',
                   label='QM', markerfacecolor='black', markeredgecolor='black',
                   markeredgewidth=0.5, markersize=5)
            
            # Units
            units = self._get_default_units(scan_type)
            ax.set_xlabel(units['xlabel'], fontsize=10)
            
            if idx == 0:
                ax.set_ylabel('Energy (Kcal/mol)', fontsize=10)
            
            # Subplot label
            ax.text(0.05, 0.95, labels[idx], transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top')
            
            # Title
            ax.set_title(f"{molecule_name} - {scan_type}", fontsize=10)
            
            # Legend only on first subplot
            if idx == 0:
                ax.legend(loc='best', frameon=True, fontsize=8)
            
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.minorticks_on()
        
        plt.tight_layout()
        
        # Save
        fig_path = os.path.join(self.output_dir, 'all_comparisons.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.close()
        
        return fig_path
    
    def _get_default_units(self, scan_type: str) -> Dict:
        """Get default units for different scan types"""
        
        units = {
            'bond': {
                'xlabel': 'Bond Length (Ang)',
                'unit': 'Angstrom'
            },
            'angle': {
                'xlabel': 'θ (degrees)',
                'unit': 'degrees'
            },
            'torsion': {
                'xlabel': 'Dihedral Angle (degrees)',
                'unit': 'degrees'
            },
            'vdw': {
                'xlabel': 'Distance (Ang)',
                'unit': 'Angstrom'
            }
        }
        
        return units.get(scan_type, units['bond'])
    
    def _save_data_for_origin(self, 
                              scan_values: np.ndarray,
                              dft_energies: np.ndarray,
                              reaxff_energies: np.ndarray,
                              dft_shifted: np.ndarray,
                              reaxff_shifted: np.ndarray,
                              scan_type: str,
                              save_prefix: str):
        """Save data in formats suitable for Origin and other plotting software"""
        
        # Create DataFrame
        df = pd.DataFrame({
            f'{scan_type}_value': scan_values,
            'DFT_energy_raw': dft_energies,
            'ReaxFF_energy_raw': reaxff_energies,
            'DFT_energy_shifted': dft_shifted,
            'ReaxFF_energy_shifted': reaxff_shifted,
            'Error': reaxff_energies - dft_energies,
            'Error_shifted': reaxff_shifted - dft_shifted
        })
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f'{save_prefix}_{scan_type}_data.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        # Save as Excel
        excel_path = os.path.join(self.output_dir, f'{save_prefix}_{scan_type}_data.xlsx')
        df.to_excel(excel_path, index=False, float_format='%.6f')
        
        # Save as Origin-friendly format (tab-separated)
        origin_path = os.path.join(self.output_dir, f'{save_prefix}_{scan_type}_origin.txt')
        with open(origin_path, 'w') as f:
            # Write header
            f.write(f"# {scan_type.upper()} scan data for Origin\n")
            f.write(f"# Molecule: MoS2\n")
            f.write(f"# Columns: {scan_type}_value\tDFT_energy\tReaxFF_energy\tDFT_shifted\tReaxFF_shifted\n")
            f.write("#" + "="*60 + "\n")
            
            # Write data
            for i in range(len(scan_values)):
                f.write(f"{scan_values[i]:.4f}\t{dft_energies[i]:.6f}\t{reaxff_energies[i]:.6f}\t")
                f.write(f"{dft_shifted[i]:.6f}\t{reaxff_shifted[i]:.6f}\n")
        
        print(f"Data saved to:")
        print(f"  CSV: {csv_path}")
        print(f"  Excel: {excel_path}")
        print(f"  Origin: {origin_path}")
        
    def create_parity_plot(self,
                           dft_energies: np.ndarray,
                           reaxff_energies: np.ndarray,
                           save_prefix: str = "parity"):
        """Create parity plot comparing DFT and ReaxFF energies"""
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot data
        ax.scatter(dft_energies, reaxff_energies, alpha=0.6, s=50,
                  edgecolors='black', linewidth=0.5)
        
        # Perfect agreement line
        min_e = min(dft_energies.min(), reaxff_energies.min())
        max_e = max(dft_energies.max(), reaxff_energies.max())
        ax.plot([min_e, max_e], [min_e, max_e], 'r--', 
               label='Perfect Agreement', linewidth=2)
        
        # Calculate statistics
        rmse = np.sqrt(np.mean((reaxff_energies - dft_energies)**2))
        mae = np.mean(np.abs(reaxff_energies - dft_energies))
        r2 = np.corrcoef(dft_energies, reaxff_energies)[0, 1]**2
        
        # Add statistics text
        stats_text = f'RMSE = {rmse:.3f} eV\nMAE = {mae:.3f} eV\nR² = {r2:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Labels
        ax.set_xlabel('DFT Energy (eV)', fontsize=12, fontweight='bold')
        ax.set_ylabel('ReaxFF Energy (eV)', fontsize=12, fontweight='bold')
        ax.set_title('DFT vs ReaxFF Parity Plot', fontsize=14, fontweight='bold')
        
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.minorticks_on()
        
        # Make it square
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save
        fig_path = os.path.join(self.output_dir, f'{save_prefix}.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.close()
        
        return fig_path


def convert_to_kcalmol(energies_ev: np.ndarray) -> np.ndarray:
    """Convert energies from eV to kcal/mol"""
    # 1 eV = 23.06035 kcal/mol
    return energies_ev * 23.06035


def extract_scan_parameter(structure_data: Dict, scan_type: str) -> float:
    """
    Extract the scan parameter value from structure data
    
    Args:
        structure_data: Dictionary with structure information
        scan_type: Type of scan ('bond', 'angle', 'torsion')
        
    Returns:
        Scan parameter value
    """
    
    if scan_type == 'bond':
        # Extract bond length from positions
        if 'positions' in structure_data:
            positions = structure_data['positions']
            if len(positions) >= 2:
                # Calculate distance between first two atoms
                return np.linalg.norm(positions[1] - positions[0])
        # Fallback to scan_value if available
        return structure_data.get('scan_value', 0.0)
        
    elif scan_type == 'angle':
        # Extract angle value
        if 'angle_value' in structure_data:
            return structure_data['angle_value']
        elif 'scan_value' in structure_data:
            # Assume scan_value is in degrees
            return structure_data['scan_value']
        else:
            # Calculate from positions if available
            if 'positions' in structure_data and len(structure_data['positions']) >= 3:
                pos = structure_data['positions']
                v1 = pos[0] - pos[1]  # Vector from center to atom 1
                v2 = pos[2] - pos[1]  # Vector from center to atom 2
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
                return np.degrees(angle_rad)
        return 0.0
        
    elif scan_type == 'torsion':
        # Extract torsion/dihedral angle
        if 'torsion_value' in structure_data:
            return structure_data['torsion_value']
        elif 'dihedral_value' in structure_data:
            return structure_data['dihedral_value']
        elif 'scan_value' in structure_data:
            return structure_data['scan_value']
        else:
            # Calculate dihedral from positions if available
            if 'positions' in structure_data and len(structure_data['positions']) >= 4:
                pos = structure_data['positions']
                # Calculate dihedral angle between 4 atoms
                b1 = pos[1] - pos[0]
                b2 = pos[2] - pos[1]
                b3 = pos[3] - pos[2]
                
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                
                n1 = n1 / (np.linalg.norm(n1) + 1e-10)
                n2 = n2 / (np.linalg.norm(n2) + 1e-10)
                
                m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))
                
                x = np.dot(n1, n2)
                y = np.dot(m1, n2)
                
                dihedral_rad = np.arctan2(y, x)
                return np.degrees(dihedral_rad)
        return 0.0
        
    else:  # vdw or other
        return structure_data.get('scan_value', 0.0)
