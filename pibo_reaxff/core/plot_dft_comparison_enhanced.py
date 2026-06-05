"""
Modified plot_dft_comparison method for IntegratedPIBO class
This replaces the original method to create publication-quality plots
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os

def plot_dft_comparison_enhanced(self, block: str, params: np.ndarray, 
                                param_names: List[str], output_dir: str):
    """
    Enhanced DFT vs ReaxFF comparison with publication-quality plots
    and data export for Origin
    """
    
    # Import plotting utilities
    from core.plotting_utils import (
        DFTComparisonPlotter, 
        convert_to_kcalmol,
        extract_scan_parameter
    )
    
    # Update parameters
    param_dict = {name: val for name, val in zip(param_names, params)}
    self.params.update_parameters(param_dict)
    
    # Create plotter
    plotter = DFTComparisonPlotter(output_dir)
    
    # Prepare data storage
    scan_data = {
        'scan_values': [],
        'dft_energies': [],
        'reaxff_energies': [],
        'structures': []
    }
    
    # Process each DFT sample
    for i, sample in enumerate(self.dft_data[block]):
        try:
            # Extract structure
            structure = {
                'positions': sample.get('R', sample.get('positions')),
                'types': sample.get('types'),
                'pairs': sample.get('pairs'),
                'triplets': sample.get('triplets'),
                'quads': sample.get('quads')
            }
            
            # Get scan parameter value
            if 'scan_value' in sample:
                scan_value = sample['scan_value']
            else:
                # Extract scan value based on block type
                scan_value = extract_scan_parameter(sample, block)
            
            # Get DFT energy
            dft_energy = sample['E_dft']
            
            # Calculate ReaxFF energy
            energies = self.calculate_reaxff_energy(structure, block)
            reaxff_energy = energies['total']
            
            # Store data
            scan_data['scan_values'].append(scan_value)
            scan_data['dft_energies'].append(dft_energy)
            scan_data['reaxff_energies'].append(reaxff_energy)
            scan_data['structures'].append(structure)
            
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    if not scan_data['scan_values']:
        print("No valid data for plotting")
        return
    
    # Convert to numpy arrays
    scan_values = np.array(scan_data['scan_values'])
    dft_energies = np.array(scan_data['dft_energies'])
    reaxff_energies = np.array(scan_data['reaxff_energies'])
    
    # Sort by scan values for proper line plots
    sort_idx = np.argsort(scan_values)
    scan_values = scan_values[sort_idx]
    dft_energies = dft_energies[sort_idx]
    reaxff_energies = reaxff_energies[sort_idx]
    
    # Convert to kcal/mol for better readability
    dft_energies_kcal = convert_to_kcalmol(dft_energies)
    reaxff_energies_kcal = convert_to_kcalmol(reaxff_energies)
    
    # Determine molecule name based on system
    molecule_names = {
        'bond': f'MoS₂ - {block.upper()} scan',
        'angle': f'MoS₂ - {block.upper()} scan',
        'torsion': f'MoS₂ - {block.upper()} scan',
        'vdw_coulomb': f'MoS₂ - vdW scan'
    }
    molecule_name = molecule_names.get(block, 'MoS₂')
    
    # Create main comparison plot
    print(f"\nCreating {block} comparison plot...")
    plot_path = plotter.plot_energy_scan(
        scan_values=scan_values,
        dft_energies=dft_energies_kcal,
        reaxff_energies=reaxff_energies_kcal,
        scan_type=block,
        molecule_name=molecule_name,
        save_prefix='dft_comparison'
    )
    
    # Create parity plot
    print("Creating parity plot...")
    parity_path = plotter.create_parity_plot(
        dft_energies=dft_energies,
        reaxff_energies=reaxff_energies,
        save_prefix=f'parity_{block}'
    )
    
    # Calculate and save statistics
    errors = reaxff_energies - dft_energies
    rel_errors = errors / np.abs(dft_energies) * 100
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))
    r2 = np.corrcoef(dft_energies, reaxff_energies)[0, 1]**2
    
    # Save comprehensive data file
    comprehensive_df = pd.DataFrame({
        f'{block}_scan_value': scan_values,
        'DFT_energy_eV': dft_energies,
        'ReaxFF_energy_eV': reaxff_energies,
        'DFT_energy_kcal/mol': dft_energies_kcal,
        'ReaxFF_energy_kcal/mol': reaxff_energies_kcal,
        'Error_eV': errors,
        'Error_kcal/mol': convert_to_kcalmol(errors),
        'Relative_Error_%': rel_errors
    })
    
    # Save in multiple formats
    csv_path = os.path.join(output_dir, f'{block}_comprehensive_data.csv')
    comprehensive_df.to_csv(csv_path, index=False, float_format='%.6f')
    
    excel_path = os.path.join(output_dir, f'{block}_comprehensive_data.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        comprehensive_df.to_excel(writer, sheet_name='Data', index=False)
        
        # Add statistics sheet
        stats_df = pd.DataFrame({
            'Metric': ['RMSE (eV)', 'MAE (eV)', 'Max Error (eV)', 'R²',
                      'RMSE (kcal/mol)', 'MAE (kcal/mol)', 'Max Error (kcal/mol)',
                      'Min DFT Energy (eV)', 'Min ReaxFF Energy (eV)',
                      'Energy Range DFT (eV)', 'Energy Range ReaxFF (eV)'],
            'Value': [rmse, mae, max_error, r2,
                     convert_to_kcalmol(rmse), convert_to_kcalmol(mae), 
                     convert_to_kcalmol(max_error),
                     np.min(dft_energies), np.min(reaxff_energies),
                     np.ptp(dft_energies), np.ptp(reaxff_energies)]
        })
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
    
    # Create Origin-specific format with proper headers
    origin_path = os.path.join(output_dir, f'{block}_origin_format.txt')
    with open(origin_path, 'w') as f:
        # Write metadata
        f.write(f"# DFT vs ReaxFF Comparison Data\n")
        f.write(f"# System: {self.system}\n")
        f.write(f"# Block: {block}\n")
        f.write(f"# Date: {self.timestamp}\n")
        f.write(f"# Statistics:\n")
        f.write(f"#   RMSE: {rmse:.6f} eV ({convert_to_kcalmol(rmse):.3f} kcal/mol)\n")
        f.write(f"#   MAE: {mae:.6f} eV ({convert_to_kcalmol(mae):.3f} kcal/mol)\n")
        f.write(f"#   R²: {r2:.4f}\n")
        f.write("#" + "="*80 + "\n")
        
        # Write column headers
        if block == 'bond':
            f.write("# Bond_Length(Ang)\tDFT(eV)\tReaxFF(eV)\tDFT(kcal/mol)\tReaxFF(kcal/mol)\tError(eV)\n")
        elif block == 'angle':
            f.write("# Angle(deg)\tDFT(eV)\tReaxFF(eV)\tDFT(kcal/mol)\tReaxFF(kcal/mol)\tError(eV)\n")
        elif block == 'torsion':
            f.write("# Dihedral(deg)\tDFT(eV)\tReaxFF(eV)\tDFT(kcal/mol)\tReaxFF(kcal/mol)\tError(eV)\n")
        else:
            f.write("# Distance(Ang)\tDFT(eV)\tReaxFF(eV)\tDFT(kcal/mol)\tReaxFF(kcal/mol)\tError(eV)\n")
        
        # Write data
        for i in range(len(scan_values)):
            f.write(f"{scan_values[i]:.4f}\t")
            f.write(f"{dft_energies[i]:.6f}\t")
            f.write(f"{reaxff_energies[i]:.6f}\t")
            f.write(f"{dft_energies_kcal[i]:.3f}\t")
            f.write(f"{reaxff_energies_kcal[i]:.3f}\t")
            f.write(f"{errors[i]:.6f}\n")
    
    # Print summary
    print(f"\n{block.upper()} Comparison Results:")
    print(f"  Data points: {len(scan_values)}")
    print(f"  RMSE: {rmse:.6f} eV ({convert_to_kcalmol(rmse):.3f} kcal/mol)")
    print(f"  MAE: {mae:.6f} eV ({convert_to_kcalmol(mae):.3f} kcal/mol)")
    print(f"  Max Error: {max_error:.6f} eV ({convert_to_kcalmol(max_error):.3f} kcal/mol)")
    print(f"  R²: {r2:.4f}")
    
    print(f"\nFiles saved:")
    print(f"  Plot: {plot_path}")
    print(f"  Parity: {parity_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {excel_path}")
    print(f"  Origin: {origin_path}")
    
    # Return data for potential further use
    return {
        'scan_values': scan_values,
        'dft_energies': dft_energies,
        'reaxff_energies': reaxff_energies,
        'statistics': {
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'r2': r2
        }
    }
