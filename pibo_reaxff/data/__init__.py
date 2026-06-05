"""
Data processing and I/O modules for PIBO
"""

from .reader import VASPDataReader
from .preprocessor import DataPreprocessor

# Data loading functions
def load_dft_data(data_dir):
    """
    Load all DFT reference data from directory
    
    Args:
        data_dir: Path to vasp_calculations directory
        
    Returns:
        Dictionary with DFT data organized by energy type
    """
    import os
    
    reader = VASPDataReader()
    dft_data = {}
    
    # Define subdirectories and their corresponding energy types
    subdirs = {
        'bond': 'bond',
        'angle': 'angle', 
        'torsion': 'torsion',
        'nonbond': 'vdw'
    }
    
    for subdir, energy_type in subdirs.items():
        dir_path = os.path.join(data_dir, subdir)
        if os.path.exists(dir_path):
            data, metadata = reader.load_directory(dir_path)
            dft_data[energy_type] = {
                'data': data,
                'metadata': metadata
            }
            
    return dft_data


def prepare_training_data(dft_data, energy_type):
    """
    Prepare training data for a specific energy type
    
    Args:
        dft_data: Dictionary with DFT data
        energy_type: Type of energy ('bond', 'angle', etc.)
        
    Returns:
        structures: List of atomic structures
        energies: Corresponding DFT energies
    """
    if energy_type not in dft_data:
        return [], []
        
    data = dft_data[energy_type]['data']
    
    structures = []
    energies = []
    
    for sample in data:
        structures.append(sample.get('structure'))
        energies.append(sample.get('E_dft', 0.0))
        
    return structures, energies


def save_optimization_results(results, output_dir):
    """
    Save optimization results in multiple formats
    
    Args:
        results: Dictionary with optimization results
        output_dir: Output directory path
    """
    import json
    import pandas as pd
    
    # Save as JSON
    json_file = os.path.join(output_dir, 'results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save parameter history as CSV
    if 'parameter_history' in results:
        df = pd.DataFrame(results['parameter_history'])
        csv_file = os.path.join(output_dir, 'parameter_history.csv')
        df.to_csv(csv_file, index=False)
    
    # Save energy history
    if 'energy_history' in results:
        df = pd.DataFrame(results['energy_history'])
        csv_file = os.path.join(output_dir, 'energy_history.csv')
        df.to_csv(csv_file, index=False)


__all__ = [
    'VASPDataReader',
    'DataPreprocessor',
    'load_dft_data',
    'prepare_training_data',
    'save_optimization_results'
]

# Module metadata
__version__ = "2.0.0"
__description__ = "Data processing and I/O for PIBO"
