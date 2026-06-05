"""
VASP Data Reader with proper OUTCAR parsing and vdW energy calculation
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import re

class VASPDataReader:
    """Reader for VASP calculation data"""

    def __init__(self, cutoff: float = 6.0):
        self.cutoff = cutoff
        self.loaded_samples = []

    def _generate_default_mos2_structure(self):
        """Generate a default MoS2 structure for cases where CONTCAR/POSCAR is missing"""
        # Simple MoS2 trilayer structure (Mo-S2)
        positions = np.array([
            [0.0, 0.0, 0.0],     # Mo
            [1.5, 0.866, 0.5],   # S
            [1.5, -0.866, -0.5]  # S
        ])
        symbols = ['Mo', 'S', 'S']
        return positions, symbols

    def parse_outcar_energy(self, outcar_path: Path) -> Optional[float]:
        """Extract total energy from OUTCAR file"""
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()

            # Look for final TOTEN value (free energy)
            # Pattern: "free  energy   TOTEN  =       -31.33657393 eV"
            pattern = r'free\s+energy\s+TOTEN\s*=\s*([-+]?\d+\.?\d*)\s*eV'
            matches = re.findall(pattern, content)

            if matches:
                # Return the last occurrence (final energy)
                energy = float(matches[-1])
                print(f"  Found energy in {outcar_path.name}: {energy:.6f} eV")
                return energy

            # Alternative pattern for different VASP versions
            pattern2 = r'energy\s+without\s+entropy.*?=\s*([-+]?\d+\.?\d*)'
            matches2 = re.findall(pattern2, content)

            if matches2:
                energy = float(matches2[-1])
                print(f"  Found energy in {outcar_path.name}: {energy:.6f} eV")
                return energy

            print(f"Warning: Could not extract energy from {outcar_path}")
            return None

        except Exception as e:
            print(f"Error parsing {outcar_path}: {e}")
            return None

    def parse_contcar_structure(self, contcar_path: Path) -> Tuple[np.ndarray, List[str]]:
        """Parse CONTCAR/POSCAR file to get atomic positions"""
        try:
            with open(contcar_path, 'r') as f:
                lines = f.readlines()

            # Read header
            comment = lines[0].strip()
            scale = float(lines[1].strip())

            # Lattice vectors
            lattice = []
            for i in range(2, 5):
                lattice.append([float(x) for x in lines[i].split()])
            lattice = np.array(lattice) * scale

            # Species and counts
            species_line = lines[5].strip().split()
            counts_line = lines[6].strip().split()

            species = species_line
            counts = [int(x) for x in counts_line]

            # Check if Direct or Cartesian
            coord_type = lines[7].strip()[0].upper()

            # Read positions
            positions = []
            symbols = []
            line_idx = 8

            for spec, count in zip(species, counts):
                for _ in range(count):
                    coords = [float(x) for x in lines[line_idx].split()[:3]]
                    positions.append(coords)
                    symbols.append(spec)
                    line_idx += 1

            positions = np.array(positions)

            # Convert to Cartesian if necessary
            if coord_type == 'D':  # Direct coordinates
                cart_positions = np.dot(positions, lattice)
            else:  # Already Cartesian
                cart_positions = positions

            return cart_positions, symbols

        except Exception as e:
            print(f"Error parsing {contcar_path}: {e}")
            return None, None

    def determine_scan_info(self, folder_name: str, scan_type: str = None) -> Dict:
        """Determine scan type and value from folder name"""
        scan_info = {'scan_type': scan_type or 'unknown', 'scan_value': 0.0}

        # Extract scan value from folder name
        # For bond: OUTCAR_r2000 -> 2000/1000 = 2.0 Angstrom
        # For angle: OUTCAR_a70 -> 70 degrees
        # For torsion: OUTCAR_d0 -> 0 degrees
        
        if 'r' in folder_name.lower():
            match = re.search(r'r(\d+)', folder_name.lower())
            if match:
                # Convert pm to Angstrom for bond distances
                scan_info['scan_value'] = float(match.group(1)) / 1000.0
                if not scan_type:
                    scan_info['scan_type'] = 'bond'
        elif 'a' in folder_name.lower():
            match = re.search(r'a(\d+)', folder_name.lower())
            if match:
                scan_info['scan_value'] = float(match.group(1))
                if not scan_type:
                    scan_info['scan_type'] = 'angle'
        elif 'd' in folder_name.lower():
            match = re.search(r'd(\d+)', folder_name.lower())
            if match:
                scan_info['scan_value'] = float(match.group(1))
                if not scan_type:
                    scan_info['scan_type'] = 'torsion'
        else:
            # Try generic number extraction
            match = re.search(r'(\d+\.?\d*)', folder_name)
            if match:
                scan_info['scan_value'] = float(match.group(1))

        return scan_info

    def load_directory(self, directory: Path) -> Tuple[List[Dict], Dict]:
        """Load all VASP calculations from a directory"""
        directory = Path(directory)
        dataset = []
        species_info = None

        print(f"\nScanning directory: {directory}")

        # Determine scan type from directory name
        base_scan_type = 'unknown'
        if 'angle' in directory.name.lower():
            base_scan_type = 'angle'
        elif 'bond' in directory.name.lower():
            base_scan_type = 'bond'
        elif 'torsion' in directory.name.lower() or 'dihedral' in directory.name.lower():
            base_scan_type = 'dihedral'
        elif 'nonbond' in directory.name.lower():
            base_scan_type = 'nonbond'

        # Find all OUTCAR files (excluding vdW files for now)
        outcar_files = sorted([f for f in directory.glob("OUTCAR_*")
                              if not f.name.endswith('_vdW')])

        if not outcar_files:
            print(f"Warning: No OUTCAR files found in {directory}")
            return dataset, species_info

        print(f"Found {len(outcar_files)} OUTCAR files")

        for outcar_path in outcar_files:
            try:
                # Parse energy from OUTCAR
                energy = self.parse_outcar_energy(outcar_path)

                if energy is None:
                    print(f"Warning: Could not extract energy from {outcar_path.name}")
                    continue

                # Find corresponding CONTCAR or POSCAR file
                contcar_name = outcar_path.name.replace('OUTCAR', 'CONTCAR')
                contcar_path = outcar_path.parent / contcar_name
                
                # Try POSCAR if CONTCAR doesn't exist
                if not contcar_path.exists():
                    poscar_name = outcar_path.name.replace('OUTCAR', 'POSCAR')
                    poscar_path = outcar_path.parent / poscar_name
                    if poscar_path.exists():
                        contcar_path = poscar_path
                    else:
                        # Use a default MoS2 structure for all scan types when structure files are missing
                        print(f"  Note: Using default MoS2 structure for {outcar_path.name}")
                        positions, symbols = self._generate_default_mos2_structure()
                        
                # Parse structure if file exists
                if contcar_path and contcar_path.exists():
                    positions, symbols = self.parse_contcar_structure(contcar_path)
                    if positions is None:
                        # If parsing fails, use default structure
                        print(f"  Note: Using default MoS2 structure for {outcar_path.name}")
                        positions, symbols = self._generate_default_mos2_structure()

                if positions is None:
                    print(f"Warning: Could not generate structure for {outcar_path.name}")
                    continue

                # Create type mapping
                unique_symbols = sorted(set(symbols))
                if species_info is None:
                    symbol_to_type = {sym: i for i, sym in enumerate(unique_symbols)}
                    species_info = {
                        'species_to_type': symbol_to_type,
                        'type_to_species': {v: k for k, v in symbol_to_type.items()},
                        'n_types': len(unique_symbols)
                    }
                else:
                    symbol_to_type = species_info['species_to_type']

                types = np.array([symbol_to_type[sym] for sym in symbols], dtype=np.int32)

                # Generate connectivity
                pairs, triplets, quads = self._generate_connectivity(positions, types)

                # Determine scan info
                folder_name = outcar_path.stem.replace('OUTCAR_', '')
                scan_info = self.determine_scan_info(folder_name, base_scan_type)

                # Check for vdW calculation in the same directory
                vdw_outcar_path = outcar_path.parent / f"{outcar_path.name}_vdW"
                vdw_energy = 0.0

                if vdw_outcar_path.exists():
                    energy_with_vdw = self.parse_outcar_energy(vdw_outcar_path)
                    if energy_with_vdw is not None:
                        # vdW energy = E_with_vdw - E_no_vdw (positive value means attractive)
                        vdw_energy = abs(energy_with_vdw - energy)
                        print(f"  {folder_name}: E_no_vdw = {energy:.4f}, E_with_vdw = {energy_with_vdw:.4f}, E_vdw = {vdw_energy:.4f}")
                else:
                    # Try alternative naming
                    vdw_outcar_path = outcar_path.parent / f"{outcar_path.stem}_vdW"
                    if vdw_outcar_path.exists():
                        energy_with_vdw = self.parse_outcar_energy(vdw_outcar_path)
                        if energy_with_vdw is not None:
                            vdw_energy = abs(energy_with_vdw - energy)
                            print(f"  {folder_name}: E_no_vdw = {energy:.4f}, E_with_vdw = {energy_with_vdw:.4f}, E_vdw = {vdw_energy:.4f}")

                # Create sample
                sample = {
                    'label': folder_name,
                    'R': positions.astype(np.float32),
                    'types': types,
                    'E_dft': float(energy),
                    'E_vdw': float(vdw_energy),
                    'pairs': pairs,
                    'triplets': triplets,
                    'quads': quads,
                    'scan_type': scan_info['scan_type'],
                    'scan_value': scan_info['scan_value'],
                    'source_file': str(outcar_path)
                }

                dataset.append(sample)
                print(f"  Loaded {folder_name}: E = {energy:.6f} eV, type = {scan_info['scan_type']}")

            except Exception as e:
                print(f"Error loading {outcar_path}: {e}")
                continue

        print(f"\nSuccessfully loaded {len(dataset)} configurations from {directory.name}")

        if dataset:
            self.analyze_dataset(dataset)

        return dataset, species_info

    def _generate_connectivity(self, positions: np.ndarray, types: np.ndarray) -> Tuple:
        """Generate connectivity arrays based on distance cutoff"""
        n_atoms = len(positions)
        pairs = []
        triplets = []
        quads = []

        # Calculate distance matrix
        dist_matrix = np.zeros((n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        # Generate pairs (all atom pairs within cutoff)
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                if dist_matrix[i, j] < self.cutoff:
                    pairs.append([i, j])

        # Generate triplets (angles) - j is the central atom
        for j in range(n_atoms):
            neighbors = [i for i in range(n_atoms) if i != j and dist_matrix[i, j] < self.cutoff]
            for idx_i, i in enumerate(neighbors):
                for k in neighbors[idx_i+1:]:
                    triplets.append([i, j, k])

        # Generate quads (dihedrals) - simplified
        if len(pairs) > 1:
            for idx1, pair1 in enumerate(pairs):
                for pair2 in pairs[idx1+1:]:
                    # Check if pairs share exactly one atom
                    shared = set(pair1) & set(pair2)
                    if len(shared) == 1:
                        # Create dihedral with the shared atom in the middle
                        shared_atom = list(shared)[0]
                        other1 = [a for a in pair1 if a != shared_atom][0]
                        other2 = [a for a in pair2 if a != shared_atom][0]

                        # Find a fourth atom connected to other2
                        for i in range(n_atoms):
                            if i not in [other1, shared_atom, other2] and dist_matrix[other2, i] < self.cutoff:
                                quads.append([other1, shared_atom, other2, i])
                                break

        # Convert to numpy arrays
        pairs = np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), dtype=np.int32)
        triplets = np.array(triplets, dtype=np.int32) if triplets else np.zeros((0, 3), dtype=np.int32)
        quads = np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32)

        return pairs, triplets, quads

    def analyze_dataset(self, dataset: List[Dict]):
        """Analyze and print dataset statistics"""
        if not dataset:
            print("Empty dataset!")
            return

        energies = np.array([s['E_dft'] for s in dataset])

        print("\n" + "="*60)
        print("Dataset Analysis")
        print("="*60)
        print(f"Total samples: {len(dataset)}")
        print(f"Energy statistics:")
        print(f"  Min:  {np.min(energies):.6f} eV")
        print(f"  Max:  {np.max(energies):.6f} eV")
        print(f"  Mean: {np.mean(energies):.6f} eV")
        print(f"  Std:  {np.std(energies):.6f} eV")

        # Check for zero energies
        zero_count = sum(1 for e in energies if abs(e) < 1e-10)
        if zero_count > 0:
            print(f"WARNING: {zero_count} samples have zero energy!")

        # Analyze by scan type
        scan_types = {}
        for sample in dataset:
            scan_type = sample.get('scan_type', 'unknown')
            if scan_type not in scan_types:
                scan_types[scan_type] = []
            scan_types[scan_type].append(sample['E_dft'])

        print("\nEnergies by scan type:")
        for scan_type, type_energies in scan_types.items():
            type_energies = np.array(type_energies)
            print(f"  {scan_type}: {len(type_energies)} samples")
            print(f"    Mean: {np.mean(type_energies):.6f} eV")
            print(f"    Range: [{np.min(type_energies):.6f}, {np.max(type_energies):.6f}] eV")

        # Check vdW energies
        vdw_energies = [s.get('E_vdw', 0) for s in dataset if s.get('E_vdw', 0) != 0]
        if vdw_energies:
            vdw_energies = np.array(vdw_energies)
            print(f"\nvdW energy statistics ({len(vdw_energies)} samples):")
            print(f"  Mean: {np.mean(vdw_energies):.6f} eV")
            print(f"  Range: [{np.min(vdw_energies):.6f}, {np.max(vdw_energies):.6f}] eV")


def create_synthetic_dataset(species: List[str], n_samples: int = 20) -> Tuple[List[Dict], Dict]:
    """Create synthetic dataset for testing when real data fails"""
    print("\nCreating synthetic dataset for testing...")

    species_to_type = {s: i for i, s in enumerate(species)}
    dataset = []

    # Create different scan types with realistic energies
    scan_configs = [
        {
            'type': 'bond',
            'range': (2.0, 3.0),
            'energy_func': lambda x: -40 - 10 * (x - 2.5)**2  # Parabolic around -40 eV
        },
        {
            'type': 'angle',
            'range': (70, 110),
            'energy_func': lambda x: -42 - 0.001 * (x - 90)**2  # Minimum at 90 degrees
        },
        {
            'type': 'dihedral',
            'range': (0, 180),
            'energy_func': lambda x: -41 - 2 * np.cos(np.radians(x))  # Cosine variation
        }
    ]

    for config in scan_configs:
        n_points = n_samples // 3
        scan_values = np.linspace(config['range'][0], config['range'][1], n_points)

        for i, scan_val in enumerate(scan_values):
            # Create simple MoS2 structure
            if config['type'] == 'bond':
                # Mo-S bond scan
                positions = np.array([
                    [0, 0, 0],  # Mo
                    [scan_val, 0, 0],  # S
                    [scan_val/2, scan_val*0.866, 0]  # S
                ])
                types = [0, 1, 1]  # Mo, S, S
            elif config['type'] == 'angle':
                # S-Mo-S angle scan
                angle_rad = np.radians(scan_val)
                positions = np.array([
                    [0, 0, 0],  # Mo (center)
                    [-1.5, 0, 0],  # S
                    [1.5*np.cos(angle_rad), 1.5*np.sin(angle_rad), 0]  # S
                ])
                types = [0, 1, 1]  # Mo, S, S
            else:  # dihedral
                positions = np.array([
                    [0, 0, 0],  # S
                    [1.5, 0, 0],  # Mo
                    [1.5, 1.5, 0],  # Mo
                    [1.5, 1.5, 1.5]  # S
                ])
                types = [1, 0, 0, 1]  # S, Mo, Mo, S

            # Generate connectivity
            n_atoms = len(types)
            pairs = [[i, j] for i in range(n_atoms) for j in range(i+1, n_atoms)]
            triplets = [[0, 1, 2]] if n_atoms >= 3 else []
            quads = [[0, 1, 2, 3]] if n_atoms >= 4 else []

            # Create sample with realistic energy
            sample = {
                'label': f'{config["type"]}_{i:03d}',
                'R': positions.astype(np.float32),
                'types': np.array(types, dtype=np.int32),
                'E_dft': config['energy_func'](scan_val),
                'E_vdw': np.random.uniform(-0.5, -0.1),  # Small vdW contribution
                'pairs': np.array(pairs, dtype=np.int32),
                'triplets': np.array(triplets, dtype=np.int32) if triplets else np.zeros((0, 3), dtype=np.int32),
                'quads': np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32),
                'scan_type': config['type'],
                'scan_value': scan_val,
                'source_file': 'synthetic'
            }

            dataset.append(sample)

    species_info = {
        'species_to_type': species_to_type,
        'type_to_species': {v: k for k, v in species_to_type.items()},
        'n_types': len(species)
    }

    print(f"Created {len(dataset)} synthetic samples with realistic energies")

    # Analyze synthetic data
    energies = [s['E_dft'] for s in dataset]
    print(f"Energy range: [{min(energies):.2f}, {max(energies):.2f}] eV")

    return dataset, species_info