import os
import re
import json
import argparse
from datetime import datetime
from ase.io import read, write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import numpy as np
import pandas as pd
from tqdm import tqdm

# Gas phase reference energies (from CatBench / OC20)
# Linear combination of H2O, N2, CO, H2
GAS_REFERENCE_ENERGIES = {
    'H': -3.477,   # eV per atom
    'C': -7.282,
    'N': -8.083,
    'O': -7.204,
}


def parse_inspect_indices(value):
    """Parse comma or space separated indices into a list of integers."""
    # Handle comma-separated values (with optional spaces)
    indices = []
    for part in value.replace(',', ' ').split():
        indices.append(int(part.strip()))
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='Slab Separation + MLIP Relaxation Pipeline')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to run directory (e.g., /workspace/logs/generate_samples/runs/generate_samples_2025-12-29_07-14-43/)')
    parser.add_argument('--fmax', type=float, default=0.05,
                        help='Force convergence criterion (eV/Å)')
    parser.add_argument('--steps', type=int, default=400,
                        help='Maximum optimization steps')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for calculation (cuda/cpu)')
    parser.add_argument('--inspect', type=str, nargs='+', default=None,
                        help='Inspection mode: process only specified indices (e.g., --inspect 0,5,10 or --inspect 0 5 10)')

    args = parser.parse_args()

    # Parse inspect indices if provided
    if args.inspect is not None:
        all_indices = []
        for item in args.inspect:
            all_indices.extend(parse_inspect_indices(item))
        args.inspect = all_indices

    return args


def extract_slab(atoms):
    """Remove adsorbate (tag=2) from structure to get slab only."""
    tags = atoms.get_tags()
    slab_idx = np.where(tags != 2)[0]

    if len(slab_idx) == len(atoms):
        print("  Warning: No adsorbate (tag=2) found in structure")
        return atoms.copy()

    slab = atoms[slab_idx]
    return slab


def is_oxide(atoms):
    """
    Check if structure is oxide based on bulk composition.
    Oxide = any oxygen in bulk (tag != 2).

    Returns:
        is_oxide (bool): True if oxide (OC22), False if non-oxide (OC20)
        oxygen_ratio (float): Fraction of oxygen in bulk atoms
    """
    tags = atoms.get_tags()
    symbols = np.array(atoms.get_chemical_symbols())

    # Bulk atoms only (tag=0: subsurface, tag=1: surface), exclude adsorbate (tag=2)
    bulk_idx = np.where(tags != 2)[0]

    if len(bulk_idx) == 0:
        return False, 0.0

    bulk_symbols = symbols[bulk_idx]

    # Count oxygen in bulk
    n_oxygen = np.sum(bulk_symbols == 'O')
    n_total = len(bulk_symbols)

    oxygen_ratio = n_oxygen / n_total if n_total > 0 else 0.0

    # Oxide if any oxygen in bulk
    return oxygen_ratio > 0, oxygen_ratio


def run_slab_separation(generated_dir, slabs_dir):
    """Extract slabs from catalyst files. Returns True if separation was performed."""
    # Find catalyst files
    catalyst_files = sorted(
        [f for f in os.listdir(generated_dir) if f.startswith("catalyst_") and f.endswith(".extxyz")],
        key=lambda x: int(re.search(r'(\d+)', x).group())
    )

    if not catalyst_files:
        raise ValueError(f"No catalyst_*.extxyz files found in {generated_dir}")

    # Check if slabs already exist with same count
    if os.path.exists(slabs_dir):
        existing_slabs = [f for f in os.listdir(slabs_dir) if f.startswith("slab_") and f.endswith(".extxyz")]
        if len(existing_slabs) == len(catalyst_files):
            print(f"[SKIP] Slab separation: {len(existing_slabs)} slabs already exist (matches {len(catalyst_files)} catalysts)")
            return False

    os.makedirs(slabs_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("STEP 0: Slab Separation")
    print("="*60)
    print(f"Found {len(catalyst_files)} catalyst files")
    print(f"Input:  {generated_dir}")
    print(f"Output: {slabs_dir}")

    # Process each file
    success_count = 0
    for filename in tqdm(catalyst_files, desc="Extracting slabs"):
        try:
            # Read catalyst structure
            filepath = os.path.join(generated_dir, filename)
            atoms = read(filepath)

            # Extract slab (remove tag=2)
            slab = extract_slab(atoms)

            # Generate output filename: catalyst_n.extxyz -> slab_n.extxyz
            match = re.search(r'catalyst_(\d+)\.extxyz', filename)
            if match:
                n = match.group(1)
                output_filename = f"slab_{n}.extxyz"
            else:
                output_filename = filename.replace("catalyst_", "slab_")

            # Save slab
            output_path = os.path.join(slabs_dir, output_filename)
            write(output_path, slab, format='extxyz')
            success_count += 1

        except Exception as e:
            print(f"\nError processing {filename}: {e}")

    print(f"Completed: {success_count}/{len(catalyst_files)} files")
    return True


def load_calculator(device='cuda', modal='oc20'):
    """Load SevenNet-Omni calculator with specified modal.

    Args:
        device: 'cuda' or 'cpu'
        modal: 'oc20' for metal alloy catalysts, 'oc22' for oxide catalysts
    """
    from sevenn.calculator import SevenNetCalculator
    calc = SevenNetCalculator(
        model="7net-omni",
        modal=modal,
        enable_cueq=False,
        enable_flash=False
    )
    return calc


def get_calculators(device='cuda'):
    """Load both OC20 and OC22 calculators."""
    print("Loading SevenNet-Omni calculators...")
    print("  - Loading modal=oc20 (metal alloy)...")
    calc_oc20 = load_calculator(device=device, modal='oc20')
    print("  - Loading modal=oc22 (oxide)...")
    calc_oc22 = load_calculator(device=device, modal='oc22')
    print("  Done.")
    return {'oc20': calc_oc20, 'oc22': calc_oc22}


def get_gas_reference_energy(atoms):
    """Calculate gas phase reference energy from atomic composition of adsorbate (tag=2)"""
    tags = atoms.get_tags()
    adsorbate_idx = np.where(tags == 2)[0]

    if len(adsorbate_idx) == 0:
        return 0.0

    adsorbate_atoms = atoms[adsorbate_idx]
    symbols = adsorbate_atoms.get_chemical_symbols()

    E_gas = 0.0
    for symbol in symbols:
        if symbol in GAS_REFERENCE_ENERGIES:
            E_gas += GAS_REFERENCE_ENERGIES[symbol]
        else:
            print(f"  Warning: No reference energy for {symbol}")

    return E_gas


class TrajectoryCollector:
    """Collect trajectory during optimization"""
    def __init__(self, atoms, interval=1):
        self.atoms = atoms
        self.interval = interval
        self.step = 0
        self.trajectory = []

    def __call__(self):
        if self.step % self.interval == 0:
            atoms_copy = self.atoms.copy()
            atoms_copy.info['step'] = self.step
            atoms_copy.info['energy'] = self.atoms.get_potential_energy()
            atoms_copy.info['max_force'] = np.max(np.linalg.norm(self.atoms.get_forces(), axis=1))
            self.trajectory.append(atoms_copy)
        self.step += 1

    def get_trajectory(self):
        return self.trajectory


def relax_structure(atoms, calc, fmax=0.05, steps=400, fix_subsurface=True, save_traj=False):
    """Relax structure with optional subsurface fixing"""
    atoms = atoms.copy()

    if fix_subsurface:
        tags = atoms.get_tags()
        subsurface_idx = np.where(tags == 0)[0]
        if len(subsurface_idx) > 0:
            atoms.set_constraint(FixAtoms(indices=subsurface_idx))

    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)

    traj_collector = None
    if save_traj:
        traj_collector = TrajectoryCollector(atoms, interval=1)
        opt.attach(traj_collector)

    converged = opt.run(fmax=fmax, steps=steps)

    # Get final force
    forces = atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))

    trajectory = traj_collector.get_trajectory() if traj_collector else None
    return atoms, converged, max_force, trajectory


def process_directory(args):
    """Main processing function - slab separation + 3 step relaxation"""
    base_path = args.path
    generated_dir = os.path.join(base_path, 'generated')
    slabs_dir = os.path.join(base_path, 'slabs')
    mlip_dir = os.path.join(base_path, 'mlip')

    # Check generated directory
    if not os.path.exists(generated_dir):
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    # Run slab separation (will skip if already done)
    run_slab_separation(generated_dir, slabs_dir)

    # Verify slabs directory exists after separation
    if not os.path.exists(slabs_dir):
        raise FileNotFoundError(f"Slabs directory not found: {slabs_dir}")

    # Create output directory
    os.makedirs(mlip_dir, exist_ok=True)

    # Inspection mode settings
    inspect_mode = args.inspect is not None
    inspect_idx_set = set(args.inspect) if args.inspect else set()

    if inspect_mode:
        traj_dir = os.path.join(mlip_dir, 'trajectories')
        os.makedirs(traj_dir, exist_ok=True)
        print(f"[INSPECTION MODE] Processing indices: {args.inspect}")
        print(f"[INSPECTION MODE] Trajectories will be saved to: {traj_dir}/")

    # Load calculators (both OC20 and OC22)
    calculators = get_calculators(args.device)

    # Get file lists
    adslab_files = sorted(
        [f for f in os.listdir(generated_dir) if f.endswith('.extxyz')],
        key=lambda x: int(re.search(r'(\d+)', x).group())
    )
    slab_files = sorted(
        [f for f in os.listdir(slabs_dir) if f.endswith('.extxyz')],
        key=lambda x: int(re.search(r'(\d+)', x).group())
    )

    print(f"Found {len(adslab_files)} adslab files")
    print(f"Found {len(slab_files)} slab files")

    # Build index mapping
    # catalyst_X.extxyz -> idx X
    # slab_X.extxyz -> idx X
    adslab_map = {}  # idx -> filename
    for f in adslab_files:
        match = re.search(r'(\d+)', f)
        if match:
            adslab_map[match.group()] = f

    slab_map = {}  # idx -> filename
    for f in slab_files:
        match = re.search(r'(\d+)', f)
        if match:
            slab_map[match.group()] = f

    # Find common indices
    common_indices = sorted(set(adslab_map.keys()) & set(slab_map.keys()), key=int)

    # In inspection mode, filter to only requested indices
    if inspect_mode:
        common_indices = [idx for idx in common_indices if int(idx) in inspect_idx_set]
        if not common_indices:
            raise ValueError(f"None of the requested indices {args.inspect} found in data")

    print(f"Processing {len(common_indices)} pairs")

    # Results storage
    adslab_results = {}  # idx -> {E_total, max_force, converged, error}
    slab_results = {}    # idx -> {E_slab, converged, error}
    gas_energies = {}    # idx -> E_gas
    structure_types = {}  # idx -> 'oc20' or 'oc22'

    #==========================================================================
    # STEP 1: Relax all adslabs and get total energy
    #==========================================================================
    print("\n" + "="*60)
    print("STEP 1: Adslab relaxation")
    print("="*60)

    for idx in tqdm(common_indices, desc="Adslab relaxation"):
        adslab_file = adslab_map[idx]
        try:
            adslab = read(os.path.join(generated_dir, adslab_file))

            # Detect oxide/non-oxide and select appropriate calculator
            oxide_flag, oxygen_ratio = is_oxide(adslab)
            struct_type = 'oc22' if oxide_flag else 'oc20'
            structure_types[idx] = struct_type
            calc = calculators[struct_type]

            # Get gas reference energy (before relaxation)
            E_gas = get_gas_reference_energy(adslab)
            gas_energies[idx] = E_gas

            # Relax (save trajectory in inspection mode)
            relaxed_adslab, converged, max_force, traj = relax_structure(
                adslab, calc, fmax=args.fmax, steps=args.steps,
                fix_subsurface=True, save_traj=inspect_mode
            )
            E_total = relaxed_adslab.get_potential_energy()

            # Save trajectory in inspection mode
            if inspect_mode and traj:
                traj_file = os.path.join(traj_dir, f"traj_adslab_{idx}.extxyz")
                write(traj_file, traj, format='extxyz')

            adslab_results[idx] = {
                'E_total': E_total,
                'max_force': max_force,
                'converged': converged,
                'error': None
            }

            if inspect_mode:
                print(f"\n  idx={idx}: type={struct_type} (O={oxygen_ratio:.1%}), E_total={E_total:.3f} eV, converged={converged}")

        except Exception as e:
            adslab_results[idx] = {
                'E_total': None,
                'max_force': None,
                'converged': False,
                'error': str(e)
            }
            gas_energies[idx] = None
            structure_types[idx] = 'unknown'
            if inspect_mode:
                print(f"\n  idx={idx}: Error - {e}")

    #==========================================================================
    # STEP 2: Relax all slabs and get slab energy
    #==========================================================================
    print("\n" + "="*60)
    print("STEP 2: Slab relaxation")
    print("="*60)

    for idx in tqdm(common_indices, desc="Slab relaxation"):
        slab_file = slab_map[idx]
        try:
            slab = read(os.path.join(slabs_dir, slab_file))

            # Use same calculator type as adslab (already determined in STEP 1)
            struct_type = structure_types.get(idx, 'oc20')
            calc = calculators[struct_type]

            # Relax (save trajectory in inspection mode)
            relaxed_slab, converged, max_force, traj = relax_structure(
                slab, calc, fmax=args.fmax, steps=args.steps,
                fix_subsurface=True, save_traj=inspect_mode
            )
            E_slab = relaxed_slab.get_potential_energy()

            # Save trajectory in inspection mode
            if inspect_mode and traj:
                traj_file = os.path.join(traj_dir, f"traj_slab_{idx}.extxyz")
                write(traj_file, traj, format='extxyz')

            slab_results[idx] = {
                'E_slab': E_slab,
                'max_force': max_force,
                'converged': converged,
                'error': None
            }

            if inspect_mode:
                print(f"\n  idx={idx}: type={struct_type}, E_slab={E_slab:.3f} eV, converged={converged}")

        except Exception as e:
            slab_results[idx] = {
                'E_slab': None,
                'max_force': None,
                'converged': False,
                'error': str(e)
            }
            if inspect_mode:
                print(f"\n  idx={idx}: Error - {e}")

    #==========================================================================
    # STEP 3: Calculate adsorption energy and compile results
    #==========================================================================
    print("\n" + "="*60)
    print("STEP 3: Calculate adsorption energy")
    print("="*60)

    results = []
    for idx in common_indices:
        adslab_res = adslab_results.get(idx, {})
        slab_res = slab_results.get(idx, {})
        E_gas = gas_energies.get(idx)

        E_total = adslab_res.get('E_total')
        E_slab = slab_res.get('E_slab')

        # Calculate adsorption energy if all components are available
        if E_total is not None and E_slab is not None and E_gas is not None:
            E_ads = E_total - E_slab - E_gas
        else:
            E_ads = None

        # Combine errors
        errors = []
        if adslab_res.get('error'):
            errors.append(f"adslab: {adslab_res['error']}")
        if slab_res.get('error'):
            errors.append(f"slab: {slab_res['error']}")
        error_str = "; ".join(errors) if errors else None

        adslab_conv = adslab_res.get('converged', False)
        slab_conv = slab_res.get('converged', False)
        both_conv = adslab_conv and slab_conv

        results.append({
            'idx': int(idx),
            'struct_type': structure_types.get(idx, 'unknown'),
            'E_total': E_total,
            'E_slab': E_slab,
            'E_gas': E_gas,
            'E_ads': E_ads,
            'adslab_max_force': adslab_res.get('max_force'),
            'slab_max_force': slab_res.get('max_force'),
            'adslab_converged': adslab_conv,
            'slab_converged': slab_conv,
            'converged': both_conv,
            'error': error_str
        })

    # Save results with timestamp and mode indicator
    df = pd.DataFrame(results)
    df = df.sort_values('idx').reset_index(drop=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if inspect_mode:
        output_file = os.path.join(mlip_dir, f'results_{timestamp}_inspect.csv')
    else:
        output_file = os.path.join(mlip_dir, f'results_{timestamp}.csv')
    df.to_csv(output_file, index=False)

    #==========================================================================
    # Summary
    #==========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    total = len(df)
    adslab_success = int(df['E_total'].notna().sum())
    slab_success = int(df['E_slab'].notna().sum())
    ads_success = int(df['E_ads'].notna().sum())

    adslab_converged_count = int(df['adslab_converged'].sum())
    slab_converged_count = int(df['slab_converged'].sum())
    both_converged_count = int(df['converged'].sum())

    # Find unconverged indices
    unconverged_idx = df[~df['converged']]['idx'].tolist()

    # Count structure types
    n_oc20 = int((df['struct_type'] == 'oc20').sum())
    n_oc22 = int((df['struct_type'] == 'oc22').sum())

    print(f"Total pairs: {total}")
    print(f"  - OC20 (metal alloy): {n_oc20} ({100*n_oc20/total:.1f}%)")
    print(f"  - OC22 (oxide):     {n_oc22} ({100*n_oc22/total:.1f}%)")
    print(f"Adslab relaxation success: {adslab_success}/{total}")
    print(f"Slab relaxation success: {slab_success}/{total}")
    print(f"Adsorption energy calculated: {ads_success}/{total}")
    print(f"Adslab convergence rate: {adslab_converged_count}/{total} ({100*adslab_converged_count/total:.1f}%)")
    print(f"Slab convergence rate: {slab_converged_count}/{total} ({100*slab_converged_count/total:.1f}%)")
    print(f"Both convergence rate: {both_converged_count}/{total} ({100*both_converged_count/total:.1f}%)")

    if unconverged_idx:
        print(f"\nUnconverged indices ({len(unconverged_idx)}): {unconverged_idx}")

    # Build summary dict
    summary = {
        'total_pairs': total,
        'n_oc20': n_oc20,
        'n_oc22': n_oc22,
        'oc20_ratio': round(100 * n_oc20 / total, 2) if total > 0 else 0,
        'oc22_ratio': round(100 * n_oc22 / total, 2) if total > 0 else 0,
        'adslab_success': adslab_success,
        'slab_success': slab_success,
        'ads_success': ads_success,
        'adslab_converged': adslab_converged_count,
        'slab_converged': slab_converged_count,
        'both_converged': both_converged_count,
        'adslab_convergence_rate': round(100 * adslab_converged_count / total, 2),
        'slab_convergence_rate': round(100 * slab_converged_count / total, 2),
        'both_convergence_rate': round(100 * both_converged_count / total, 2),
        'unconverged_indices': unconverged_idx,
    }

    if ads_success > 0:
        valid_df = df[df['E_ads'].notna()]
        print(f"\nTotal Energy (E_total):")
        print(f"  Mean: {valid_df['E_total'].mean():.3f} eV")
        print(f"  Std:  {valid_df['E_total'].std():.3f} eV")
        print(f"\nSlab Energy (E_slab):")
        print(f"  Mean: {valid_df['E_slab'].mean():.3f} eV")
        print(f"  Std:  {valid_df['E_slab'].std():.3f} eV")
        print(f"\nAdsorption Energy (E_ads = E_total - E_slab - E_gas):")
        print(f"  Mean: {valid_df['E_ads'].mean():.3f} eV")
        print(f"  Std:  {valid_df['E_ads'].std():.3f} eV")
        print(f"\nAdslab Max Force:")
        print(f"  Mean: {valid_df['adslab_max_force'].mean():.4f} eV/Å")

        # Add to summary (convert to Python float for JSON serialization)
        summary['E_total_mean'] = round(float(valid_df['E_total'].mean()), 3)
        summary['E_total_std'] = round(float(valid_df['E_total'].std()), 3)
        summary['E_slab_mean'] = round(float(valid_df['E_slab'].mean()), 3)
        summary['E_slab_std'] = round(float(valid_df['E_slab'].std()), 3)
        summary['E_ads_mean'] = round(float(valid_df['E_ads'].mean()), 3)
        summary['E_ads_std'] = round(float(valid_df['E_ads'].std()), 3)
        summary['adslab_max_force_mean'] = round(float(valid_df['adslab_max_force'].mean()), 4)

    # Save summary to JSON with same timestamp as CSV
    mode_suffix = "_inspect" if inspect_mode else ""
    summary_filename = f"summary_{timestamp}{mode_suffix}.json"
    summary_file = os.path.join(mlip_dir, summary_filename)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

    if inspect_mode:
        print(f"Trajectories saved to: {traj_dir}/")

    return df


if __name__ == '__main__':
    args = parse_args()
    process_directory(args)
