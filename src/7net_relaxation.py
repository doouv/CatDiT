import os
import re
import json
import argparse
from datetime import datetime
from ase.io import read, write as ase_write
from ase.optimize import LBFGS
from ase.constraints import FixAtoms
import numpy as np
import pandas as pd
from tqdm import tqdm

# Gas phase reference energies (computed by SevenNet-Omni, modal=oc20)
# chemical potential (C) = E(CO) - E(H2O) + E(H2)
# chemical potential (H) = 0.5*E(H2)
# chemical potential (N) = 0.5*E(N2)
# chemical potential (O) = E(H2O) - E(H2)
GAS_REFERENCE_ENERGIES = {
    'H': -3.476299285888672,
    'C': -7.193307876586914,
    'N': -8.07209300994873,
    'O': -7.197885513305664,
}


def parse_inspect_indices(value):
    """Parse indices from one of:
      - CSV file path with an 'idx' column (e.g., 7net_NRR_screened_722.csv)
      - comma/space separated indices or ranges (e.g., '0-100,200-300')
    """
    if os.path.isfile(value) and value.endswith('.csv'):
        df = pd.read_csv(value)
        if 'idx' not in df.columns:
            raise ValueError(f"CSV {value} has no 'idx' column (cols: {list(df.columns)})")
        return df['idx'].astype(int).tolist()
    indices = []
    for part in value.replace(',', ' ').split():
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            start, end = part.split('-', 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description='Slab Separation + MLIP Relaxation Pipeline')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to run directory (e.g., /workspace/logs/generate_samples/runs/generate_samples_2025-12-29_07-14-43/)')
    parser.add_argument('--fmax', type=float, default=0.05,
                        help='Force convergence criterion (eV/Å)')
    parser.add_argument('--steps', type=int, default=500,
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



def process_single_sample(catalyst_atoms, calc, fmax=0.05, steps=500,
                          save_dir=None, idx=None):
    """Process a single catalyst structure: split in memory, relax, compute E_ads.
    Mirrors UMA relaxation pipeline exactly.

    Args:
        catalyst_atoms: ASE Atoms object with tags (0=subsurface, 1=surface, 2=adsorbate)
        calc: ASE calculator
        fmax: force convergence criterion
        steps: max optimization steps
        save_dir: if given, write relaxed system/slab as extxyz here (filename uses idx).
        idx: integer index used in saved filenames (system_{idx}.extxyz / slab_{idx}.extxyz).

    Returns:
        dict with all results
    """
    import contextlib
    import io as _io

    result = {
        'initial_e_ads': None, 'relaxed_e_ads': None,
        'e_sys_unrelaxed': None, 'e_sys_relaxed': None,
        'e_slab_relaxed': None, 'e_adsorbate': None,
        'converged_system': False, 'converged_slab': False,
        'steps_system': 0, 'steps_slab': 0, 'error': None,
    }

    try:
        catalyst_atoms.center()

        # Split in memory (same as UMA/CatFlow)
        system = catalyst_atoms.copy()
        if 0 in system.get_tags():
            system.set_constraint(FixAtoms(
                indices=[atom.index for atom in system if atom.tag == 0]
            ))

        slab = catalyst_atoms.copy()[catalyst_atoms.get_tags() != 2]
        if 0 in slab.get_tags():
            slab.set_constraint(FixAtoms(
                indices=[atom.index for atom in slab if atom.tag == 0]
            ))

        adsorbate = catalyst_atoms.copy()[catalyst_atoms.get_tags() == 2]
        if len(adsorbate) == 0:
            result['error'] = "No adsorbate atoms (tag=2)"
            return result

        # Adsorbate energy from table (no relaxation)
        e_adsorbate = 0.0
        for symbol in adsorbate.get_chemical_symbols():
            if symbol in GAS_REFERENCE_ENERGIES:
                e_adsorbate += GAS_REFERENCE_ENERGIES[symbol]
            else:
                result['error'] = f"No reference energy for {symbol}"
                return result
        result['e_adsorbate'] = e_adsorbate

        # Assign calculator
        system.calc = calc
        slab.calc = calc

        # Unrelaxed system energy
        e_sys_unrelaxed = system.get_potential_energy()
        result['e_sys_unrelaxed'] = e_sys_unrelaxed

        # Relax system
        opt_sys = LBFGS(system, logfile=None)
        with contextlib.redirect_stdout(_io.StringIO()):
            converged_system = opt_sys.run(fmax, steps)
        steps_system = opt_sys.get_number_of_steps()
        e_sys_relaxed = system.get_potential_energy()

        result['e_sys_relaxed'] = e_sys_relaxed
        result['converged_system'] = converged_system
        result['steps_system'] = steps_system

        if save_dir is not None and idx is not None:
            ase_write(os.path.join(save_dir, f'system_{idx}.extxyz'),
                      system, format='extxyz')

        # Relax slab
        opt_slab = LBFGS(slab, logfile=None)
        with contextlib.redirect_stdout(_io.StringIO()):
            converged_slab = opt_slab.run(fmax, steps)
        steps_slab = opt_slab.get_number_of_steps()
        e_slab_relaxed = slab.get_potential_energy()

        result['e_slab_relaxed'] = e_slab_relaxed
        result['converged_slab'] = converged_slab
        result['steps_slab'] = steps_slab

        if save_dir is not None and idx is not None:
            ase_write(os.path.join(save_dir, f'slab_{idx}.extxyz'),
                      slab, format='extxyz')

        # E_ads = E_system - E_slab - E_adsorbate
        result['initial_e_ads'] = e_sys_unrelaxed - (e_slab_relaxed + e_adsorbate)
        result['relaxed_e_ads'] = e_sys_relaxed - (e_slab_relaxed + e_adsorbate)

    except Exception as e:
        result['error'] = str(e)

    return result


def process_directory(args):
    """Main processing function - mirrors UMA relaxation pipeline with 7net OC20/OC22 modal."""
    base_path = args.path
    generated_dir = os.path.join(base_path, 'generated')
    mlip_dir = os.path.join(base_path, 'mlip')

    # Check generated directory
    if not os.path.exists(generated_dir):
        raise FileNotFoundError(f"Generated directory not found: {generated_dir}")

    os.makedirs(mlip_dir, exist_ok=True)
    relaxed_dir = os.path.join(mlip_dir, 'relaxed')
    os.makedirs(relaxed_dir, exist_ok=True)

    # Find catalyst files
    catalyst_files = sorted(
        [f for f in os.listdir(generated_dir)
         if f.endswith(".extxyz")],
        key=lambda x: int(re.search(r'(\d+)', x).group())
    )
    if not catalyst_files:
        raise ValueError(f"No catalyst_*.extxyz files found in {generated_dir}")

    # Build index -> filename mapping
    file_map = {}
    for f in catalyst_files:
        match = re.search(r'catalyst_(\d+)\.extxyz', f)
        if match:
            file_map[int(match.group(1))] = f

    indices = sorted(file_map.keys())

    # Inspection mode filter
    inspect_mode = args.inspect is not None
    inspect_idx_set = set()
    if inspect_mode:
        inspect_idx_set = set(args.inspect)
        indices = [i for i in indices if i in inspect_idx_set]
        if not indices:
            raise ValueError(f"None of the requested indices found in data")

    # Load calculators (both OC20 and OC22)
    calculators = get_calculators(args.device)

    print(f"Processing {len(indices)} samples (fmax={args.fmax}, max_steps={args.steps})")
    print("=" * 60)

    # Process each sample
    all_results = []
    for idx in tqdm(indices, desc="Relaxation"):
        filepath = os.path.join(generated_dir, file_map[idx])
        try:
            atoms = read(filepath)
        except Exception as e:
            all_results.append({
                'idx': idx, 'struct_type': 'unknown',
                'initial_e_ads': None, 'relaxed_e_ads': None,
                'e_sys_unrelaxed': None, 'e_sys_relaxed': None,
                'e_slab_relaxed': None, 'e_adsorbate': None,
                'converged_system': False, 'converged_slab': False,
                'steps_system': 0, 'steps_slab': 0, 'error': f"Read error: {e}",
            })
            continue

        # Select calculator based on oxide detection
        oxide_flag, oxygen_ratio = is_oxide(atoms)
        struct_type = 'oc22' if oxide_flag else 'oc20'
        calc = calculators[struct_type]

        result = process_single_sample(atoms, calc, args.fmax, args.steps,
                                       save_dir=relaxed_dir, idx=idx)
        result['idx'] = idx
        result['struct_type'] = struct_type
        all_results.append(result)

    # Build DataFrame
    df = pd.DataFrame(all_results).sort_values('idx').reset_index(drop=True)

    # Rename columns for output CSV (match CatFlow/UMA naming)
    df_out = df.rename(columns={
        'initial_e_ads': 'E_ads_initial',
        'relaxed_e_ads': 'E_ads',
        'e_sys_unrelaxed': 'E_sys_unrelaxed',
        'e_sys_relaxed': 'E_sys_relaxed',
        'e_slab_relaxed': 'E_slab',
        'e_adsorbate': 'E_gas',
    })

    # Add combined convergence column
    df_out['converged'] = df_out['converged_system'] & df_out['converged_slab']

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_inspect" if inspect_mode else ""
    output_file = os.path.join(mlip_dir, f'7net_results_{timestamp}{suffix}.csv')
    df_out.to_csv(output_file, index=False)

    #==========================================================================
    # Summary
    #==========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total = len(df_out)
    valid_mask = df_out['E_ads'].notna() & (df_out['E_ads'] != 999.0)
    conv_mask = df_out['converged'] & valid_mask
    error_count = int(df_out['error'].notna().sum())

    n_oc20 = int((df_out['struct_type'] == 'oc20').sum())
    n_oc22 = int((df_out['struct_type'] == 'oc22').sum())

    print(f"Total samples: {total}")
    print(f"  - OC20 (metal alloy): {n_oc20} ({100*n_oc20/total:.1f}%)")
    print(f"  - OC22 (oxide):       {n_oc22} ({100*n_oc22/total:.1f}%)")
    print(f"Errors: {error_count}")
    print(f"System converged: {int(df_out['converged_system'].sum())}/{total}")
    print(f"Slab converged: {int(df_out['converged_slab'].sum())}/{total}")
    print(f"Both converged: {int(df_out['converged'].sum())}/{total} "
          f"({100*int(df_out['converged'].sum())/total:.1f}%)")

    summary = {
        'fmax': args.fmax,
        'max_steps': args.steps,
        'total_samples': total,
        'n_oc20': n_oc20,
        'n_oc22': n_oc22,
        'error_count': error_count,
        'both_converged': int(df_out['converged'].sum()),
        'convergence_rate': round(100 * int(df_out['converged'].sum()) / total, 2) if total > 0 else 0,
    }

    if valid_mask.any():
        valid = df_out[valid_mask]
        print(f"\nE_ads (all valid, N={len(valid)}):")
        print(f"  Mean: {valid['E_ads'].mean():.3f} eV")
        print(f"  Std:  {valid['E_ads'].std():.3f} eV")
        summary['E_ads_mean'] = round(float(valid['E_ads'].mean()), 3)
        summary['E_ads_std'] = round(float(valid['E_ads'].std()), 3)

    if conv_mask.any():
        conv_valid = df_out[conv_mask]
        print(f"\nE_ads (converged only, N={len(conv_valid)}):")
        print(f"  Mean: {conv_valid['E_ads'].mean():.3f} eV")
        print(f"  Std:  {conv_valid['E_ads'].std():.3f} eV")
        print(f"\nAvg steps (system): {conv_valid['steps_system'].mean():.1f}")
        print(f"Avg steps (slab): {conv_valid['steps_slab'].mean():.1f}")
        summary['E_ads_mean_converged'] = round(float(conv_valid['E_ads'].mean()), 3)
        summary['E_ads_std_converged'] = round(float(conv_valid['E_ads'].std()), 3)
        summary['avg_steps_system'] = round(float(conv_valid['steps_system'].mean()), 1)
        summary['avg_steps_slab'] = round(float(conv_valid['steps_slab'].mean()), 1)

    summary_file = os.path.join(mlip_dir, f'7net_summary_{timestamp}{suffix}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

    return df_out


if __name__ == '__main__':
    args = parse_args()
    process_directory(args)
