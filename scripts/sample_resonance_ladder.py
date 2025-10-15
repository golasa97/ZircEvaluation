import numpy as np
import sys
import os
import csv
import argparse
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

# Add the project root to the Python path to allow importing PySesh
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if project_root not in sys.path:
#    sys.path.insert(0, project_root)
import PySesh

# --- Physics functions needed to calculate Gn0 from Gn ---

def kfunc(E, A):
    """Vectorized wave number k (in units of 1/sqrt(barn))."""
    E = np.asarray(E)
    k2 = np.zeros_like(E, dtype=float)
    mask = E > 0
    k2[mask] = 2.1968e-3 * E[mask] * (A / (A + 1))
    return np.sqrt(k2)

def p_factors_vec(E_vec, A, a, lmax):
    """
    Vectorized calculation of penetrability (P) for all l-waves up to lmax.
    """
    k_vec = kfunc(E_vec, A)
    a_barn = a * 0.1
    rho_vec = k_vec * a_barn
    
    P_l = np.zeros((lmax + 1, len(E_vec)))
    mask = rho_vec > 1e-12

    # l=0
    P_l[0, mask] = rho_vec[mask]
    if lmax > 0:
        # l=1
        rho_sq = rho_vec[mask]**2
        P_l[1, mask] = rho_sq / (1.0 + rho_sq)
    if lmax > 1:
        # l=2
        rho_4 = rho_sq**2
        P_l[2, mask] = rho_4 / (9.0 + 3.0 * rho_sq + rho_4)
    if lmax > 2:
        # l=3
        rho_6 = rho_4 * rho_sq
        P_l[3, mask] = rho_6 / (225.0 + 45.0 * rho_sq + 6.0 * rho_4 + rho_6)
    return P_l

def default_channel_factory():
    """Provides a default dictionary structure for a new (L, J) channel."""
    return {'E': [], 'D_J': [], 'Gn0': [], 'Gg': [], 'AMUN': 0}



def p_factor_single(E, A, a, l):
    """Calculates penetrability for a single energy and l-wave."""
    if E <= 1e-9: # Avoid sqrt of zero or negative
        return 0.0
    
    # Constant is approx (2 * m_n / hbar^2) in units of (1/barn)/eV
    k2 = 2.1968e-3 * E * (A / (A + 1))
    k = np.sqrt(k2)
    
    # Convert channel radius from fm to sqrt(barn)
    a_barn = a * 0.1
    rho = k * a_barn

    if rho < 1e-9:
        return 0.0

    if l == 0:
        return rho
    elif l == 1:
        rho2 = rho**2
        return rho2 / (1.0 + rho2)
    elif l == 2:
        rho2 = rho**2
        rho4 = rho2**2
        return rho4 / (9.0 + 3.0 * rho2 + rho4)
    elif l == 3:
        rho2 = rho**2
        rho4 = rho2**2
        rho6 = rho4*rho2
        return rho6 / (225.0 + 45.0 * rho2 + 6.0 * rho4 + rho6)
    
    return 0.0 # For l > 3, not implemented

def sample_statistical_ladder(E_grid, D_J_grid, Gn0_grid, Gg_grid, l, j, nu_Gn, nu_Gg, A, a):
    """
    Generates a statistical resonance ladder for a single (L, J) channel
    by interpolating the mean parameters on the provided energy grid.
    """
    ladder = []
    
    # Ensure grids are numpy arrays for interpolation
    E_grid = np.asarray(E_grid)
    D_J_grid = np.asarray(D_J_grid)
    Gn0_grid = np.asarray(Gn0_grid)
    Gg_grid = np.asarray(Gg_grid)

    E_min, E_max = E_grid[0], E_grid[-1]
    
    # Start sampling with a random step from the beginning of the energy range
    D_J_start = np.interp(E_min, E_grid, D_J_grid)
    current_E = E_min - (D_J_start * np.log(np.random.rand()))

    while current_E < E_max:
        # 1. Interpolate the average level spacing at the current energy
        # Use left=D_J_grid[0] to handle cases where current_E is slightly below E_min
        avg_D_at_E = np.interp(current_E, E_grid, D_J_grid, left=D_J_grid[0])
        if avg_D_at_E <= 0: # Safety check for invalid spacing
            break

        # 2. Sample the next spacing from a Wigner distribution (using exponential approximation)
        spacing =1.12837 * avg_D_at_E *np.sqrt(-np.log(np.random.rand()))
        current_E += spacing
        
        if current_E >= E_max:
            break
            
        # 3. Interpolate the average widths at the new resonance energy
        avg_Gn0_at_E = np.interp(current_E, E_grid, Gn0_grid)
        avg_Gg_at_E = np.interp(current_E, E_grid, Gg_grid)

        # 4. Sample the actual widths from chi-squared distributions (Porter-Thomas is nu=1)
        Gn0_sample = avg_Gn0_at_E * np.random.chisquare(nu_Gn) / nu_Gn if avg_Gn0_at_E > 0 else 0.0
        Gg_sample = avg_Gg_at_E 
        
        # 5. Calculate the full neutron width Gn from the reduced width Gn0
        penetrability = p_factor_single(current_E, A, a, l)
        Gn_sample = Gn0_sample * np.sqrt(current_E) * penetrability
        
        ladder.append({
            'E': current_E,
            'Gn': Gn_sample,
            'Gg': Gg_sample,
            'L': l,
            'J': j
        })
        
    return ladder




def read_urr_parameters(filepath):
    """Reads the URR parameter CSV file into a dictionary keyed by (L, J)."""
    params_by_channel = defaultdict(default_channel_factory)
    ground_state_spin = None
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        
        for i, row in enumerate(reader):
            try:
                if i == 0:
                    ground_state_spin = float(row[idx['I']])
                
                l = int(float(row[idx['L']]))
                j = float(row[idx['J']])
                channel = (l, j)
                
                params_by_channel[channel]['E'].append(float(row[idx['E']]))
                params_by_channel[channel]['D_J'].append(float(row[idx['D_J']]))
                params_by_channel[channel]['Gn0'].append(float(row[idx['Gn0']]))
                params_by_channel[channel]['Gg'].append(float(row[idx['Gg']]))
                params_by_channel[channel]['AMUN'] = int(float(row[idx['AMUN']]))
                
            except (ValueError, KeyError, IndexError) as e:
                print(f"Skipping malformed row {i+2} in {filepath}: {row}. Error: {e}")
                continue
    return params_by_channel, ground_state_spin


def generate_and_write_ladder(task_args):
    """Samples one complete resonance ladder and writes it to a file."""
    params_by_channel, mass, radius, output_filename, isotope_name, spin = task_args
    
    full_ladder = []
    
    max_l = max(l for l, j in params_by_channel.keys())

    for (l, j), params in params_by_channel.items():
        if not params['E']:
            continue
        
        ladder = sample_statistical_ladder(
            params['E'], params['D_J'], params['Gn0'], params['Gg'],
            l, j, params['AMUN'], 1, # NDOF_gg is always 1
            mass, radius # Pass mass and radius for penetrability calculation
        )

        full_ladder.extend(ladder)
    
    # Sort the combined ladder by energy
    full_ladder.sort(key=lambda x: x['E'])
    
    # Calculate Gn0 for each resonance
    energies = np.array([res['E'] for res in full_ladder])
    penetrabilities = p_factors_vec(energies, mass, radius, max_l)
    
    for i, res in enumerate(full_ladder):
        l = res['L']
        E_r = res['E']
        P_l_at_E = penetrabilities[l, i]
        
        if E_r > 0 and P_l_at_E > 1e-9:
             # PySesh returns Gn, convert back to Gn0 for R-Matrix use
            res['Gn0'] = res['Gn'] / (np.sqrt(E_r) * P_l_at_E)
        else:
            res['Gn0'] = 0.0

    # Write the ladder to a CSV file
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        f.write(f"# Isotope: {isotope_name}\n")
        f.write(f"# Ground State Spin (I): {spin}\n")
        f.write(f"# Atomic Mass (A): {mass}\n")
        f.write(f"# Channel Radius (a): {radius}\n")
        f.write(f"# Ladder sampled from {min(energies):.2f} eV to {max(energies):.2f} eV\n")
        
        header = ['E_r (eV)', 'L', 'J', 'Gn (eV)', 'Gg (eV)', 'Gn0 (eV)']
        writer.writerow(header)
        for res in full_ladder:
            writer.writerow([res['E'], res['L'], res['J'], res['Gn'], res['Gg'], res['Gn0']])
            
    return output_filename

def main():
    parser = argparse.ArgumentParser(description="Sample complete resonance ladders in parallel from a URR parameter file.")
    parser.add_argument("param_file", help="Path to the URR parameter CSV file.")
    parser.add_argument("--mass", type=float, required=True, help="Atomic mass of the target nucleus.")
    parser.add_argument("--radius", type=float, required=True, help="Channel radius in fm.")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to save the ladder files.")
    parser.add_argument("-n", "--num_ladders", type=int, default=1, help="Number of resonance ladders to generate.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    params_by_channel, spin = read_urr_parameters(args.param_file)
    
    base_name = os.path.basename(args.param_file)
    isotope_name = base_name.split('_')[0]

    tasks = []
    for i in range(args.num_ladders):
        output_filename = os.path.join(args.output_dir, f"{isotope_name}_ladder_{i+1:03d}.csv")
        tasks.append((params_by_channel, args.mass, args.radius, output_filename, isotope_name, spin))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(generate_and_write_ladder, tasks), total=len(tasks), desc=f"Sampling {isotope_name} Ladders"))

    print(f"\nFinished sampling. Ladders saved in {args.output_dir}")

if __name__ == "__main__":
    main()
