import numpy as np
import pandas as pd
import csv
import argparse
import os
import glob
from tqdm import tqdm
import concurrent.futures
from collections import defaultdict

# --- R-Matrix Physics and Math Functions (Vectorized for grid calculations) ---

def kfunc(E, A):
    """Vectorized wave number k (in units of 1/sqrt(barn))."""
    E = np.asarray(E)
    k2 = np.zeros_like(E, dtype=float)
    mask = E > 0
    # Constant is approx (2 * m_n / hbar^2) in units of (1/barn)/eV
    k2[mask] = 2.1968e-3 * E[mask] * (A / (A + 1))
    return np.sqrt(k2)

def gfunc(I, J):
    """Statistical spin factor g_J."""
    return (2.0 * J + 1.0) / (2.0 * (2.0 * I + 1.0))

def p_factors_vec(E_vec, A, a, lmax):
    """
    Vectorized calculation of penetrability (P), shift factor (S), and phase shift (phi)
    for all l-waves up to lmax across an energy grid.
    """
    k_vec = kfunc(E_vec, A)
    # Convert channel radius from fm to sqrt(barn)
    a_barn = a * 0.1 
    rho_vec = k_vec * a_barn
    
    P_l = np.zeros((lmax + 1, len(E_vec)))
    S_l = np.zeros((lmax + 1, len(E_vec)))
    phi_l = np.zeros((lmax + 1, len(E_vec)))

    # Handle rho > 0 to avoid division by zero or log(0)
    mask = rho_vec > 1e-12

    # l=0 (s-wave)
    P_l[0, mask] = rho_vec[mask]
    S_l[0, mask] = 0.0
    phi_l[0, mask] = rho_vec[mask]

    # l=1 (p-wave)
    rho_sq = rho_vec[mask]**2
    P_l[1, mask] = rho_sq / (1.0 + rho_sq)
    S_l[1, mask] = -1.0 / (1.0 + rho_sq)
    phi_l[1, mask] = rho_vec[mask] - np.arctan(rho_vec[mask])

    # l=2 (d-wave)
    rho_4 = rho_sq**2
    P_l[2, mask] = rho_4 / (9.0 + 3.0 * rho_sq + rho_4)
    S_l[2, mask] = -(18.0 + 3.0 * rho_sq) / (9.0 + 3.0 * rho_sq + rho_4)
    phi_l[2, mask] = rho_vec[mask] - np.arctan(rho_vec[mask] * 3.0 / (3.0 - rho_sq))
    # Handle arctan discontinuity for d-waves
    phi_l[2, (rho_vec**2 > 3) & mask] += np.pi

#    # l=3 (f-wave)
#    rho_6 = rho_4 * rho_sq
#    P_l[3, mask] = rho_6 / (225.0 + 45.0 * rho_sq + 6.0 * rho_4 + rho_6)
#    S_l[3, mask] = -(675.0 + 90.0 * rho_sq + 6.0 * rho_4) / (225.0 + 45.0 * rho_sq + 6.0 * rho_4 + rho_6)
#    phi_l[3, mask] = rho_vec[mask] - np.arctan(rho_vec[mask] * (15.0 - rho_sq) / (15.0 - 6.0*rho_sq))

    return P_l, S_l, phi_l

def create_adaptive_grid(resonances, emin, emax, global_points, local_points_per_res, width_multiplier):
    """Creates an adaptive energy grid focused around resonance peaks."""
    grid = set(np.logspace(np.log10(emin), np.log10(emax), global_points))
    for res in resonances:
        # Use Gn (full width) for grid generation as it's more representative of the peak width
        E_r, Gn, Gg = res['E_r (eV)'], res['Gn (eV)'], res['Gg (eV)']
        G_total = Gn + Gg
        half_width = (G_total / 2.0) * width_multiplier
        local_grid = np.linspace(max(emin, E_r - half_width), min(emax, E_r + half_width), local_points_per_res)
        grid.update(local_grid)
    return np.sort(list(grid))

def process_file(filepath, output_dir, nuc, grid_params):
    """Processes a single ladder file to calculate cross sections using R-Matrix theory."""
    try:
        # Correctly read the CSV, skipping comments and using the right delimiter
        df = pd.read_csv(filepath, delimiter=';', comment='#')
        resonances = df.to_dict('records')
    except Exception as e:
        print(f"Could not process {filepath}: {e}")
        return None
    
    if not resonances:
        print(f"No data found in {filepath}. Skipping.")
        return None

    l_max = df['L'].max()
    E_grid = create_adaptive_grid(resonances, **grid_params)
    
    # Pre-calculate Penetrability, Shift Factor, and Phase Shift on the energy grid
    P_grid, S_grid, phi_grid = p_factors_vec(E_grid, nuc['A'], nuc['a'], l_max)
    k_vec_sq = kfunc(E_grid, nuc['A'])**2
    
    total_xs = np.zeros_like(E_grid)
    capture_xs = np.zeros_like(E_grid)
    
    # --- R-Matrix Calculation ---
    # Group resonances by their (J, L) channel for efficient lookup
    resonances_by_channel = defaultdict(list)
    for res in resonances:
        resonances_by_channel[(res['J'], res['L'])].append(res)
    
    unique_J_values = sorted(list(set(r['J'] for r in resonances)))

    # Main loop is over energy points, not resonances
    for i, E in enumerate(tqdm(E_grid, desc="R-Matrix Calculation", leave=False)):
        if E <= 0:
            continue
            
        s_tot_E_sum = 0.0
        s_cap_E_sum = 0.0

        # Sum contributions over all J-pi channels
        for J in unique_J_values:
            g_J = gfunc(nuc['I'], J)
            
            # For a given J, iterate over possible l-waves
            for l in range(l_max + 1):
                channel = (J, l)
                if channel not in resonances_by_channel:
                    continue

                R_jl = 0j 
                for res in resonances_by_channel[channel]:
                    # Use Gn0 for the R-Matrix calculation
                    R_jl += res['Gn0 (eV)'] / (res['E_r (eV)'] - E - 1j * res['Gg (eV)'] / 2.0)

                P = P_grid[l, i]
                S = S_grid[l, i]
                phi = phi_grid[l, i]
                
                L = S + 1j * P
                
                # Avoid division by zero if (1 - R_jl * L) is very small
                denom = 1 - R_jl * L
                if abs(denom) < 1e-12:
                    U_jl = -np.exp(-2j * phi)
                else:
                    U_jl = np.exp(-2j * phi) * (1 + (2j * P * R_jl) / denom)
                
                s_tot_E_sum += g_J * (2.0 - 2.0 * np.real(U_jl))
                s_cap_E_sum += g_J * (1.0 - np.abs(U_jl)**2)

        prefactor = np.pi / k_vec_sq[i] if k_vec_sq[i] > 0 else 0
        total_xs[i] = prefactor * s_tot_E_sum
        capture_xs[i] = prefactor * s_cap_E_sum

    # Save results
    output_filename = os.path.join(output_dir, os.path.basename(filepath))
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Energy (eV)', 'Total_XS (b)', 'Capture_XS (b)'])
        writer.writerows(zip(E_grid, total_xs, capture_xs))
        
    return output_filename

def process_file_wrapper(args):
    """Helper function to unpack arguments for process_file."""
    return process_file(*args)

def main():
    parser = argparse.ArgumentParser(description="Calculate 0K cross sections from sampled resonance ladders using R-Matrix theory.")
    parser.add_argument("ladder_dir", help="Directory containing resonance ladder CSV files.")
    parser.add_argument("output_dir", help="Directory to save the cross-section files.")
    parser.add_argument("--mass", type=float, required=True, help="Atomic mass of the target nucleus.")
    parser.add_argument("--radius", type=float, required=True, help="Channel radius in fm.")
    parser.add_argument("--spin", type=float, required=True, help="Ground state spin of the target nucleus.")
    parser.add_argument("--emin", type=float, default=1e-5, help="Minimum energy for the grid (eV).")
    parser.add_argument("--emax", type=float, default=2e7, help="Maximum energy for the grid (eV).")
    parser.add_argument("--global_points", type=int, default=2000, help="Number of points for the coarse background grid.")
    parser.add_argument("--local_points", type=int, default=10, help="Number of points to add around each resonance.")
    parser.add_argument("--width_multiplier", type=float, default=50.0, help="Multiplier for resonance width to determine local grid span.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    ladder_files = glob.glob(os.path.join(args.ladder_dir, '*.csv'))
    nuc = {'A': args.mass, 'I': args.spin, 'a': args.radius}
    
    grid_params = {
        'emin': args.emin, 
        'emax': args.emax,
        'global_points': args.global_points,
        'local_points_per_res': args.local_points,
        'width_multiplier': args.width_multiplier
    }

    tasks = [(f, args.output_dir, nuc, grid_params) for f in ladder_files]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file_wrapper, tasks), total=len(tasks), desc="Processing Ladder Files"))

    print(f"\nFinished processing. Cross sections saved in {args.output_dir}")

if __name__ == "__main__":
    main()
