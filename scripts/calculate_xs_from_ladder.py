import numpy as np
import csv
import argparse
import os
import glob
from tqdm import tqdm
import concurrent.futures

# --- R-Matrix Functions (Vectorized) ---

def kfunc(E, A):
    """Vectorized wave number k"""
    # Using np.asarray to handle both single values and arrays gracefully
    E = np.asarray(E)
    k2 = np.zeros_like(E, dtype=float)
    # Create a mask for positive energies to avoid division by zero or sqrt of negative
    mask = E > 0
    # Neutron mass in eV/c^2, reduced Planck constant in eV*s, speed of light in m/s
    # This combination simplifies to a constant for calculating k^2
    # Constant derived from (2 * m_n / hbar^2)
    # k^2 = 2 * mu * E / hbar^2, where mu is the reduced mass m_n * A / (A+1)
    # The constant 2.1968e-3 is approximately (2 * m_n / hbar^2) in units of (1/barn)/eV
    k2[mask] = 2.1968e-3 * E[mask] * (A / (A + 1))
    return np.sqrt(k2)

def gfunc(I, J):
    """Statistical spin factor"""
    return (2.0 * J + 1.0) / (2.0 * (2.0 * I + 1.0))

def p_factors_vec(E_vec, A, a, lmax):
    """Vectorized calculation of penetrability, shift factor, and phase shift"""
    rho = kfunc(E_vec, A) * a
    n_points = len(E_vec)
    
    P = np.zeros((n_points, lmax + 1))
    S = np.zeros((n_points, lmax + 1))
    phi = np.zeros((n_points, lmax + 1))
    
    # l = 0 (s-wave)
    mask_rho = np.abs(rho) > 0
    P[mask_rho, 0] = rho[mask_rho]
    phi[mask_rho, 0] = rho[mask_rho]
    
    # l > 0 using recursion relations
    # Note: Using real numbers as these factors are real for positive energy
    for l_val in range(1, lmax + 1):
        prev_l = l_val - 1
        rho2 = rho**2
        # Denominator in recursion relations
        D = S[:, prev_l]**2 + P[:, prev_l]**2
        mask_D = np.abs(D) > 1e-12 # Avoid division by zero

        P[mask_D, l_val] = P[mask_D, prev_l] * rho2[mask_D] / D[mask_D]
        S[mask_D, l_val] = S[mask_D, prev_l] * rho2[mask_D] / D[mask_D]
        
        phi[:, l_val] = phi[:, prev_l] - np.arctan2(P[:, prev_l], S[:, prev_l] - prev_l)

    return P, S, phi

def R_m_vec(E_vec, ladder, nuc, APLs, nJ_channels, nl_channels):
    """Vectorized R-Matrix construction"""
    n_points = len(E_vec)
    R = np.zeros((n_points, nJ_channels, nl_channels), dtype=complex)

    # Pre-calculate reduced width amplitudes (gamma) for each resonance once
    for res in ladder:
        # Penetrability at the resonance energy is needed for the reduced width
        P_res, _, _ = p_factors(res['E0'], nuc['A'], APLs[res['l']], res['l'])
        penetrability = P_res[res['l']]
        # Gn = 2 * P * gamma^2 => gamma^2 = Gn / (2 * P)
        res['gamma_sq'] = res['Gn'] / (2.0 * penetrability) if penetrability > 1e-12 else 0.0

    # Sum the contribution of each resonance to the R-Matrix
    for res in ladder:
        if res['gamma_sq'] == 0:
            continue
        
        E0, l, Gg, j_idx = res['E0'], res['l'], res['Gg'], res['J_idx']
        
        # SLBW approximation for the R-matrix contribution of a single level
        denominator = (E0 - E_vec) - 1j * Gg / 2.0
        R[:, j_idx, l] += res['gamma_sq'] / denominator
        
    return R

def U_mat_vec(E_vec, l, j_idx, nuc, R_vec, APLs):
    """Vectorized U-matrix (collision matrix) calculation"""
    P, S, phi = p_factors_vec(E_vec, nuc['A'], APLs[l], l)
    
    L = (S[:, l] - l) + 1j * P[:, l]
    RR = R_vec[:, j_idx, l]
    phi_l = phi[:, l]
    
    denominator = (1.0 - RR * L)
    
    # Handle cases where the denominator is zero to avoid errors
    U = np.exp(-2.0j * phi_l) # Default for when denominator is zero
    mask = np.abs(denominator) >= 1e-12
    U[mask] = np.exp(-2.0j * phi_l[mask]) * (1.0 - RR[mask] * np.conj(L[mask])) / denominator[mask]
    
    return U

def sigma_vec(E_vec, ladder, nuc, APLs, J_map, g_factors):
    """Vectorized 0K cross section calculation"""
    nl_channels = len(APLs)
    nJ_channels = len(J_map)
    n_points = len(E_vec)
    
    R_vec = R_m_vec(E_vec, ladder, nuc, APLs, nJ_channels, nl_channels)
    
    s_t = np.zeros(n_points, dtype=float)
    s_g = np.zeros(n_points, dtype=float)
    
    active_channels = set((res['l'], res['J']) for res in ladder)

    for l in range(nl_channels):
        for j_val, j_idx in J_map.items():
            if (l, j_val) in active_channels:
                U_vec = U_mat_vec(E_vec, l, j_idx, nuc, R_vec, APLs)
                g = g_factors[(l, j_val)]
                s_t += g * (1.0 - np.real(U_vec))
                s_g += g * (1.0 - np.abs(U_vec)**2)

    k2_vec = kfunc(E_vec, nuc['A'])**2
    
    # Cross section formula: sigma = (pi / k^2) * g * T, where T is the transmission coefficient
    # Total XS: T = 2 * (1 - Re(U))
    # Capture XS: T = 1 - |U|^2
    # The factor of 100 converts from fm^2 to barns
    prefactor = np.zeros(n_points, dtype=float)
    mask = np.abs(k2_vec.real) > 1e-12
    prefactor[mask] = np.pi / k2_vec[mask].real * 100.0 # Factor of 100 for fm^2 -> barns
    
    total_xs = 2.0 * prefactor * s_t
    capture_xs = prefactor * s_g
    
    return total_xs, capture_xs

# --- Non-vectorized p_factors for single energy points (used for pre-calculation) ---
def p_factors(E, A, a, lmax):
    rho = kfunc(np.array([E]), A)[0] * a
    P, S, phi = [np.zeros(lmax + 1) for _ in range(3)]
    if abs(rho) == 0: return P, S, phi
    P[0], phi[0] = rho, rho
    for i in range(1, lmax + 1):
        l = float(i - 1)
        D = (S[i-1] - l)**2 + P[i-1]**2
        if abs(D) < 1e-12: continue
        P[i] = P[i-1] * rho**2 / D
        S[i] = S[i-1] * rho**2 / D
        phi[i] = phi[i-1] - np.arctan2(P[i-1], S[i-1] - l)
    return P, S, phi

# --- Main Script Logic ---

def generate_adaptive_grid(ladder, emin, emax, global_points, local_points_per_res, width_multiplier):
    """
    Generates an adaptive energy grid that is dense around resonances and coarse elsewhere.
    
    Args:
        ladder (list): The list of resonance dictionaries.
        emin (float): The minimum energy of the grid.
        emax (float): The maximum energy of the grid.
        global_points (int): Number of points for the coarse logarithmic background grid.
        local_points_per_res (int): Number of points to add around each resonance.
        width_multiplier (float): Determines the span of the local grid around a resonance,
                                  as a multiple of its total width (Gn + Gg).
    
    Returns:
        np.ndarray: A sorted array of unique energy points.
    """
    # 1. Create the coarse, global background grid
    background_grid = np.logspace(np.log10(emin), np.log10(emax), global_points)
    
    all_points = [background_grid]
    
    # 2. Create fine grids and add exact resonance energies
    for res in ladder:
        E0 = res['E0']
        total_width = res['Gn'] + res['Gg']
        
        if E0 > emax or E0 < emin:
            continue

        # Ensure the exact resonance energy is included, as requested!
        all_points.append(np.array([E0]))
            
        if total_width <= 0:
            continue
            
        # Define the window for the fine grid around the resonance energy
        window = width_multiplier * total_width
        start_E = max(emin, E0 - window)
        end_E = min(emax, E0 + window)
        
        if start_E >= end_E:
            continue
            
        local_grid = np.linspace(start_E, end_E, local_points_per_res)
        all_points.append(local_grid)
        
    # 3. Combine all points, filter to be strictly within bounds, sort, and remove duplicates
    final_grid = np.concatenate(all_points)
    final_grid = final_grid[(final_grid >= emin) & (final_grid <= emax)]
    final_grid = np.unique(final_grid) # np.unique also sorts the array
    
    print(f"Generated adaptive grid with {len(final_grid)} points for {os.getpid()}.")
    return final_grid

def read_ladder_file(filepath):
    """Reads a single resonance ladder file."""
    ladder = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or "E_r" in line:
                continue
            try:
                # Use replace to handle both comma and dot decimals and strip spaces
                line = line.strip().replace(',', '.')
                E0, l, j, gn, gg = line.split(';')
                ladder.append({'E0': float(E0), 'l': int(l), 'J': float(j), 'Gn': float(gn), 'Gg': float(gg)})
            except (ValueError, IndexError):
                print(f"Skipping malformed row in {filepath}: {line.strip()}")
    return ladder

def process_ladder_file(task_args):
    """Worker function to process a single ladder file."""
    ladder_file, output_dir, nuc, grid_params = task_args
    
    ladder = read_ladder_file(ladder_file)
    if not ladder:
        return f"Skipped {ladder_file} (no data)."

    # Generate the adaptive grid inside the worker for this specific ladder
    energy_grid = generate_adaptive_grid(
        ladder, 
        grid_params['emin'], 
        grid_params['emax'],
        global_points=grid_params['global_points'],
        local_points_per_res=grid_params['local_points_per_res'],
        width_multiplier=grid_params['width_multiplier']
    )

    l_values = sorted(list(set(res['l'] for res in ladder)))
    j_values = sorted(list(set(res['J'] for res in ladder)))
    j_map = {j: i for i, j in enumerate(j_values)}
    
    g_factors = {(l, j): gfunc(nuc['I'], j) for l in l_values for j in j_values}

    for res in ladder:
        res['J_idx'] = j_map[res['J']]

    APLs = [nuc['a']] * (max(l_values) + 1)

    total_xs_vec, capture_xs_vec = sigma_vec(energy_grid, ladder, nuc, APLs, j_map, g_factors)
    
    base_name = os.path.basename(ladder_file)
    output_filename = os.path.join(output_dir, base_name.replace('ladder', 'xs'))
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Energy (eV)', 'Total_XS (b)', 'Capture_XS (b)'])
        rows = np.stack((energy_grid, total_xs_vec, capture_xs_vec), axis=1)
        writer.writerows(rows)
            
    return output_filename

def main():
    parser = argparse.ArgumentParser(description="Calculate 0K cross sections from resonance ladders using an adaptive energy grid.")
    parser.add_argument("ladder_dir", help="Directory containing the sampled ladder CSV files.")
    parser.add_argument("output_dir", help="Directory to save the cross-section files.")
    parser.add_argument("--mass", type=float, required=True, help="Atomic mass of the target nucleus (in amu).")
    parser.add_argument("--radius", type=float, required=True, help="Channel radius in fm.")
    parser.add_argument("--spin", type=float, required=True, help="Ground state spin of the target nucleus.")
    parser.add_argument("--emin", type=float, default=1e-5, help="Minimum energy for the grid (eV).")
    parser.add_argument("--emax", type=float, default=2e7, help="Maximum energy for the grid (eV).")
    # Arguments for the adaptive grid generation
    parser.add_argument("--global_points", type=int, default=2000, help="Number of points for the coarse background grid.")
    parser.add_argument("--local_points", type=int, default=100, help="Number of points to add around each resonance.")
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
        results = list(tqdm(executor.map(process_ladder_file, tasks), total=len(tasks), desc="Calculating Cross Sections"))

    print(f"\nProcessing complete. {len(results)} cross-section files written to {args.output_dir}.")

if __name__ == '__main__':
    main()
