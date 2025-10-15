import numpy as np
import pandas as pd
import argparse
from scipy.interpolate import interp1d
from scipy.integrate import quad
import concurrent.futures
from tqdm import tqdm
import os

# Boltzmann constant in eV/K
KB_EV = 8.617333262145e-5

def integrand(E_in, E_out, sigma_interp, alpha):
    """
    The integrand for the Doppler broadening integral (Solbrig kernel).
    
    Args:
        E_in (float): The 0K energy variable of integration (eV).
        E_out (float): The target energy for the broadened cross section (eV).
        sigma_interp (callable): Interpolation function for the 0K cross section.
        alpha (float): The Doppler broadening parameter, A / (kB * T).
        
    Returns:
        float: The value of the integrand at E_in.
    """
    if E_in <= 0:
        return 0.0
    
    # Get 0K cross section at the integration energy
    s_E_in = sigma_interp(E_in)
    
    # If cross section is zero, integrand is zero
    if s_E_in == 0:
        return 0.0
    
    sqrt_E_in = np.sqrt(E_in)
    sqrt_E_out = np.sqrt(E_out)
    
    # Calculate the two exponential terms of the Solbrig kernel
    # The kernel is extremely peaked, so these exponents become large and negative
    # very quickly as E_in moves away from E_out.
    term1 = -alpha * (sqrt_E_in - sqrt_E_out)**2
    term2 = -alpha * (sqrt_E_in + sqrt_E_out)**2
    
    val = np.sqrt(E_in) * s_E_in * (np.exp(term1) - np.exp(term2))
    return val

def broaden_point(E_out, sigma_interp, alpha, E_grid_min, E_grid_max):
    """
    Calculates the Doppler broadened cross section at a single energy point.
    
    Args:
        E_out (float): The target energy point (eV).
        sigma_interp (callable): Interpolation function for the 0K cross section.
        alpha (float): The Doppler broadening parameter, A / (kB * T).
        E_grid_min (float): Minimum energy of the original 0K data grid (eV).
        E_grid_max (float): Maximum energy of the original 0K data grid (eV).
        
    Returns:
        float: The Doppler broadened cross section at E_out.
    """
    if E_out <= 0:
        return 0.0
    
    # The Solbrig kernel is a sharply peaked function around E_out.
    # Integrating over the entire grid (E_grid_min to E_grid_max) is inefficient
    # and can lead to numerical errors (like returning zero) if the integrator
    # misses the narrow peak. We define a narrow window where the kernel is non-zero.
    
    # The kernel is Gaussian-like in sqrt(E) space. Its width is related to 1/sqrt(alpha).
    # A window of +/- 20 / sqrt(alpha) is more than enough to capture the peak.
    sqrt_E_out = np.sqrt(E_out)
    sqrt_E_window_half_width = 20.0 / np.sqrt(alpha)

    # Define the integration bounds in sqrt(E) space, then convert to energy
    sqrt_E_int_min = max(0.0, sqrt_E_out - sqrt_E_window_half_width)
    sqrt_E_int_max = sqrt_E_out + sqrt_E_window_half_width
    E_int_min = sqrt_E_int_min**2
    E_int_max = sqrt_E_int_max**2

    # The integration range must still be within the bounds of the original 0K data.
    E_int_min = max(E_int_min, E_grid_min)
    E_int_max = min(E_int_max, E_grid_max)

    if E_int_min >= E_int_max:
        return 0.0

    # Perform the integration over the narrowed window.
    # We also give the integrator a hint to pay attention to the peak at E_out.
    integral_val, _ = quad(
        integrand, 
        E_int_min, 
        E_int_max, 
        args=(E_out, sigma_interp, alpha),
        limit=1000,
        points=[E_out] if E_int_min < E_out < E_int_max else []
    )
    
    prefactor = (1.0 / (2.0 * E_out)) * np.sqrt(alpha / np.pi)
    result = prefactor * integral_val

    # Clamp result to be non-negative to avoid small negative values from numerical error.
    return max(0.0, result)

def broaden_point_wrapper(args):
    """
    A helper function to unpack arguments for use with ProcessPoolExecutor.map.
    This allows us to pass multiple arguments to the target function.
    """
    return broaden_point(*args)

def main():
    """
    Main function to parse arguments and run the Doppler broadening calculation.
    """
    parser = argparse.ArgumentParser(
        description="""Numerically Doppler broaden 0K cross sections to a target temperature 
                     using the Solbrig kernel method."""
    )
    parser.add_argument("input", help="Path to the input 0K cross-section file (.csv).")
    parser.add_argument("output", help="Path to save the output Doppler broadened cross-section file (.csv).")
    parser.add_argument("--mass", type=float, required=True, help="Mass of the target nucleus (amu).")
    parser.add_argument("--temp", type=float, default=294.0, help="Target temperature in Kelvin (default: 294.0 K).")
    parser.add_argument("--jobs", type=int, default=None, help="Number of parallel processes to use (default: all available cores).")
    args = parser.parse_args()

    print(f"Reading 0K cross sections from: {args.input}")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.input}")
        return

    energy_col = df.columns[0]
    E_vec = df[energy_col].values
    
    xs_cols_to_broaden = [col for col in df.columns if 'XS' in col]
    if not xs_cols_to_broaden:
        print("Error: No columns with 'XS' in the name found to broaden.")
        return

    print(f"Found cross section columns: {', '.join(xs_cols_to_broaden)}")

    output_df = pd.DataFrame({energy_col: E_vec})
    
    alpha = args.mass / (KB_EV * args.temp)
    E_grid_min, E_grid_max = E_vec.min(), E_vec.max()

    for col_name in xs_cols_to_broaden:
        print(f"\nBroadening column: '{col_name}' to {args.temp} K...")
        sigma_vec = df[col_name].values
        
        sigma_interp = interp1d(
            E_vec, 
            sigma_vec, 
            kind='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        tasks = [(E_out, sigma_interp, alpha, E_grid_min, E_grid_max) for E_out in E_vec]

        with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
            results = list(tqdm(executor.map( broaden_point_wrapper, tasks), total=len(tasks)))
        
        output_df[col_name] = results
        
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    output_df.to_csv(args.output, index=False, float_format='%.8e')
    print(f"\nSuccessfully broadened cross sections and saved to: {args.output}")

if __name__ == "__main__":
    main()
