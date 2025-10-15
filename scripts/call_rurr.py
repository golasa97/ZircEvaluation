import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import juliacall
import multiprocessing
import functools


def generate_ladder(ladder_num, args, all_upars, nuc_data, energy_grid):
    """
    Worker function to be run in a separate process.
    It generates one full resonance ladder and saves the corresponding XS file.
    """
    try:
        # CRITICAL: Each new process needs its own Julia environment.
        # We must initialize juliacall and load the package INSIDE the worker.
        jl = juliacall.newmodule(f"PyCallSigma_Worker_{ladder_num}")
        jl.seval("using r_urr")
        
        # Re-create the Nucleus object for this process
        Nucleus = jl.Nucleus_type(nuc_data['A'], nuc_data['I'], nuc_data['a'], nuc_data['lmax'] + 1)
        
        # --- Core logic for one ladder ---
        sampled_pars = jl.sample_pars_by_energy(all_upars, args.emin, args.emax)
        nl, nJ, JJ, g, APLs = jl.pars_for_xsec(sampled_pars, Nucleus)

        total_xs = []
        capture_xs = []
        # The inner loop for the energy grid
        for E in energy_grid:
            xs_tuple = jl.sigma(E, sampled_pars, nl, nJ, JJ, Nucleus, g, APLs)
            total_xs.append(xs_tuple[0])
            capture_xs.append(xs_tuple[2])
        
        # Generate the unique output filename
        base, ext = os.path.splitext(args.output_file)
        output_filename = f"{base}_{ladder_num:03d}{ext}"

        # Save the results to a CSV
        results_df = pd.DataFrame({
            'Energy (eV)': energy_grid,
            'Total_XS (b)': total_xs,
            'Capture_XS (b)': capture_xs
        })
        results_df.to_csv(output_filename, index=False)
        
        # Return the filename for progress tracking
        return output_filename

    except Exception as e:
        print(f"\n--- ERROR in worker for Ladder {ladder_num}: {e} ---")
        return None


def main():
    """
    Main function to drive the cross-section calculation by calling the R-URR Julia package.
    """
    parser = argparse.ArgumentParser(
        description="Calculate 0K cross sections by calling a Julia R-URR package.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # ... (all your other arguments) ...
    parser.add_argument("--param_file", help="Path to the URR average parameter CSV file.")
    parser.add_argument("--output_file", help="Path to save the output cross-section CSV file.")
    # We no longer need julia_project_path if the package is developed
    # parser.add_argument("--julia_project_path", required=True, help="Path to the directory where the Julia project (R-URR) is located.")
    parser.add_argument("--emin", type=float, default=2e5, help="Minimum energy for the grid (eV).")
    parser.add_argument("--emax", type=float, default=1e6, help="Maximum energy for the grid (eV).")
    parser.add_argument("--points", type=int, default=5000, help="Number of points for the energy grid.")
    parser.add_argument("--mass", type=float, required=True, help="Mass of the target nucleus")
    parser.add_argument("--radius", type=float, required=True, help="Channel radius of target nucleus")
    parser.add_argument("--nLadders", type=int, default=100, help="Number of unique resonance ladders to sample.")
    
    # NEW: Argument to control the number of parallel processes
    parser.add_argument("--cores", type=int, default=os.cpu_count(), help="Number of CPU cores to use for parallel processing.")
    args = parser.parse_args()

    # --- Setup that only needs to be done once in the main process ---
    print("--- Initializing main process ---")
    
    # We only need to initialize Julia once in the main process to read the upars.
    # The workers will create their own instances.
    try:
        jl_main = juliacall.newmodule("PyCallSigma_Main")
        jl_main.seval("using r_urr")
        print("Julia environment loaded successfully for main process.")
    except Exception as e:
        print(f"\n--- FATAL ERROR: Could not initialize Julia in main process: {e} ---")
        return
        
    print(f"--- Reading URR parameter file: {args.param_file} ---")
    df = pd.read_csv(args.param_file, delimiter=';')
    first_row = df.iloc[0]
    # We pass nuc_data as a dictionary, since the Julia object can't be passed between processes
    nuc_data = {
        'A': args.mass,
        'I': float(first_row['I']),
        'a': args.radius,
        'lmax': int(df['L'].max())
    }
    
    # Read the upars and create the energy grid once
    all_upars = jl_main.read_upars(args.param_file)
    energy_grid = np.logspace(np.log10(args.emin), np.log10(args.emax), args.points)
    ladder_numbers = range(1, args.nLadders + 1)
    
    # --- NEW: The multiprocessing Pool ---
    print(f"\n--- Starting parallel sampling for {args.nLadders} ladders using {args.cores} cores ---")
    
    # 'functools.partial' lets us "freeze" the arguments that are the same for every worker
    worker_with_args = functools.partial(generate_ladder, args=args, all_upars=all_upars, nuc_data=nuc_data, energy_grid=energy_grid)
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=args.cores) as pool:
        # Use tqdm to create a single, clean progress bar for the whole operation
        # pool.imap_unordered processes tasks as they complete, which is great for progress bars
        results = list(tqdm(pool.imap_unordered(worker_with_args, ladder_numbers), total=args.nLadders, desc="Processing Ladders"))

    print(f"\nCalculation complete. Generated {len(results)} cross-section files.")

if __name__ == "__main__":
    main()


#def main():
#    """
#    Main function to drive the cross-section calculation by calling the R-URR Julia package.
#    """
#    parser = argparse.ArgumentParser(
#        description="Calculate 0K cross sections by calling a Julia R-URR package.",
#        formatter_class=argparse.RawTextHelpFormatter
#    )
#    parser.add_argument("--param_file", help="Path to the URR average parameter CSV file (e.g., zr90_urr_parameters.csv).")
#    parser.add_argument("--output_file", help="Path to save the output cross-section CSV file.")
#    parser.add_argument("--julia_project_path", required=True, help="Path to the directory where the Julia project (R-URR) is located.")
#    parser.add_argument("--emin", type=float, default=1e5, help="Minimum energy for the grid (eV).")
#    parser.add_argument("--emax", type=float, default=1e6, help="Maximum energy for the grid (eV).")
#    parser.add_argument("--points", type=int, default=10000, help="Number of points for the energy grid.")
#    parser.add_argument("--mass", type=float, required=True, help="Mass of the target nucleus")
#    parser.add_argument("--radius", type=float, required=True, help="Channel radius of target nucleus")
#    parser.add_argument("--nLadders", type=int, default=1, help="Number of unique resonance ladders to sample.")
#    args = parser.parse_args()
#
#    print("--- Setting up Julia environment ---")
#    try:
#        # Initialize juliacall, activating the environment in the specified project path
#        jl = juliacall.newmodule("PyCallSigma")
#        jl.seval("using r_urr")
#        #jl.Pkg.activate(args.julia_project_path)
#        
#        # Now, import the necessary modules from that activated project
#        #res_types = jl.r_urr.res_types
#        #R_matrix = jl.r_urr.R_matrix
#        #urr_params = jl.r_urr.urr_parameters
#
#        #res_types = jl.include("/home/alec/ResearchData/ZircEvaluation/r-urr/src/res_types.jl")
#        #R_matrix = jl.include("/home/alec/ResearchData/ZircEvaluation/r-urr/src/R_matrix.jl")
#        #urr_params = jl.include("/home/alec/ResearchData/ZircEvaluation/r-urr/src/urr_parameters.jl")
#        print("Julia environment and modules loaded successfully.")
#    except Exception as e:
#        print("\n--- FATAL ERROR: Could not initialize Julia or load modules. ---")
#        print("Please check the following:")
#        print(f"1. Is Julia installed and accessible in your system's PATH?")
#        print(f"2. Is the path to the Julia project correct? You provided: '{args.julia_project_path}'")
#        print(f"3. Have you instantiated the Julia project? (See instructions)")
#        print(f"Error details: {e}")
#        return
#
#    print(f"--- Reading and processing URR parameter file: {args.param_file} ---")
#    df = pd.read_csv(args.param_file, delimiter=';')
#    A = args.mass
#    r = args.radius
#
#    # Extract nuclear data from the first row of parameters
#    first_row = df.iloc[0]
#    nuc_data = {
#        'A': A, # Atomic mass - This should ideally be in the file or an argument
#        'I': float(first_row['I']),      # Ground state spin
#        'a': r,   # Channel radius in fm - This should also be an argument or in the file
#        'lmax': int(df['L'].max())
#    }
#    
#    # Create the Nucleus_type object required by the Julia functions
#    Nucleus = jl.Nucleus_type(nuc_data['A'], nuc_data['I'], nuc_data['a'],  nuc_data['lmax'] + 1 )
#    
#    print("--- Preparing Julia inputs from parameter file ---")
#    # `prep_pars` is the function in the Julia package designed to read this file format
#    # It returns the resonance parameters structured correctly for the `sigma` function
#    # The second argument (1e5) is a dummy energy value, as the function reads the whole grid.
#    upars = jl.read_upars(args.param_file)
#    for i in range(args.nLadders):
#        ladder_num = i + 1
#        print(f"\n[Ladder {ladder_num}/{args.nLadders}]")
#
#        sampled_pars = jl.sample_pars_by_energy(upars, args.emin, args.emax)
#
#    
#        # The Julia function also needs some pre-calculated values
#        nl, nJ, JJ, g, APLs = jl.pars_for_xsec(sampled_pars, Nucleus)
#
#        print("--- Calculating cross sections across the energy grid ---")
#        energy_grid = np.logspace(np.log10(args.emin), np.log10(args.emax), args.points)
#        total_xs = []
#        capture_xs = []
#
#        for E in tqdm(energy_grid, desc="Calculating 0K Cross Section"):
#            # The magic moment! Calling the Julia 'sigma' function directly from Python.
#            # It returns a tuple: (total, elastic, capture, fission)
#            xs_tuple = jl.sigma(E, sampled_pars, nl, nJ, JJ, Nucleus, g, APLs)
#        
#            # The sigma function returns a tuple of cross sections.
#            # Based on typical R-matrix codes: (total, elastic, capture, fission)
#            total_xs.append(xs_tuple[0])
#            capture_xs.append(xs_tuple[2])
#
#        base, ext = os.path.splitext(args.output_file)
#        output_filename = f"{base}_{ladder_num:03d}{ext}"
#
#        print(f"--- Saving results for Ladder {ladder_num} to {output_filename} ---")
#        results_df = pd.DataFrame({
#            'Energy (eV)': energy_grid,
#            'Total_XS (b)': total_xs,
#            'Capture_XS (b)': capture_xs
#        })
#        results_df.to_csv(output_filename, index=False)
#
#    print(f"\nCalculation complete. You're all set! Generated {args.nLadders} cross-section files.")
#
#
#
#if __name__ == "__main__":
#    main()
