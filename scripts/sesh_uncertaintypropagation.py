import numpy as np
import PySesh
import concurrent.futures
from tqdm import tqdm
import json
import os

def get_resonance_ladder_uncertainty(
    sesh_params,
    dataset_info,
    num_ladders,
    num_integration_steps,
    temperature
):
    """
    Worker function to calculate transmission and cross-section for one resonance ladder.
    """
    # Initialize Sesh objects for each isotope
    sesh_objects = {iso: PySesh.Sesh(*params) for iso, params in sesh_params.items()}

    # Determine the global energy range for the ladder
    E_min_global = min(entry['avg_E'] - entry['half_bin_width'] for entry in dataset_info)
    E_max_global = max(entry['avg_E'] + entry['half_bin_width'] for entry in dataset_info)

    # Generate one global ladder for each isotope for this iteration
    for sesh_obj in sesh_objects.values():
        sesh_obj.generate_global_ladders(E_min_global, E_max_global)

    # Calculate results for each energy bin using the same ladder
    results_for_ladder = []
    for entry in dataset_info:
        total_transmission = 1.0
        total_xs = 0.0
        
        for iso, ab in entry['abundances'].items():
            sesh_obj = sesh_objects[iso]
            thickness = entry['thickness'] * ab
            
            T_iso, xs_iso = sesh_obj.EnergyIntegratedTransmission(
                entry['avg_E'], entry['half_bin_width'], 
                num_integration_steps, thickness, temperature
            )
            total_transmission *= T_iso
            total_xs += ab * xs_iso
        
        denominator = np.exp(-entry['thickness'] * total_xs)
        ct = total_transmission / denominator if denominator != 0 else np.inf
        results_for_ladder.append(ct)
        
    return results_for_ladder

def main():
    # Load resonance parameters
    sesh_params = {
        'zr90': (89.9047, 7.195, 1.25974, 6.31, 9459.0, 0.0, 3, np.array([5.68e-5, 1.406e-4, 3.08e-5]), np.array([-0.18860, -0.26640, -0.27440]), np.array([0.130, 0.250, 0.130])),
        'zr91': (90.9056, 8.635, 2.5022, 6.33, 550.0, 2.5, 3, np.array([0.55e-4, 7.04e-4, 0.53e-5]), np.array([-0.16588, -0.19491, -0.23081]), np.array([0.140, 0.220, 0.140])),
        # Add other isotopes if needed
    }

    input_dir = 'experimental_data/resonance_averaged'
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json') or 'nat-zr' in filename: # Skip natural Zr for now
            continue

        print(f"Processing {filename}...")
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r') as f:
            binned_data = json.load(f)

        # This part needs to be adapted to your specific file structure and abundance info
        # Example for Musgrove Zr-90
        if 'zr90_Musgrove' in filename:
            abundances = {'zr90': 0.9765, 'zr91': 0.0093} 
            thickness = 0.08271
        elif 'zr91_Musgrove' in filename:
            abundances = {'zr90': 0.058, 'zr91': 0.892}
            thickness = 0.06423
        else:
            continue # Skip files you don't have abundance info for

        dataset_info = [
            {
                'avg_E': bin_data['avg_E'],
                'half_bin_width': (bin_data['E_max'] - bin_data['E_min']) / 2.0,
                'thickness': thickness,
                'abundances': abundances
            } for bin_data in binned_data
        ]
        
        num_ladders = 100  # Number of Monte Carlo iterations
        num_integration_steps = 50
        temperature = 294.0

        all_results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(get_resonance_ladder_uncertainty, sesh_params, dataset_info, 1, num_integration_steps, temperature) for _ in range(num_ladders)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_ladders):
                all_results.append(future.result())

        # Transpose results to get [num_bins, num_ladders]
        results_by_bin = np.array(all_results).T
        
        # Calculate uncertainties for each bin
        for i, bin_data in enumerate(binned_data):
            mean_ct = np.mean(results_by_bin[i])
            std_ct = np.std(results_by_bin[i])
            bin_data['self_shielding_correction_factor'] = mean_ct
            bin_data['self_shielding_correction_uncertainty'] = std_ct
            print(f"  Bin {i+1}: E_avg={bin_data['avg_E']/1e6:.2f} MeV, C_T={mean_ct:.4f} +/- {std_ct:.4f}")

        # Save updated data
        with open(input_path, 'w') as f:
            json.dump(binned_data, f, indent=4)

if __name__ == '__main__':
    main()