import numpy as np
import sys
import os
import csv
import argparse
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

# Add the project root to the Python path to allow importing PySesh
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#if project_root not in sys.path:
#    sys.path.insert(0, project_root)
import PySesh


def read_urr_parameters(filepath):
    """Reads the URR parameter CSV file into a dictionary keyed by (L, J)."""
    params_by_channel = defaultdict(lambda: {'E': [], 'D_J': [], 'Gn0': [], 'Gg': [], 'AMUN' : 0})
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
                params_by_channel[channel]['AMUN'] = float(row[idx['AMUN']])
            except (ValueError, IndexError):
                print(f"Skipping malformed row: {row}")
                continue

    # Convert lists to numpy arrays for easier interpolation
    for channel in params_by_channel:
        for key in params_by_channel[channel]:
            params_by_channel[channel][key] = np.array(params_by_channel[channel][key])
            
    return dict(params_by_channel), ground_state_spin

def sample_full_ladder(params_by_channel, mass, chan_rad):
    """
    Samples a complete resonance ladder across the entire energy range defined
    in the parameter file.
    """
    # Create a Sesh object with the correct physics parameters for penetrability calculation.
    # Other parameters are dummies but need to have the correct shape to avoid init errors.
    sesh_sampler = PySesh.Sesh(mass, 1, 1, chan_rad, 1, 1, 2, np.array([1,1]), np.array([1,1]), np.array([1,1]))
    
    full_ladder = []
    
    # Determine the global energy range from the data
    all_energies = [e for channel_data in params_by_channel.values() for e in channel_data['E']]
    if not all_energies:
        return [], 0, 0
    e_min = min(all_energies)
    e_max = max(all_energies)

    rng1 = np.random.default_rng()
    wig_factor = np.sqrt(-np.log(rng1.random(10000000)))
    pt_factor_1 = rng1.chisquare(1, 10000000)
    pt_factor_2 = rng1.chisquare(2, 10000000)

    pt_factor = [0, pt_factor_1, pt_factor_2]

    i = 0

    for channel, data in params_by_channel.items():
        l, j = channel
        
        if len(data['E']) < 2:
            continue

        current_e = e_min
        
        while current_e < e_max:
            i += i
            # Interpolate parameters at the current energy to find the next spacing
            d_avg = np.interp(current_e, data['E'], data['D_J'])
            
            if d_avg <= 0:
                break

            # Sample the next spacing from the Wigner distribution
            #spacing = sesh_sampler.Wigner(d_avg, 1)[0]

            spacing = d_avg*wig_factor[i]
            if spacing <= 1e-9: spacing = 1e-9

            e_r = current_e + spacing
            
            if e_r >= e_max:
                break
            
            # Interpolate widths at the new resonance energy
            gn0_avg = np.interp(e_r, data['E'], data['Gn0'])
            gg_avg = np.interp(e_r, data['E'], data['Gg'])

            # Calculate penetrability (VL)
            EkeV = e_r / 1000.0
            AK = sesh_sampler.chan_rad * np.sqrt(EkeV / sesh_sampler.AA) / 143.92
            VL = sesh_sampler.Pf(AK, l) / AK if AK > 0 else 0
            
            # Convert average reduced width to average full width
            gn_avg = gn0_avg * np.sqrt(e_r) * VL

            gn = gn_avg*np.random.chisquare(data['AMUN'])

            # Sample the full neutron width from the Porter-Thomas distribution
            #gn = sesh_sampler.Porter(gn_avg, 1)[0]
            
            # Add the new resonance to the full ladder
            full_ladder.append((e_r, l, j, gn, gg_avg))
            
            current_e = e_r
            
    # Sort the final combined ladder by energy
    full_ladder.sort(key=lambda x: x[0])
    
    return full_ladder, e_min, e_max

def generate_and_write_ladder(task_args):
    """Worker function to generate one ladder and write it to a file."""
    params_by_channel, mass, radius, output_filename, isotope_name, spin = task_args
    
    resonance_ladder, e_min, e_max = sample_full_ladder(params_by_channel, mass, radius)
    
    with open(output_filename, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        f.write(f"# Isotope: {isotope_name}\n")
        f.write(f"# Ground State Spin (I): {spin}\n")
        f.write(f"# Atomic Mass (A): {mass}\n")
        f.write(f"# Channel Radius (a): {radius}\n")
        f.write(f"# Ladder sampled from {e_min:.2f} eV to {e_max:.2f} eV\n")
        writer.writerow(["E_r (eV)", "L", "J", "Gn (eV)", "Gg (eV)"])
        
        for res in resonance_ladder:
            writer.writerow([f"{res[0]:.4f}", res[1], res[2], f"{res[3]:.6e}", f"{res[4]:.6e}"])
            
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
        results = list(tqdm(executor.map(generate_and_write_ladder, tasks), total=len(tasks), desc=f"Generating {args.num_ladders} ladders for {isotope_name}"))

    print(f"\nProcessing complete. {len(results)} ladder files written to {args.output_dir}.")

if __name__ == '__main__':
    main()

