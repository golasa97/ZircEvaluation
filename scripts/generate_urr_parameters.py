import numpy as np
import PySesh
import csv
import os

def generate_parameters(isotope_name, sesh_params, energy_grid):
    """
    Generates a table of URR parameters for a given isotope over an energy grid.
    """
    sesh = PySesh.Sesh(*sesh_params)
    
    data_rows = []
    header = ['I', 'L', 'J', 'AMUX', 'AMUN', 'AMUG', 'AMUF', 'E', 'D_J', 'Gx', 'Gn0', 'Gg', 'Gf']
    data_rows.append(header)

    for E in energy_grid:
        EDGG, EDD = sesh.EnergyDependence(E)
        
        for l in range(sesh.Numelv):
            j_min_2 = sesh.J2N[l]
            j_max_2 = sesh.J2X[l]
            j_local_idx = 0
            for j2 in range(j_min_2, j_max_2 + 1, 2):
                if j2 < 0: continue
                j_val = j2 / 2.0
                
                I = sesh.spin
                L = l
                J = j_val

                AMUN = calculate_multiplicity_factor(I, l, J)
                AMUX = 0.0  # Ignored
                AMUG = 0.0  # Degrees of freedom for capture width, effectively infinite
                AMUF = 0.0  # Ignored
                
                D_J = sesh.DJL[j_local_idx, l] * EDD
                Gx = 0.0  # Competitive width, assumed 0 for now
                # Per user request, Gn0 from GNR. Applying energy dependence factor EDD.
                Gn0 = sesh.GNR[j_local_idx, l] * EDD
                Gg = sesh.GammaGammas[l] * EDGG
                Gf = 0.0  # Fission width, ignored
                
                row = [I, L, J, AMUX, AMUN, AMUG, AMUF, E, D_J, Gx, Gn0, Gg, Gf]
                data_rows.append(row)
                
                j_local_idx += 1

    # Correctly construct the path within the 'scripts' directory
    out_dir = os.path.dirname('UncertaintyQuantification')
    filename = os.path.join(out_dir, f"{isotope_name}_urr_parameters.csv")
    filename = f"UncertaintyQuantification/AvgParameters/{isotope_name}_urr_parameters.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(data_rows)
    
    print(f"Generated URR parameters for {isotope_name} and saved to {filename}")

def calculate_multiplicity_factor(I, l, J):
    """
    Calculates the multiplicity factor (mu) for a given target spin (I),
    orbital angular momentum (l), and total angular momentum (J).

    Args:
        I (float): Target nucleus spin.
        l (int): Orbital angular momentum.
        J (float): Total angular momentum.

    Returns:
        int: The multiplicity factor (1 or 2).
    """

    # Calculate channel spins
    s_minus = I - 0.5
    s_plus = I + 0.5

    # Check if J is valid for s_minus channel spin
    # The range is [abs(s_minus - l), s_minus + l]
    # Note: s_minus + l is the upper bound from the text (s_minus + l),
    # and abs(s_minus - l) is the lower bound from the text (|s_minus - l|).
    # Since J is always positive, we don't need abs on the upper bound for practical ranges.
    j_range_s_minus_lower = abs(s_minus - l)
    j_range_s_minus_upper = s_minus + l
    is_in_s_minus_range = (J >= j_range_s_minus_lower) and (J <= j_range_s_minus_upper)

    # Check if J is valid for s_plus channel spin
    j_range_s_plus_lower = abs(s_plus - l)
    j_range_s_plus_upper = s_plus + l
    is_in_s_plus_range = (J >= j_range_s_plus_lower) and (J <= j_range_s_plus_upper)

    # Determine multiplicity based on whether J, l can be formed from both s_plus and s_minus
    if is_in_s_minus_range and is_in_s_plus_range:
        return 2.0
    else:
        # If it's only in one or neither (which implies J,l is invalid if neither),
        # but for the context of multiplicity, if it exists, it's 1.
        # Assuming the calling function ensures J,l combination is physically possible.
        #if is_in_s_minus_range or is_in_s_plus_range:
        return 1.0

def main():
    sesh_params = {
        'zr90': (89.9047, 7.195, 1.25974, 6.31, 9459.0, 0.0, 3, np.array([5.68e-5, 1.406e-4, 3.08e-5]), np.array([-0.18860, -0.26640, -0.27440]), np.array([0.130, 0.250, 0.130])),
        'zr91': (90.9056, 8.635, 2.5022, 6.33, 550.0, 2.5, 3, np.array([0.55e-4, 7.04e-4, 0.53e-5]), np.array([-0.16588, -0.19491, -0.23081]), np.array([0.140, 0.220, 0.140])),
        'zr92': (91.9050, 7.2,   1.2443, 7.734, 3500.0, 0.0, 3, np.array([0.7e-4, 6.4e-4, 0.053e-4]), np.array([0.0, 0.0, 0.0]), np.array([0.1350, 0.220, 0.1350])  ),
        'zr94': (93.9063, 7.2,   1.2312, 6.462, 3200.0, 0.0, 3, np.array([0.52e-4, 9.4e-4, 0.12e-4]), np.array([0.0, 0.0, 0.0]), np.array([0.0850, 0.180, 0.0850])  ),
    }

    # Define energy grid (e.g., 1 keV to 1 MeV, 50 points)
    energy_grid = np.logspace(3, 6.5, 200)

    for name, params in sesh_params.items():
        generate_parameters(name, params, energy_grid)

if __name__ == '__main__':
    main()
