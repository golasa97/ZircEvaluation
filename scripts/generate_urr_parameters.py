import numpy as np
import PySesh
import csv
import os

def channel_iterator(num_l_waves, target_spin):
    """
    Generates valid (L, J) channels and their multiplicities based on nuclear spin rules.
    This logic is based on the ChannelIterator subroutine in sesh1.f90.
    """
    channels = []
    
    I2 = int(2.0 * target_spin + 0.001)

    if I2 > 0:
        s2_vals = [abs(I2 - 1), I2 + 1]
    else:
        s2_vals = [1]

    for l_val in range(num_l_waves):
        l2 = 2 * l_val
        
        j2_vals_with_duplicates = []
        for s2 in s2_vals:
            j2_min = abs(s2 - l2)
            j2_max = s2 + l2
            for j2 in range(j2_min, j2_max + 1, 2):
                j2_vals_with_duplicates.append(j2)
        
        # Find unique J2 values and their counts (multiplicities)
        unique_j2s = sorted(list(set(j2_vals_with_duplicates)))
        
        j_local_idx = 0
        for j2 in unique_j2s:
            multiplicity = j2_vals_with_duplicates.count(j2)
            j_val = j2 / 2.0
            channels.append({'L': l_val, 'J': j_val, 'AMUN': float(multiplicity), 'j_local_idx': j_local_idx})
            j_local_idx += 1
            
    return channels

def get_density(U_ev, pairing_energy_ev, alevel_per_ev, target_mass):
    """
    Gilbert-Cameron composite level density.
    Based on Get_Density from macs6.f90.
    All energy inputs are in eV.
    """
    # Convert inputs to MeV as the formula expects
    U = U_ev / 1e6
    pairing_energy = pairing_energy_ev / 1e6
    alevel = alevel_per_ev * 1e6  # convert 1/eV to 1/MeV

    aplus1 = int(target_mass + 1.5)
    ematch = 2.5 + 150.0 / aplus1  # This is in MeV

    density = 0.0
    if U <= ematch:
        # Constant-temperature formula
        ueffx = ematch - pairing_energy
        if ueffx > 0:
            ueffx2 = np.sqrt(ueffx)
            sqrt_al = np.sqrt(alevel)
            
            denominator_T = (2.0 * sqrt_al / ueffx2 - 3.0 / ueffx)
            if denominator_T != 0:
                T = 2.0 / denominator_T
                denominator_A = (ueffx * ueffx2)
                if denominator_A != 0:
                    A = np.exp(2.0 * sqrt_al * ueffx2) / denominator_A
                    density = A * np.exp((U - ematch) / T)
    else:
        # Fermi-gas formula
        ueffx = U - pairing_energy
        if ueffx > 0:
            ueffx2 = np.sqrt(ueffx)
            sqrt_al = np.sqrt(alevel)
            denominator = (ueffx * ueffx2)
            if denominator != 0:
                density = np.exp(2.0 * sqrt_al * ueffx2) / denominator
    
    return density

def calculate_level_spacings(sesh, channels):
    """
    Calculates level spacings for all relevant J values at E=0, based on the
    Lespac subroutine from mfff3.f90.
    """
    target_mass = sesh.A
    binding_energy = sesh.BindingEnergy  # in eV
    pairing_energy = sesh.PairingEnergy  # in eV
    d_l0 = sesh.LevelSpacing  # in eV
    target_spin = sesh.spin
    num_l_waves = sesh.Numelv

    # 1. Constants and initial values
    Cr = 0.240
    aplus1 = int(target_mass + 1.5)
    max_spin = abs(target_spin) + 0.5

    # 2. Iteration to find alevel and disper
    ueff = binding_energy - pairing_energy
    
    if d_l0 <= 0 or max_spin <= 0 or ueff <= 0:
        return {}, 0.0

    cc = np.log(0.2365 * aplus1 * ueff / (d_l0 * max_spin))
    xo = cc
    varj2 = 0.0
    
    for _ in range(20):
        varj2 = 0.608 * Cr * xo * (aplus1)**(2.0/3.0)
        if varj2 <= 0: break

        fj_num = (np.exp(-(max_spin - 1.0)**2 / varj2) - np.exp(-(max_spin + 1.0)**2 / varj2))
        fj = fj_num * varj2 / (4.0 * max_spin)
        
        if fj <= 0 or xo <= 0: break

        xx = cc - np.log(fj) + 2.0 * np.log(xo)
        if abs(xx - xo) < 1.0e-6 * abs(xo):
            xo = xx
            break
        xo = xx
    
    alevel = xo**2 / (4.0 * ueff)  # alevel is in 1/eV
    varj2 = 0.608 * Cr * xo * (aplus1)**(2.0/3.0)

    if varj2 <= 0:
        return {}, alevel

    # 3. Calculate unnormalized level densities rhoj
    unique_j_vals = sorted(list(set(c['J'] for c in channels)))
    
    rhoj_map = {}
    for j_val in unique_j_vals:
        j_times_2 = 2.0 * j_val
        f1 = j_times_2**2
        f2 = (j_times_2 + 2.0)**2
        term1 = np.exp(-f1 / (4.0 * varj2))
        term2 = np.exp(-f2 / (4.0 * varj2))
        rhoj_map[j_val] = term1 - term2

    # 4. Calculate unnormalized level densities rho for each L
    rho_l = np.zeros(num_l_waves)
    for l_val in range(num_l_waves):
        j_vals_for_l = [c['J'] for c in channels if c['L'] == l_val]
        if j_vals_for_l:
            rho_l[l_val] = sum(rhoj_map.get(j, 0.0) for j in j_vals_for_l)

    # 5. Calculate level spacings d_j
    d_j_map = {}
    if rho_l[0] > 0:
        for j_val in unique_j_vals:
            rho_j = rhoj_map.get(j_val, 0.0)
            if rho_j > 0:
                d_j_map[j_val] = d_l0 * rho_l[0] / rho_j
            else:
                d_j_map[j_val] = np.inf
    
    return d_j_map, alevel

def calculate_aparam(sesh):
    target_mass = sesh.A
    binding_energy = sesh.BindingEnergy  # in eV
    pairing_energy = sesh.PairingEnergy  # in eV
    d_l0 = sesh.LevelSpacing  # in eV
    target_spin = sesh.spin
    num_l_waves = sesh.Numelv

    # 1. Constants and initial values
    Cr = 0.240
    aplus1 = int(target_mass + 1.5)
    max_spin = abs(target_spin) + 0.5

    # 2. Iteration to find alevel and disper
    ueff = binding_energy - pairing_energy
    
    if d_l0 <= 0 or max_spin <= 0 or ueff <= 0:
        return {}, 0.0

    cc = np.log(0.2365 * aplus1 * ueff / (d_l0 * max_spin))
    xo = cc
    varj2 = 0.0
    
    for _ in range(20):
        varj2 = 0.608 * Cr * xo * (aplus1)**(2.0/3.0)
        if varj2 <= 0: break

        fj_num = (np.exp(-(max_spin - 1.0)**2 / varj2) - np.exp(-(max_spin + 1.0)**2 / varj2))
        fj = fj_num * varj2 / (4.0 * max_spin)
        
        if fj <= 0 or xo <= 0: break

        xx = cc - np.log(fj) + 2.0 * np.log(xo)
        if abs(xx - xo) < 1.0e-6 * abs(xo):
            xo = xx
            break
        xo = xx
    
    alevel = xo**2 / (4.0 * ueff)  # alevel is in 1/eV
    return alevel


def bethe_formula(J, E, BE, PE, a, A):
    sigma_2 = (0.14592)*(A+1)**(2/3)*np.sqrt(a*(E + BE - PE))

    rho_j_unnorm = np.exp((-J**2)/(2*sigma_2)) - np.exp((-(J + 1)**2)/(2*sigma_2))
    return rho_j_unnorm

def lespac(sesh):

    spin = sesh.spin
    numelv = sesh.Numelv
    D_0 = sesh.LevelSpacing

    aparam = calculate_aparam(sesh)

    J_minus = np.abs(spin - 0.5)
    J_plus = spin + 0.5

    rho_minus = bethe_formula(J_minus, 0, sesh.BindingEnergy, sesh.PairingEnergy, aparam, sesh.A)
    rho_plus = bethe_formula(J_plus, 0, sesh.BindingEnergy, sesh.PairingEnergy, aparam, sesh.A)

    s_wave_rho = 1/(1/rho_minus + 1/rho_plus)
    return None
    



def generate_parameters(isotope_name, sesh_params, energy_grid):
    """
    Generates a table of URR parameters for a given isotope over an energy grid.
    """
    sesh = PySesh.Sesh(*sesh_params)
    
    data_rows = []
    header = ['I', 'L', 'J', 'AMUX', 'AMUN', 'AMUG', 'AMUF', 'E', 'D_J', 'Gx', 'Gn0', 'Gg', 'Gf']
    data_rows.append(header)

    channels = channel_iterator(sesh.Numelv, sesh.spin)
    
    # Calculate a-parameter needed for Bethe formula
    aparam = calculate_aparam(sesh)
    D0 = sesh.LevelSpacing
    
    d_j_map = {}

    # Loop over all L-waves to calculate normalized level spacings
    for l_val in range(sesh.Numelv):
        l_channels = [c for c in channels if c['L'] == l_val]
        if not l_channels:
            continue

        # Calculate target total level density for this L-wave
        # rho_L = (2L+1) * rho_0 = (2L+1) / D0
        target_total_rho_L = (2 * l_val + 1) / D0 if D0 > 0 else 0.0

        # Calculate unnormalized densities for all J in this L-wave
        unnormalized_rhos_L = {}
        for channel in l_channels:
            J = channel['J']
            # At E_neutron=0, excitation energy is Binding Energy
            rho_j_unnorm = bethe_formula(J, 0, sesh.BindingEnergy, sesh.PairingEnergy, aparam, sesh.A)
            unnormalized_rhos_L[J] = rho_j_unnorm
        
        total_unnormalized_rho_L = sum(unnormalized_rhos_L.values())

        # Determine normalization factor for this L-wave
        normalization_factor = 0.0
        if total_unnormalized_rho_L > 0:
            normalization_factor = target_total_rho_L / total_unnormalized_rho_L

        # Calculate normalized densities and level spacings for this L-wave
        for J, rho_unnorm in unnormalized_rhos_L.items():
            rho_norm = rho_unnorm * normalization_factor
            d_j = 1.0 / rho_norm if rho_norm > 0 else np.inf
            d_j_map[(l_val, J)] = d_j

    for E in energy_grid:
        # We still call this for the energy-dependent Gamma-gamma
        EDGG, EDD = sesh.EnergyDependence(E)
        
        for channel in channels:
            l = channel['L']
            j_val = channel['J']
            amun = channel['AMUN']
            j_local_idx = channel['j_local_idx']

            I = sesh.spin
            L = l
            J = j_val

            AMUN = amun
            AMUX = 0.0  # Ignored
            AMUG = 0.0  # Degrees of freedom for capture width, effectively infinite
            AMUF = 0.0  # Ignored
            
            d_j_base = d_j_map.get((l, j_val), np.inf)
            D_J = d_j_base * EDD if d_j_base != np.inf else np.inf

            Gx = 0.0  # Competitive width, assumed 0 for now
            Gn0 = sesh.GNR[j_local_idx, l] * EDD
            Gg = sesh.GammaGammas[l] * EDGG
            Gf = 0.0  # Fission width, ignored
            
            row = [I, L, J, AMUX, AMUN, AMUG, AMUF, E, D_J, Gx, Gn0, Gg, Gf]
            data_rows.append(row)

    # Correctly construct the path within the 'scripts' directory
    out_dir = os.path.dirname('UncertaintyQuantification')
    filename = os.path.join(out_dir, f"{isotope_name}_urr_parameters.csv")
    filename = f"UncertaintyQuantification/AvgParameters/{isotope_name}_urr_parameters.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerows(data_rows)
    
    print(f"Generated URR parameters for {isotope_name} and saved to {filename}")

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
