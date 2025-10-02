import json
import os
import numpy as np
from scipy.signal import find_peaks

def weighted_trapezoid(x, y, yerr):
    """
    Calculates the average of y over x using a modified trapezoidal rule
    where segment heights are weighted by endpoint uncertainties.
    """
    integral_val = 0.0
    integral_variance = 0.0
    integral_x = 0.0
    num_points = len(x)

    if num_points < 2:
        return (x[0], y[0], yerr[0]) if num_points == 1 else (np.nan, np.nan, np.nan)

    for i in range(num_points - 1):
        y_0, y_1 = y[i], y[i+1]
        x_0, x_1 = x[i], x[i+1]
        yerr_0, yerr_1 = yerr[i], yerr[i+1]

        if yerr_0 <= 0 or yerr_1 <= 0:
            raise ValueError(f"yerr must be positive.")

        w_0 = 1.0 / (yerr_0**2)
        w_1 = 1.0 / (yerr_1**2)
        delta_x = x_1 - x_0

        if delta_x <= 0:
            raise ValueError(f"x values must be distinct and monotonically increasing.")

        integral_val += (w_0 * y_0 + w_1 * y_1) / (w_0 + w_1) * delta_x
        integral_x += (w_0 * x_0 + w_1 * x_1) / (w_0 + w_1) * delta_x
        
        yerr0_sq = yerr_0**2
        yerr1_sq = yerr_1**2
        sum_yerr_sq = yerr0_sq + yerr1_sq

        current_segment_variance = (delta_x**2 * yerr0_sq * yerr1_sq) / sum_yerr_sq if sum_yerr_sq > 0 else 0.0
        integral_variance += current_segment_variance

    total_width = x[-1] - x[0]
    average_value = integral_val / total_width
    average_x = integral_x / total_width
    average_uncertainty = np.sqrt(integral_variance) / total_width

    return average_x, average_value, average_uncertainty

def cut_data(sorted_x, sorted_y, sorted_yerr, x_start, x_end):
    """
    Selects data points within a given x-range from sorted arrays.
    """
    selection_mask = (sorted_x >= x_start) & (sorted_x <= x_end)
    return sorted_x[selection_mask], sorted_y[selection_mask], sorted_yerr[selection_mask]

def analyze_std_dev_vs_bin_width(sorted_x, sorted_y, E_upper, max_width, step_size):
    """
    Analyzes the standard deviation of y-values within bins of increasing width starting from E_upper.
    """
    target_widths = np.arange(step_size, max_width + step_size, step_size)
    results = []

    for width in target_widths:
        E_lower = E_upper - width
        if E_lower >= E_upper:
            continue

        bin_x, bin_y, _ = cut_data(sorted_x, sorted_y, sorted_y, E_lower, E_upper)
        
        std_dev = np.std(bin_y) if len(bin_y) >= 2 else np.nan
        results.append([width, std_dev])

    return np.array(results)

def find_optimal_bin_edges(sorted_x, sorted_y, start_energy, min_energy, max_bin_width, step_size, min_bin_width):
    """
    Determines the optimal bin edges by iteratively finding the minimum in the std. dev. vs. bin width curve,
    respecting a minimum bin width.
    """
    bin_edges = [start_energy]
    current_E_upper = start_energy

    while current_E_upper > min_energy and (current_E_upper - min_energy) > min_bin_width:
        std_dev_data = analyze_std_dev_vs_bin_width(sorted_x, sorted_y, current_E_upper, max_bin_width, step_size)
        
        valid_indices = ~np.isnan(std_dev_data[:, 1])
        if np.sum(valid_indices) < 3: # Need enough points to find a meaningful minimum
            break

        widths = std_dev_data[valid_indices, 0]
        std_devs = std_dev_data[valid_indices, 1]

        peaks, _ = find_peaks(-std_devs, prominence=0.05)
        print(f"Current Energy Upper: {current_E_upper} -- Num peaks: {len(peaks)}")
        
        optimal_width = None
        if len(peaks) > 0:
            # Find the first peak that corresponds to a width >= min_bin_width
            for peak_index in peaks:
                if widths[peak_index] >= min_bin_width:
                    optimal_width = widths[peak_index]
                    break # Found the first valid one

        # If no peak satisfied the condition, or if there were no peaks, fallback to max_bin_width
        if optimal_width is None:
            optimal_width = max_bin_width

        new_edge = current_E_upper - optimal_width
        if new_edge < min_energy:
            break
            
        bin_edges.append(new_edge)
        current_E_upper = new_edge

    bin_edges.append(min_energy)
    return sorted(list(set(bin_edges)))


def main():
    input_dir = 'experimental_data/pointwise'
    output_dir = 'experimental_data/resonance_averaged'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue

        print(f"Processing pointwise data file {filename}")

        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r') as f:
            data = json.load(f)

        x = np.array(data['EXFOR']['x-vals'])
        y = np.array(data['EXFOR']['y-vals'])
        y_err = np.array(data['EXFOR']['yerr-vals'])
        
        pointwise_t = np.array(data['Processed']['pointwise T'])
        pointwise_t_err = np.array(data['Processed']['pointwise Terr'])

        # Remove duplicate x-values
        _, unique_indices = np.unique(x, return_index=True)
        x = x[unique_indices]
        y = y[unique_indices]
        y_err = y_err[unique_indices]
        pointwise_t = pointwise_t[unique_indices]
        pointwise_t_err = pointwise_t_err[unique_indices]

        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_y_err = y_err[sorted_indices]
        sorted_t = pointwise_t[sorted_indices]
        sorted_t_err = pointwise_t_err[sorted_indices]

        start_energy = sorted_x[-1]
        min_energy = sorted_x[0]

        # If the highest energy is above 2.5 MeV, start the analysis from the first point below it.
        if start_energy > 2.5e6:
            try:
                start_energy = sorted_x[sorted_x <= 2.5e6][-1]
                print(f"  Highest energy is above 2.5 MeV. Starting analysis from {start_energy/1e6:.2f} MeV.")
            except IndexError:
                print("  Warning: No data points found below 2.5 MeV. Skipping file.")
                continue

        # Adjust binning parameters for the energy-dependent region
        min_bin_width = 40e3  # 40 keV minimum bin width
        if 'zr90' in filename and 700e3 - 275e3 < start_energy:
             max_bin_width = 100e3
             step_size = 1e1
        else:
             max_bin_width = 200e3
             step_size = 1e1

        bin_edges = find_optimal_bin_edges(sorted_x, sorted_y, start_energy, min_energy, max_bin_width, step_size, min_bin_width)
        
        processed_bins = []
        for i in range(len(bin_edges) - 1):
            E_min, E_max = bin_edges[i], bin_edges[i+1]
            
            bin_x, bin_y, bin_y_err = cut_data(sorted_x, sorted_y, sorted_y_err, E_min, E_max)
            _, bin_t, bin_t_err = cut_data(sorted_x, sorted_t, sorted_t_err, E_min, E_max)

            if len(bin_x) > 0:
                avg_E, avg_xs, avg_xs_err = weighted_trapezoid(bin_x, bin_y, bin_y_err)
                _, avg_t, avg_t_err = weighted_trapezoid(bin_x, bin_t, bin_t_err)

                processed_bins.append({
                    'E_min': E_min,
                    'E_max': E_max,
                    'avg_E': avg_E,
                    'avg_xs': avg_xs,
                    'avg_xs_err': avg_xs_err,
                    'avg_t': avg_t,
                    'avg_t_err': avg_t_err
                })
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w') as f:
            json.dump(processed_bins, f, indent=4)
        
        print(f"Processed data for {filename} saved to {output_path}")

if __name__ == '__main__':
    main()
