import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    # --- Tick Settings ---
    'xtick.direction': 'in',             # X-axis major ticks direction
    'ytick.direction': 'in',             # Y-axis major ticks direction
    'xtick.major.size': 6,               # X-axis major tick length
    'ytick.major.size': 6,               # Y-axis major tick length
    'xtick.minor.size': 3,               # X-axis minor tick length
    'ytick.minor.size': 3,               # Y-axis minor tick length
    'xtick.minor.visible': True,         # Display X-axis minor ticks
    'ytick.minor.visible': True,         # Display Y-axis minor ticks
    'xtick.top': True,                   # Display X-axis ticks on top
    'ytick.right': True,                 # Display Y-axis ticks on right

    # --- Grid Settings ---
    'axes.grid': True,                   # Display grid
    'axes.grid.which': 'major',          # Grid on major ticks
    'grid.color': 'gray',                # Grid color
    'grid.linestyle': '--',              # Grid line style
    'grid.alpha': 0.7,                   # Grid transparency

    # --- Font Settings (from previous discussions) ---
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],   # Or 'STIXGeneral', 'Georgia', etc.
    'mathtext.fontset': 'stix',          # Serif math font

    # --- Optional: Font sizes ---
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})

def weighted_trapezoid(x, y, yerr):
    """
    Calculates the average of y over x using a modified trapezoidal rule
    where segment heights are weighted by endpoint uncertainties.
    The uncertainty of the average is also propagated.
    This method assumes segment uncertainty contributions are summed (approximation).
    """
    integral_val = 0.0
    integral_variance = 0.0
    integral_x = 0.0
    num_points = len(x)

    if num_points < 2:
        if num_points == 1:
            # For a single point, the "average" is the point itself.
            # The concept of an integral over a zero-width interval is tricky.
            # Depending on context, error could be yerr[0] or np.nan.
            return x[0], y[0], yerr[0]
        return np.nan, np.nan, np.nan

    for i in range(num_points - 1):
        y_0, y_1 = y[i], y[i+1]
        x_0, x_1 = x[i], x[i+1]
        yerr_0, yerr_1 = yerr[i], yerr[i+1]

        if yerr_0 <= 0 or yerr_1 <= 0:
            # Handle non-positive errors: could raise error, use a floor, or specific logic for infinite weights
            # For this example, let's raise an error if not positive, as weights 1/err^2 need err > 0.
            # A more robust solution would define behavior for yerr=0 (infinite weight).
            raise ValueError(f"yerr must be positive. Found yerr[{i}]={yerr_0}, yerr[{i+1}]={yerr_1}")

        w_0 = 1.0 / (yerr_0**2)
        w_1 = 1.0 / (yerr_1**2)

        delta_x = x_1 - x_0
        if delta_x <= 0:
            raise ValueError(f"x values must be distinct and monotonically increasing. Found x[{i}]={x_0}, x[{i+1}]={x_1}")

        # --- Integral Value Calculation (Method 1 segment) ---
        integral_val += (w_0 * y_0 + w_1 * y_1) / (w_0 + w_1) * delta_x
        integral_x += (w_0 * x_0 + w_1 * x_1) / (w_0 + w_1) * delta_x

        # --- Error Calculation for Method 1 segment ---
        # (Delta A_k)^2 = (delta_x)^2 * (yerr_0^2 * yerr_1^2) / (yerr_0^2 + yerr_1^2)
        yerr0_sq = yerr_0**2
        yerr1_sq = yerr_1**2
        sum_yerr_sq = yerr0_sq + yerr1_sq

        current_segment_variance = 0.0 # Initialize for safety
        if sum_yerr_sq > 0: # Avoid division by zero if both errors happened to be zero (though checked above)
            current_segment_variance = (delta_x**2 * yerr0_sq * yerr1_sq) / sum_yerr_sq
        # If sum_yerr_sq is 0 (both yerr_0 and yerr_1 are zero), segment variance is 0.

        integral_variance += current_segment_variance

    total_width = x[-1] - x[0]
    # total_width should be positive due to checks if num_points >= 2

    average_value = integral_val / total_width
    average_x = integral_x / total_width
    average_uncertainty = np.sqrt(integral_variance) / total_width

    return average_x, average_value, average_uncertainty

def cut_data(unsorted_x, unsorted_y, unsorted_yerr, x_start, x_end):
    """
    Selects data points within a given x-range and sorts them by x.
    """
    # Convert to numpy arrays if they aren't already
    unsorted_x_np = np.array(unsorted_x)
    unsorted_y_np = np.array(unsorted_y)
    unsorted_yerr_np = np.array(unsorted_yerr)

    # Get sorted indices based on x values
    sorted_indices = np.argsort(unsorted_x_np)

    # Sort all three arrays based on these indices
    sorted_x = unsorted_x_np[sorted_indices]
    sorted_y = unsorted_y_np[sorted_indices]
    sorted_yerr = unsorted_yerr_np[sorted_indices]

    # Create a boolean mask for the desired x_start and x_end range
    # Inclusive of x_start and x_end
    selection_mask = (sorted_x >= x_start) & (sorted_x <= x_end)

    # Apply the mask to get the final lists/arrays
    final_x = sorted_x[selection_mask]
    final_y = sorted_y[selection_mask]
    final_yerr = sorted_yerr[selection_mask]

    # The problem asked for lists, so convert if necessary, though arrays are often more useful.
    # For direct use with your previous trapezoid function expecting lists, this conversion is fine.
    return list(final_x), list(final_y), list(final_yerr)

def extract_exfor_data(url: str):
    """
    Fetches data from a plain text EXFOR URL, extracts specified metadata,
    X, Y, YERR columns, and their units, converting to a standard output format.
    Minimal implementation for plain text EXFOR files.
    """
    # Initialize the output dictionary with default target units
    out = {
        'x-vals': [], 'x-unit': 'EV',
        'y-vals': [], 'y-unit': 'BARNS',
        'yerr-vals': [], 'yerr-unit': 'BARNS', # Assumes yerr converts like y
        'url': url,
        'REFERENCE': "",
        'AUTHOR': "",
        'TITLE': "",
        'SAMPLE': "",
        'METHOD': "",
        'DETECTOR': ""
    }

    raw_x_values = []
    raw_y_values = []
    raw_yerr_values = []
    parsed_x_unit_original = "UNKNOWN"
    parsed_y_unit_original = "UNKNOWN"
    parsed_yerr_unit_original = "UNKNOWN"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        text_content = response.text
    except requests.exceptions.RequestException:
        return out # Return default structure on fetch error

    lines = text_content.splitlines()

    current_metadata_keyword_internal = None # For tracking multi-line metadata

    # State for parsing data block:
    # 0 = seeking metadata or "DATA" keyword
    # 1 = seeking column name header (e.g., EN DATA DATA-ERR)
    # 2 = seeking column unit header (e.g., MEV MB MB)
    # 3 = parsing numerical data
    data_parse_state = 0

    for line_content in lines:
        stripped_line = line_content.strip()
        original_line = line_content # Keep for checking indentation

        columns = stripped_line.split()
        if not columns: # Skip empty or whitespace-only lines
            continue

        first_word = columns[0]

        # --- Metadata Parsing ---
        # Check if the line starts with one of our target metadata keywords
        # These keywords are case-sensitive in EXFOR.
        # Assumes value starts after the keyword on the same line.
        metadata_keywords_to_check = ["REFERENCE", "AUTHOR", "TITLE", "SAMPLE", "METHOD", "DETECTOR"]
        is_new_metadata_keyword_line = False

        if data_parse_state == 0: # Only look for general metadata if not in data block parsing
            for kw in metadata_keywords_to_check:
                if stripped_line.startswith(kw):
                    # Finalize any previously accumulating metadata field
                    if current_metadata_keyword_internal and out[current_metadata_keyword_internal]:
                        out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()

                    current_metadata_keyword_internal = kw
                    # Extract value: rest of the line after keyword, strip leading/trailing spaces
                    value_part = stripped_line[len(kw):].strip()
                    if out[kw]: # Append if keyword appears multiple times (e.g. multiple AUTHOR lines)
                        out[kw] += "; " + value_part
                    else:
                        out[kw] = value_part
                    is_new_metadata_keyword_line = True
                    break # Found a keyword for this line

            if not is_new_metadata_keyword_line and current_metadata_keyword_internal and \
               (original_line.startswith("          ") or original_line.startswith("            ") or \
                (stripped_line and original_line.startswith(" ") and not columns[0].isupper())):
                # Heuristic for continuation line: starts with significant spaces
                # or doesn't look like a new uppercase keyword.
                if out[current_metadata_keyword_internal]:
                     out[current_metadata_keyword_internal] += " " + stripped_line
                else: # Should not happen if current_metadata_keyword_internal is set
                     out[current_metadata_keyword_internal] = stripped_line
                is_new_metadata_keyword_line = True # Processed as part of current metadata

        # --- Data Block Parsing State Machine ---
        if first_word == "DATA" and data_parse_state == 0:
            if current_metadata_keyword_internal: # Finalize any pending metadata
                 out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()
                 current_metadata_keyword_internal = None
            data_parse_state = 1 # Transition to seeking column names
            continue

        if data_parse_state == 1: # Seeking column NAME header
            if len(columns) >= 3 and \
               ("EN" in columns[0]) and \
               ("DATA" in columns[1]) and \
               ("ERR" in columns[2] or "DATA-ERR" in columns[2]):
                data_parse_state = 2 # Found name header, next non-empty line should be units
            # If it's not a name header, it might be a misidentified line; stay in state 1 or reset.
            # For minimal, we just continue and hope next line is units or data.
            continue

        elif data_parse_state == 2: # Seeking column UNIT header
            if len(columns) >= 3:
                try: # Check if it's data already by trying to parse first column as float
                    float(columns[0].upper().replace('D','E'))
                    data_parse_state = 3 # If number, it's data, not units. Fall through to parse this line as data.
                except ValueError: # Not a number, so likely units
                    parsed_x_unit_original = columns[0].upper()
                    parsed_y_unit_original = columns[1].upper()
                    parsed_yerr_unit_original = columns[2].upper() if len(columns) > 2 else parsed_y_unit_original
                    data_parse_state = 3 # Data should start from the next line
                    continue
            else: # Line too short for units, assume data might start or format error
                data_parse_state = 3 # Try parsing data from next line
                continue


        if data_parse_state == 3: # Parsing numerical data
            if first_word == "ENDDATA": # End of current data block
                data_parse_state = 0 # Reset to look for more metadata or another DATA block
                if current_metadata_keyword_internal: # Finalize any pending metadata
                     out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()
                     current_metadata_keyword_internal = None
                continue

            # Avoid re-processing lines identified as metadata keywords, even if in data_parse_state 3
            if is_new_metadata_keyword_line:
                 continue

            if len(columns) >= 3:
                try:
                    x_val = float(columns[0].upper().replace('D','E'))
                    y_val = float(columns[1].upper().replace('D','E'))
                    yerr_val = float(columns[2].upper().replace('D','E'))

                    raw_x_values.append(x_val)
                    raw_y_values.append(y_val)
                    raw_yerr_values.append(yerr_val)
                except ValueError:
                    pass # Not a numerical data line, skip

    # Finalize any last metadata item if loop ended while accumulating
    if current_metadata_keyword_internal and out[current_metadata_keyword_internal]:
        out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()

    # --- Perform unit conversions and populate final dictionary values ---
    final_x_vals = list(raw_x_values)
    final_y_vals = list(raw_y_values)
    final_yerr_vals = list(raw_yerr_values)

    # Set default target units (can be overridden if conversion is not possible/applicable)
    current_target_x_unit = out['x-unit']
    current_target_y_unit = out['y-unit']
    current_target_yerr_unit = out['yerr-unit']

    if parsed_x_unit_original == 'MEV':
        final_x_vals = [val * 1e6 for val in raw_x_values]
    elif parsed_x_unit_original == 'KEV':
        final_x_vals = [val * 1e3 for val in raw_x_values]
    elif parsed_x_unit_original == 'EV':
        pass # No conversion needed, unit is already EV
    elif parsed_x_unit_original != "UNKNOWN":
        current_target_x_unit = parsed_x_unit_original # Use original if not convertible to EV

    # Y and Yerr conversion (assume yerr converts like y unless yerr_unit implies relative)
    # For minimal, assume yerr is absolute and follows y_unit's conversion path.
    if parsed_y_unit_original == 'MB': # Milli-barns
        final_y_vals = [val * 1e-3 for val in raw_y_values]
        final_yerr_vals = [val * 1e-3 for val in raw_yerr_values]
    elif parsed_y_unit_original == 'B': # Barns
        pass # No conversion needed
    elif parsed_y_unit_original == 'MICRO-B' or parsed_y_unit_original == 'UB':
        final_y_vals = [val * 1e-6 for val in raw_y_values]
        final_yerr_vals = [val * 1e-6 for val in raw_yerr_values]
    elif parsed_y_unit_original != "UNKNOWN":
        current_target_y_unit = parsed_y_unit_original
        # If y_unit is not converted to BARNS, yerr_unit should also reflect its original parsing.
        if parsed_yerr_unit_original != "UNKNOWN" and parsed_yerr_unit_original != parsed_y_unit_original :
             current_target_yerr_unit = parsed_yerr_unit_original
        else:
             current_target_yerr_unit = parsed_y_unit_original # Default to y's original unit

    out['x-vals'] = final_x_vals
    out['x-unit'] = current_target_x_unit
    out['y-vals'] = final_y_vals
    out['y-unit'] = current_target_y_unit
    out['yerr-vals'] = final_yerr_vals
    out['yerr-unit'] = current_target_yerr_unit

    # Ensure if y-unit is BARNS, yerr-unit is also BARNS (as per initial out dict spec)
    if out['y-unit'] == 'BARNS':
        out['yerr-unit'] = 'BARNS'

    return out

def calc_transmission(dataset):
    """
    Calculates the transmission error based on the cross section error
    using error propagation formula: dT = t * exp(-sigma*t) * dsigma
    """
    t = dataset['Processed']['thickness']
    sigma = np.array(dataset['EXFOR']['y-vals'])
    dsigma = np.array(dataset['EXFOR']['yerr-vals'])

    # Calculate transmission
    transmission = np.exp(-t * sigma)

    # Calculate transmission error using error propagation

    return transmission, t * transmission * dsigma

def process_dataset(datasets, label, thickness, temperature, abundances):
    """
    Adds processed information and calculates transmission for a given dataset.

    Args:
        datasets (dict): The main dictionary holding all datasets.
        label (str): The label (key) for the specific dataset to process.
        thickness (float): The thickness of the sample.
        temperature (int or float): The temperature of the sample.
        abundances (list): A list of isotopic abundances.
    """
    if label in datasets:
        # Ensure the 'Processed' key exists
        datasets[label]['Processed'] = {}
        datasets[label]['Processed']['thickness'] = thickness
        datasets[label]['Processed']['temperature'] = temperature
        datasets[label]['Processed']['abundances'] = abundances

        # Calculate transmission if EXFOR data is present
        if 'EXFOR' in datasets[label] and datasets[label]['EXFOR']['x-vals']: # Check if EXFOR data exists and is not empty
             t, terr = calc_transmission(datasets[label])
             datasets[label]['Processed']['pointwise T'] = t
             datasets[label]['Processed']['pointwise Terr'] = terr
        else:
             print(f"Warning: Cannot calculate transmission for {label}. EXFOR data missing or empty.")
             datasets[label]['Processed']['pointwise T'] = None
             datasets[label]['Processed']['pointwise Terr'] = None
    else:
        print(f"Warning: Dataset with label '{label}' not found.")

def process_data_into_bins(bin_edges, raw_x_ev, raw_y, raw_y_err):

    bin_bounds = []

    for lb, rb in zip(bin_edges[:-1], bin_edges[1:]):
        bin_bounds.append((lb, rb))

    x_mids = []
    y_values = []
    y_uncertainties = []

    for lb, rb in bin_bounds:

        bin_x_orig, bin_y_orig, bin_yerr_orig = cut_data(raw_x_ev, raw_y, raw_y_err, lb, rb)

        num_points_in_bin = len(bin_y_orig)

        avg_x_midpoint_ev = np.nan
        avg_y_value = np.nan
        avg_y_uncertainty = np.nan

        if num_points_in_bin > 1:
            try:
                avg_x_midpoint_ev, avg_y_value, avg_y_uncertainty = weighted_trapezoid(bin_x_orig, bin_y_orig, bin_yerr_orig)
            except Exception:
                pass
        elif num_points_in_bin == 1:
            avg_x_midpoint_ev = bin_x_orig[0]
            avg_y_value = bin_y_orig[0]
            avg_y_uncertainty = bin_yerr_orig[0]

        x_mids.append(float(avg_x_midpoint_ev))
        y_values.append(float(avg_y_value))
        y_uncertainties.append(float(avg_y_uncertainty))

    return x_mids, y_values, y_uncertainties

def analyze_std_dev_vs_bin_width_from_top(unsorted_x, unsorted_y, unsorted_yerr, E_i_upper, max_width, step_size):
    """
    Analyzes the standard deviation of original data within bins of increasing
    width ending at a specific energy E_i_upper.

    Args:
        unsorted_x (list or np.array): The unsorted X (energy) values of the data.
        unsorted_y (list or np.array): The unsorted Y (cross-section) values of the data.
        unsorted_yerr (list or np.array): The unsorted YERR (uncertainty) values of the data.
        E_i_upper (float): The fixed upper energy boundary (in the same units as x) for the bins.
        max_width (float): The maximum full bin width (in the same units as x) to test.
                           The lower boundary will range from E_i_upper down to E_i_upper - max_width.
        step_size (float): The increment size (in the same units as x) for increasing
                           the bin width (by decreasing the lower boundary).

    Returns:
        np.array: A 2D numpy array with shape (N, 2), where N is the number of steps.
                  Column 0 contains the full bin width (in the same units as x).
                  Column 1 contains the standard deviation of the original data
                  within that bin. Rows where std dev could not be calculated (e.g.,
                  < 2 points) will have NaN in the std dev column. The results are
                  ordered by increasing bin width.
    """
    # Sort the data by energy
    unsorted_x_np = np.array(unsorted_x)
    unsorted_y_np = np.array(unsorted_y)
    unsorted_yerr_np = np.array(unsorted_yerr)

    sorted_indices = np.argsort(unsorted_x_np)
    sorted_x = unsorted_x_np[sorted_indices]
    sorted_y = unsorted_y_np[sorted_indices]
    sorted_yerr = unsorted_yerr_np[sorted_indices]

    # Ensure E_i_upper is within the data range, otherwise adjust
    if E_i_upper > sorted_x[-1]:
        E_i_upper = sorted_x[-1]
        print(f"Warning: E_i_upper adjusted to highest data point: {E_i_upper/1e6:.3f} MeV")
    if E_i_upper < sorted_x[0]:
         print(f"Error: E_i_upper {E_i_upper/1e6:.3f} MeV is below the lowest data point {sorted_x[0]/1e6:.3f} MeV.")
         return np.array([]) # Return empty if E_i_upper is invalid

    # Define the target lower boundaries based on increasing width
    # Start with a very small width to get the first point
    target_widths = np.arange(step_size, max_width + step_size, step_size)

    results = []

    for target_width in target_widths:
        bin_end = E_i_upper
        # Calculate the target bin start based on the target width
        target_bin_start = E_i_upper - target_width

        # Ensure the bin start does not go below the lowest data point
        bin_start = max(sorted_x[0], target_bin_start)

        # If the calculated bin_start is equal to or greater than bin_end,
        # the bin is invalid or empty. This can happen if target_width is too small
        # or E_i_upper is near the start of data.
        if bin_start >= bin_end:
            # Add a point with NaN std dev and actual width if needed for plot continuity,
            # or just continue. Let's just continue as it simplifies the output.
            continue

        # Find indices corresponding to the bin boundaries in the sorted array
        start_idx = np.searchsorted(sorted_x, bin_start, side='left')
        end_idx = np.searchsorted(sorted_x, bin_end, side='right')

        # Select data within the indices
        bin_x_orig = sorted_x[start_idx:end_idx]
        bin_y_orig = sorted_y[start_idx:end_idx]
        bin_yerr_orig = sorted_yerr[start_idx:end_idx]

        num_points_in_bin = len(bin_y_orig)

        # Calculate the standard deviation of the original y values within the bin
        std_dev_original_in_bin = np.nan
        if num_points_in_bin >= 2:
            std_dev_original_in_bin = np.std(bin_y_orig)

        # Calculate the actual full bin width used
        actual_full_bin_width = bin_end - bin_start

        # Only append results for valid bins (at least one point, although std dev needs 2)
        # We include bins with 1 point so their width is represented, even if std dev is NaN
        if num_points_in_bin > 0:
             results.append([actual_full_bin_width, std_dev_original_in_bin])

    # Sort results by bin width
    results = np.array(results)
    if results.size > 0:
        results = results[np.argsort(results[:, 0])]

    return results

datasets = {}



datasets['Musgrove (1977) (80m)'] = {}
datasets['Musgrove (1977) (80m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=13736002')

datasets['Musgrove (1977) (200m)'] = {}
datasets['Musgrove (1977) (200m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=13736003')

datasets['Guenther (1974)'] = {}
datasets['Guenther (1974)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=10468002')

datasets['Green (1973)'] = {}
datasets['Green (1973)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=10225018')

# Define processing parameters for each dataset
dataset_parameters = {
    'Musgrove (1977) (200m)': {
        'thickness': 0.08271,
        'temperature': 294,
        'abundances': [0.9765, 0.0093, 0.0070, 0.0056, 0.0016]
    },
    'Musgrove (1977) (80m)': {
        'thickness': 0.08271,
        'temperature': 294,
        'abundances': [0.9765, 0.0093, 0.0070, 0.0056, 0.0016]
    },
    'Green (1973)': {
        'thickness': 0.0872,
        'temperature': 294,
        'abundances': [0.9772, 0.0107, 0.0051, 0.0056, 0.0015]
    },
    'Guenther (1974)': {
        'thickness': 0.0799,
        'temperature': 294,
        'abundances': [0.9772, 0.0107, 0.0051, 0.0056, 0.0015]
    },



datasets_zr91 = {}

datasets_zr91['Musgrove (1977) (80m)'] = {}
datasets_zr91['Musgrove (1977) (80m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=8620&subID=13741002')

datasets_zr91['Musgrove (1977) (200m)'] = {}
datasets_zr91['Musgrove (1977) (200m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=8620&subID=13741003')

zr91_parameters = {
    'Musgrove (1977) (200m)': {
        'thickness': 0.06423,
        'temperature': 294,
        'abundances': [0.058, 0.892, 0.0314, 0.0150, 0.0026]
    },
    'Musgrove (1977) (80m)': {
        'thickness': 0.06423,
        'temperature': 294,
        'abundances': [0.058, 0.892, 0.0314, 0.0150, 0.0026]
    },

  }
# Process each dataset using the new function
for label, params in dataset_parameters.items():
    process_dataset(datasets, label, params['thickness'], params['temperature'], params['abundances'])

labels = [
    'Musgrove (1977) (200m)',
    # 'Musgrove (1977) (80m)',
    'Green (1973)',
    'Guenther (1974)'
    ]
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

for label in labels:


  ax.errorbar(datasets[label]['EXFOR']['x-vals'],
              datasets[label]['EXFOR']['y-vals'],
              yerr=datasets[label]['EXFOR']['yerr-vals'],
              fmt='.', markersize=2, elinewidth=1, label=label)

#ax.set_yscale('log')
ax.set_xlabel('Incident Energy (MeV)') # Update label
ax.set_ylabel('Total Cross Section (barns)')
ax.set_title('Cross Section vs. Incident Energy')

# --- Apply FuncFormatter for X-axis ---
def eV_to_MeV_formatter(x, pos):
    """Converts tick value from eV to MeV and formats it."""
    return f'{x / 1e6:.2f}' # Format to one decimal place, e.g., 0.2, 0.3

ax.xaxis.set_major_formatter(ticker.FuncFormatter(eV_to_MeV_formatter))

# Adjust x-limits (still in original eV scale for data)
ax.set_xlim(2e5-0.1e5, 2e6)
ax.set_ylim(3,30)
ax.set_yscale("log")

# Add the vertical dashed lines (still using eV for their positions)
#eV_lines = [250000, 300000, 350000, 400000, 450000, 500000, 550000]

eV_line = 250000
while eV_line < 2e6:
    ax.axvline(x=eV_line, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    eV_line += 50000
# for ev_val in eV_lines:
#     ax.axvline(ev_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

# Ensure the horizontal grid is not affected by x-axis formatting changes if it's only for y
ax.grid(True, axis='y', which='major', linestyle='--', color='#cccccc', alpha=0.7) # Major Y grid
ax.grid(True, axis='y', which='minor', linestyle=':', color='#dddddd', alpha=0.5) # Minor Y grid (optional)
ax.legend()

plt.tight_layout()
plt.show()

# @title
# Choose the specific dataset to analyze
dataset_label = 'Musgrove (1977) (200m)' # Using the 200m data as requested
dataset_label = 'Green (1973)'

# Get the original data for the chosen dataset
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

# Sort the data by energy if it's not already
sorted_indices = np.argsort(original_x)
sorted_x = np.array(original_x)[sorted_indices]
sorted_y = np.array(original_y)[sorted_indices]
sorted_yerr = np.array(original_yerr)[sorted_indices]

# Define a list of different bin widths (in eV) to try
# You can adjust these widths to experiment
bin_widths_ev = [25000, 50000] # Example bin widths

# Store binned results for each bin width
all_binned_results = {}

for bin_width in bin_widths_ev:
    binned_results_for_width = []
    # Define the start and end of the first bin based on the sorted data range
    current_bin_start = sorted_x[0]
    current_bin_end = current_bin_start + bin_width

    # Iterate through the data to create and process bins
    # We need to ensure the last bin goes at least up to the last data point
    while current_bin_start <= sorted_x[-1]:

        # Use the cut_data function to get points within the current bin
        bin_x, bin_y, bin_yerr = cut_data(sorted_x, sorted_y, sorted_yerr, current_bin_start, current_bin_end)

        if len(bin_x) > 1: # Need at least two points to form a trapezoid
            try:
                # Calculate the weighted average for the bin
                avg_x, avg_y, avg_yerr = weighted_trapezoid(bin_x, bin_y, bin_yerr)
                # Store the result
                binned_results_for_width.append({
                    'avg_x': avg_x,
                    'avg_y': avg_y,
                    'avg_yerr': avg_yerr,
                    'bin_start': current_bin_start,
                    'bin_end': current_bin_end
                })
            except ValueError as e:
                 # print(f"Skipping bin [{current_bin_start}, {current_bin_end}] for width {bin_width/1e3:.0f} keV due to error: {e}")
                 # Optionally, you could append None or a placeholder
                 pass # Skip bins with errors for plotting this way

        # Move to the next bin
        current_bin_start = current_bin_end
        current_bin_end = current_bin_start + bin_width

    all_binned_results[bin_width] = binned_results_for_width

# Now, plot the original data and the binned results for each bin width
fig, axs = plt.subplots(nrows=len(bin_widths_ev), figsize=(10, 3 * (len(bin_widths_ev)+1)), sharex=True, dpi=200)

# # Plot the original data on the top subplot
ax_original = axs[0]
ax_original.errorbar(sorted_x, sorted_y, yerr=sorted_yerr,
                     fmt='.', markersize=2, elinewidth=1, alpha=0.7, label=f'{dataset_label} (Original Data)')
ax_original.set_ylabel('Cross Section (barns)')
ax_original.set_title(f'{dataset_label} Total Cross Section vs. Incident Energy')
ax_original.grid(True, which='both', linestyle='--', color='#cccccc', alpha=0.7)
ax_original.legend()


# Plot binned data for each bin width on subsequent subplots
for i, bin_width in enumerate(bin_widths_ev):
    ax = axs[i + 0]
    binned_data = all_binned_results[bin_width]

    if binned_data: # Only plot if there are results for this bin width
        binned_x = [res['avg_x'] for res in binned_data]
        binned_y = [res['avg_y'] for res in binned_data]
        binned_yerr = [res['avg_yerr'] for res in binned_data]

        ax.errorbar(binned_x, binned_y, yerr=binned_yerr,
                    fmt='.', markersize=5, elinewidth=1.5, capsize=3, label=f'Bin Width: {bin_width/1e3:.0f} keV') # Label in keV

        # Add vertical lines for bin edges
        for res in binned_data:
             ax.axvline(x=res['bin_start'], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        # Add the last bin end edge line
        if binned_data:
             ax.axvline(x=binned_data[-1]['bin_end'], color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

    ax.set_ylabel('Cross Section (barns)')
    ax.grid(True, which='both', linestyle='--', color='#cccccc', alpha=0.7)
    ax.legend()

# Set common x-axis label on the bottom subplot
axs[-1].set_xlabel('Incident Energy (MeV)')

# Apply the MeV formatter to all x-axes (since sharex=True)
def eV_to_MeV_formatter(x, pos):
    """Converts tick value from eV to MeV and formats it."""
    if isinstance(x, (int, float)):
        return f'{x / 1e6:.2f}'
    return str(x)

for ax in axs:
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(eV_to_MeV_formatter))

# Optional: Set common x-limits based on the data range
if sorted_x.size > 0:
    for ax in axs:
        #ax.set_xlim(sorted_x[0] * 0.95, sorted_x[-1] * 1.05)
        ax.set_xlim(5e5-0.1e5, 1.2e6)
        ax.set_ylim(3,30)
        ax.set_yscale("log")


plt.tight_layout()
plt.show()

#for item in all_binned_results[10000]:
  #print("[", item['avg_x'], ", ", item['avg_y'], ", ", item['avg_yerr'], "], ")
  #print(item['avg_x'], ", ")
  #print(item['avg_x'], ",   ", item['avg_y'], ",   ", item['avg_yerr'])

bin_edges = [5.5e5+5.0e4*i for i in range(44)]
#print(bin_edges)

energies = ""
trans = ""
trans_err = ""

label = 'Green (1973)'
original_x = datasets[label]['EXFOR']['x-vals']
original_y = datasets[label]['Processed']['pointwise T']
original_yerr = datasets[label]['Processed']['pointwise Terr']

xvals, yvals, yerr = process_data_into_bins(bin_edges, original_x, original_y, original_yerr)

for i in range(len(xvals)):
  energies = energies + f"{xvals[i]:.3e},    "
  trans = trans + f"{yvals[i]:.4f},    "
  trans_err = trans_err + f"{yerr[i]:.4f},    "

  print(f"{xvals[i]:.3e}    {yvals[i]:.4f}    {yerr[i]:.4f}")

#print(energies)
#print(trans)
#print(trans_err)

#print(

# print(energies)
# print(trans)
# print(trans_err)


energies = ""
xss = ""
xs_errs = ""
for i, T in enumerate(datasets['Green (1973)']['Processed']['pointwise T']):
  energy = datasets['Green (1973)']['EXFOR']['x-vals'][i]

  if energy > 1.9e6: continue
  Terr = datasets['Green (1973)']['Processed']['pointwise Terr'][i]
  xs_val = datasets['Green (1973)']['EXFOR']['y-vals'][i]
  xs_err = datasets['Green (1973)']['EXFOR']['yerr-vals'][i]

  energies = energies + f"{energy:.3e},    "
  xss = xss + f"{xs_val:.4f},    "
  xs_errs = xs_errs + f"{xs_err:.4f},    "

  #print(f"{energy:.3e}", "")

# print(energies)
# print(xss)
# print(xs_errs)

  #print(f"{energy:.3e},    {T:.4f},    {Terr:.4f}")

# print(energies)
# print(trans)
# print(trans_err)
# #datasets['Green (1973)']['Processed']['pointwise T']

# Choose the specific dataset to analyze
dataset_label = 'Musgrove (1977) (200m)' # Using the 200m data

# Get the original data for the chosen dataset
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

# Sort the data by energy
sorted_indices = np.argsort(original_x)
sorted_x = np.array(original_x)[sorted_indices]
sorted_y = np.array(original_y)[sorted_indices]
sorted_yerr = np.array(original_yerr)[sorted_indices]

# Define the center energies (in eV) you want to analyze
# Choose energies where you might expect different behavior (e.g., near resonances, in valleys)
center_energies_ev = [3e5, 5e5, 7e5, 9e5, 11e5, 13e5, 15e5, 17e5, 19e5] # Example center energies (300 keV, 500 keV, 800 keV, 1.2 MeV)

#center_energies_ev = np.linspace(3e5, 1.8e6, num=12) # np.arange(3e5, 1.2e6, 1e4)
# Define a list of increasing bin half-widths (in eV)
# The full bin width will be 2 * half_width
half_widths_ev = np.arange(1000, 150000, 5000) # Example half-widths from 5 keV to 300 keV

# Store results for each center energy
variance_vs_width_results = {}

for center_energy in center_energies_ev:
    variance_vs_width_results[center_energy] = []

    for half_width in half_widths_ev:
        bin_start = center_energy - half_width
        bin_end = center_energy + half_width

        # Ensure the bin is within the range of the original data
        if bin_start < sorted_x[0]:
            bin_start = sorted_x[0]
        if bin_end > sorted_x[-1]:
            bin_end = sorted_x[-1]

        # Use cut_data to get points within this symmetric bin
        bin_x_orig, bin_y_orig, bin_yerr_orig = cut_data(sorted_x, sorted_y, sorted_yerr, bin_start, bin_end)

        # Calculate the standard deviation of the original y values within the bin
        # Need at least 2 points to calculate standard deviation
        std_dev_original_in_bin = np.std(bin_y_orig) if len(bin_y_orig) > 1 else np.nan # Use NaN if not enough points

        avg_x, avg_y, avg_yerr = weighted_trapezoid(bin_x_orig, bin_y_orig, bin_yerr_orig) if len(bin_y_orig) > 1 else (np.nan, np.nan, np.nan)

        rel_std_dev_in_bin = std_dev_original_in_bin / avg_y if avg_y != 0 else np.nan

        # Calculate the average cross section for this bin (optional, but good for context)
        #avg_y = np.mean(bin_y_orig) if len(bin_y_orig) > 0 else np.nan # Mean of points, not weighted avg

        variance_vs_width_results[center_energy].append({
            'half_width': half_width,
            'full_width': 2 * half_width,
            'std_dev': rel_std_dev_in_bin*100,
            'average_y': avg_y, # Average of the raw points
            'bin_start': bin_start,
            'bin_end': bin_end,
            'num_points': len(bin_y_orig)
        })

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    # --- Tick Settings ---
    'xtick.direction': 'in',             # X-axis major ticks direction
    'ytick.direction': 'in',             # Y-axis major ticks direction
    'xtick.major.size': 6,               # X-axis major tick length
    'ytick.major.size': 6,               # Y-axis major tick length
    'xtick.minor.size': 3,               # X-axis minor tick length
    'ytick.minor.size': 3,               # Y-axis minor tick length
    'xtick.minor.visible': False,         # Display X-axis minor ticks
    'ytick.minor.visible': False,         # Display Y-axis minor ticks
    'xtick.top': True,                   # Display X-axis ticks on top
    'ytick.right': True,                 # Display Y-axis ticks on right

    # --- Grid Settings ---
    'axes.grid': True,                   # Display grid
    'axes.grid.which': 'major',          # Grid on major ticks
    'grid.color': 'gray',                # Grid color
    'grid.linestyle': '--',              # Grid line style
    'grid.alpha': 0.7,                   # Grid transparency

    # --- Font Settings (from previous discussions) ---
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],   # Or 'STIXGeneral', 'Georgia', etc.
    'mathtext.fontset': 'stix',          # Serif math font

    # --- Optional: Font sizes ---
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


#from matplotlib.collections import Poly3DCollection # For the slice effect

# Assume 'center_energies_ev' and 'variance_vs_width_results' are pre-defined
# Example Data (replace with your actual data)
# center_energies_ev = [100000, 150000, 200000] # Example: 100 keV, 150 keV, 200 keV
# variance_vs_width_results = {
#     100000: [{'full_width': 1000, 'std_dev': 0.5}, {'full_width': 2000, 'std_dev': 0.4}, {'full_width': 3000, 'std_dev': 0.6}],
#     150000: [{'full_width': 1000, 'std_dev': 0.7}, {'full_width': 2000, 'std_dev': 0.65}, {'full_width': 3000, 'std_dev': 0.8}],
#     200000: [{'full_width': 1000, 'std_dev': 0.9}, {'full_width': 2000, 'std_dev': 0.85}, {'full_width': 3000, 'std_dev': 1.0}],
# }
# # Ensure 'std_dev' can be NaN as in your original code
# variance_vs_width_results[100000].append({'full_width': 4000, 'std_dev': np.nan})


# --- Configuration for Styling ---
plot_style_config = {
    'fig_face_color': 'white',
    'axis_face_color': 'white',
    'label_color': 'black',
    'tick_color': 'black',
    'spine_color': 'black',
    'line_color': 'tab:blue',
    'slice_fill_color': '#E0E0E0',  # Light gray for the filled slice
    'marker_size': 3,
    'line_style': '-',
    'label_pad': 10, # Adjust for desired label spacing
    'z_label_pad': 5, # Z label might need more padding
}

# --- 1. Data Preparation ---
# Get sorted unique center energies in MeV for Y-axis ticks and iteration
unique_center_energies_mev = sorted(list(set(ce / 1e3 for ce in center_energies_ev))) # Assuming 1e6 for MeV from your label

# --- Create the 3D plot ---
fig = plt.figure(figsize=(16, 12), dpi=200) # Increased figure size for clarity
ax = fig.add_subplot(111, projection='3d')

# Set face colors
fig.set_facecolor(plot_style_config['fig_face_color'])
ax.set_facecolor(plot_style_config['axis_face_color'])

# To determine global plot limits for consistent slice appearance if needed
all_plot_widths_kev = []
all_plot_std_devs = []

# --- Loop through each center energy to plot slices and lines ---
# Iterate in sorted order of energies for consistent layering if plots overlap
sorted_center_energies_ev = sorted(center_energies_ev)
#reversed(sorted_center_energies_ev)

for center_energy in sorted_center_energies_ev:
    results_for_energy = variance_vs_width_results[center_energy]

    # Sort results by 'full_width' to ensure polygons are drawn correctly
    results_for_energy.sort(key=lambda r: r['full_width'])

    # Extract data, filtering out NaN standard deviations
    current_widths_kev = []
    current_std_devs = []
    for res in results_for_energy:
        if not np.isnan(res['std_dev']):
            current_widths_kev.append(res['full_width'] / 1e3)
            current_std_devs.append(res['std_dev'])

    if not current_widths_kev: # Skip if no valid data for this energy
        continue

    plot_widths_kev = np.array(current_widths_kev)
    plot_std_devs = np.array(current_std_devs)

    all_plot_widths_kev.extend(plot_widths_kev)
    all_plot_std_devs.extend(plot_std_devs)

    # Create a constant array for the center energy for this line (in MeV)
    plot_center_energy_mev_val = center_energy / 1e3 # Consistent with unique_center_energies_mev

    # --- 4. "Slice Effect": Filled polygon under the curve ---
    # Define the base of the filled area (e.g., z=0 or slightly below min data)
    z_base = 0 # Assuming standard deviations are non-negative

    if len(plot_widths_kev) > 1: # Need at least 2 points to form a polygon
        # Vertices for the polygon:
        # Start at (x_first, z_base), go up to (x_first, z_first),
        # then along the curve (x_i, z_i),
        # then down to (x_last, z_base), and back to start.
        verts = []
        # Bottom edge from left to right
        verts.append((plot_widths_kev[0], plot_center_energy_mev_val, z_base))
        # Along the data curve
        for i in range(len(plot_widths_kev)):
            verts.append((plot_widths_kev[i], plot_center_energy_mev_val, plot_std_devs[i]))
        # Bottom edge from right to left (back to start)
        verts.append((plot_widths_kev[-1], plot_center_energy_mev_val, z_base))

        poly = Poly3DCollection([verts],
                                facecolors=plot_style_config['slice_fill_color'],
                                edgecolors=plot_style_config['line_color'], # Optional: add edge to fill
                                linewidths=0.5,
                                alpha=0.9) # Adjust alpha for transparency
        ax.add_collection3d(poly)

    # Plot the actual data line on top
    ax.plot(plot_widths_kev,
            np.full_like(plot_widths_kev, plot_center_energy_mev_val), # Y-values are constant for this slice
            plot_std_devs,
            marker='.',
            markersize=plot_style_config['marker_size'],
            linestyle=plot_style_config['line_style'],
            color=plot_style_config['line_color'],
           )

# --- 3. Axis Configuration ---
# Set labels
ax.set_xlabel(r'$\Delta E$ (keV)', color=plot_style_config['label_color'], labelpad=plot_style_config['label_pad'])
ax.set_ylabel(r'$E_{mid}$ (keV)', color=plot_style_config['label_color'], labelpad=plot_style_config['label_pad']) # Changed to MeV to match data
#ax.set_zlabel(r'$\frac{\Delta \sigma_{tot}}{\sigma_{tot}}$ (percent)', color=plot_style_config['label_color'], labelpad=plot_style_config['z_label_pad']) # Simplified label
ax.set_zlabel('')
text_label = r'$\frac{\Delta \sigma_{tot}}{\sigma_{tot}} \: (\%)$'
x_pos = -30
y_pos = 0
z_pos = 35
ax.text(x_pos, y_pos, z_pos, text_label, fontsize=16, ha='center', va='center', rotation='vertical')

# Set tick colors
ax.tick_params(axis='x', colors=plot_style_config['tick_color'])
ax.tick_params(axis='y', colors=plot_style_config['tick_color'])
ax.tick_params(axis='z', colors=plot_style_config['tick_color'])

# Set spine colors (the lines of the axes box)
ax.xaxis.line.set_color(plot_style_config['spine_color'])
ax.yaxis.line.set_color(plot_style_config['spine_color'])
ax.zaxis.line.set_color(plot_style_config['spine_color'])

ax.set_zlim(0,70)
ax.set_ylim(300,1900)
ax.set_xlim(1,290)

ax.set_yticks(unique_center_energies_mev)


ax.set_yticklabels([f'{val:.0f}' for val in unique_center_energies_mev]) # Format tick labels

# Axis Proportions: Make <span class="math-inline">E\_\{mid\}</span> (Y) axis appear longer than <span class="math-inline">\\Delta E</span> (X) axis
# This depends on your data ranges. (ratio_x, ratio_y, ratio_z)
# For Y to be longer, give it a larger ratio relative to X.
# Calculate appropriate ratios based on data range, or set manually.
if all_plot_widths_kev and unique_center_energies_mev and all_plot_std_devs:
    x_range = np.ptp(all_plot_widths_kev) if np.ptp(all_plot_widths_kev) > 0 else 1
    y_range = np.ptp(unique_center_energies_mev) if np.ptp(unique_center_energies_mev) > 0 else 1
    z_range = np.ptp(all_plot_std_devs) if np.ptp(all_plot_std_devs) > 0 else 1

    # Aim for Y to be visually 1.5x longer than X, Z scaled by its own data range
    # You might need to adjust these factors based on your specific data and preference
    #y_emphasis_factor = 100.5
    #ax.set_box_aspect((x_range, y_range * y_emphasis_factor, z_range))
    ax.set_box_aspect((1, 2, 1))
else: # Default if no data
    ax.set_box_aspect((1, 1, 1)) # Default: Y axis 1.5 times longer

# --- 2. Styling for "Black and White Vibe" (Panes and Grid) ---
# Remove pane fills
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set pane edge colors to form the bounding box
ax.xaxis.pane.set_edgecolor(plot_style_config['spine_color'])
ax.yaxis.pane.set_edgecolor(plot_style_config['spine_color'])
ax.zaxis.pane.set_edgecolor(plot_style_config['spine_color'])

# Turn off grid lines
ax.grid(True)

# --- 5. View Angle ---
# Adjust view angle to be more like the example (E_mid receding to back-right)
# Original: elev=10, azim=135
# Try: elev=25, azim=-45 (or azim=315)
ax.view_init(elev=15, azim=110)

# Optional: Add a legend (can be tricky in 3D, might need to adjust its position)
# If you used labels in the ax.plot:
# handles, labels = ax.get_legend_handles_labels()
# if handles:
#     ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, title="E_mid (MeV)")

# Adjust layout
#plt.tight_layout(pad=2.0) # Add some padding

ax.tick_params(axis='y', rotation=-10, pad=0)
ax.tick_params(axis='x', rotation=-20, pad=-2)
ax.tick_params(axis='z', rotation=0, pad=5)



plt.show()

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    # --- Tick Settings ---
    'xtick.direction': 'in',             # X-axis major ticks direction
    'ytick.direction': 'in',             # Y-axis major ticks direction
    'xtick.major.size': 6,               # X-axis major tick length
    'ytick.major.size': 6,               # Y-axis major tick length
    'xtick.minor.size': 3,               # X-axis minor tick length
    'ytick.minor.size': 3,               # Y-axis minor tick length
    'xtick.minor.visible': True,         # Display X-axis minor ticks
    'ytick.minor.visible': False,         # Display Y-axis minor ticks
    'xtick.top': True,                   # Display X-axis ticks on top
    'ytick.right': True,                 # Display Y-axis ticks on right

    # --- Grid Settings ---
    'axes.grid': True,                   # Display grid
    'axes.grid.which': 'major',          # Grid on major ticks
    'grid.color': 'gray',                # Grid color
    'grid.linestyle': '--',              # Grid line style
    'grid.alpha': 0.7,                   # Grid transparency

    # --- Font Settings (from previous discussions) ---
    'font.family': 'serif',
    'font.serif': ['STIXGeneral'],   # Or 'STIXGeneral', 'Georgia', etc.
    'mathtext.fontset': 'stix',          # Serif math font

    # --- Optional: Font sizes ---
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


#from matplotlib.collections import Poly3DCollection # For the slice effect

# Assume 'center_energies_ev' and 'variance_vs_width_results' are pre-defined
# Example Data (replace with your actual data)
# center_energies_ev = [100000, 150000, 200000] # Example: 100 keV, 150 keV, 200 keV
# variance_vs_width_results = {
#     100000: [{'full_width': 1000, 'std_dev': 0.5}, {'full_width': 2000, 'std_dev': 0.4}, {'full_width': 3000, 'std_dev': 0.6}],
#     150000: [{'full_width': 1000, 'std_dev': 0.7}, {'full_width': 2000, 'std_dev': 0.65}, {'full_width': 3000, 'std_dev': 0.8}],
#     200000: [{'full_width': 1000, 'std_dev': 0.9}, {'full_width': 2000, 'std_dev': 0.85}, {'full_width': 3000, 'std_dev': 1.0}],
# }
# # Ensure 'std_dev' can be NaN as in your original code
# variance_vs_width_results[100000].append({'full_width': 4000, 'std_dev': np.nan})


# --- Configuration for Styling ---
plot_style_config = {
    'fig_face_color': 'white',
    'axis_face_color': 'white',
    'label_color': 'black',
    'tick_color': 'black',
    'spine_color': 'black',
    'line_color': 'tab:blue',
    'slice_fill_color': '#E0E0E0',  # Light gray for the filled slice
    'marker_size': 3,
    'line_style': '-',
    'label_pad': 10, # Adjust for desired label spacing
    'z_label_pad': 15 # Z label might need more padding
}

# --- 1. Data Preparation ---
# Get sorted unique center energies in MeV for Y-axis ticks and iteration
unique_center_energies_mev = sorted(list(set(ce / 1e3 for ce in center_energies_ev))) # Assuming 1e6 for MeV from your label

# --- Create the 3D plot ---
fig = plt.figure(figsize=(16, 12)) # Increased figure size for clarity
ax = fig.add_subplot(111, projection='3d')

# Set face colors
fig.set_facecolor(plot_style_config['fig_face_color'])
ax.set_facecolor(plot_style_config['axis_face_color'])

# To determine global plot limits for consistent slice appearance if needed
all_plot_widths_kev = []
all_plot_std_devs = []

# --- Loop through each center energy to plot slices and lines ---
# Iterate in sorted order of energies for consistent layering if plots overlap
sorted_center_energies_ev = sorted(center_energies_ev)
#reversed(sorted_center_energies_ev)

for center_energy in sorted_center_energies_ev:
    results_for_energy = variance_vs_width_results[center_energy]

    # Sort results by 'full_width' to ensure polygons are drawn correctly
    results_for_energy.sort(key=lambda r: r['full_width'])

    # Extract data, filtering out NaN standard deviations
    current_widths_kev = []
    current_std_devs = []
    for res in results_for_energy:
        if not np.isnan(res['std_dev']):
            current_widths_kev.append(res['full_width'] / 1e3)
            current_std_devs.append(res['std_dev'])

    if not current_widths_kev: # Skip if no valid data for this energy
        continue

    plot_widths_kev = np.array(current_widths_kev)
    plot_std_devs = np.array(current_std_devs)

    all_plot_widths_kev.extend(plot_widths_kev)
    all_plot_std_devs.extend(plot_std_devs)

    # Create a constant array for the center energy for this line (in MeV)
    plot_center_energy_mev_val = center_energy / 1e3 # Consistent with unique_center_energies_mev

    # --- 4. "Slice Effect": Filled polygon under the curve ---
    # Define the base of the filled area (e.g., z=0 or slightly below min data)
    z_base = 0 # Assuming standard deviations are non-negative

    if len(plot_widths_kev) > 1: # Need at least 2 points to form a polygon
        # Vertices for the polygon:
        # Start at (x_first, z_base), go up to (x_first, z_first),
        # then along the curve (x_i, z_i),
        # then down to (x_last, z_base), and back to start.
        verts = []
        # Bottom edge from left to right
        verts.append((plot_widths_kev[0], plot_center_energy_mev_val, z_base))
        # Along the data curve
        for i in range(len(plot_widths_kev)):
            verts.append((plot_widths_kev[i], plot_center_energy_mev_val, plot_std_devs[i]))
        # Bottom edge from right to left (back to start)
        verts.append((plot_widths_kev[-1], plot_center_energy_mev_val, z_base))

        poly = Poly3DCollection([verts],
                                facecolors=plot_style_config['slice_fill_color'],
                                edgecolors=plot_style_config['line_color'], # Optional: add edge to fill
                                linewidths=0.5,
                                alpha=0.7) # Adjust alpha for transparency
        ax.add_collection3d(poly)

    # Plot the actual data line on top
    ax.plot(plot_widths_kev,
            np.full_like(plot_widths_kev, plot_center_energy_mev_val), # Y-values are constant for this slice
            plot_std_devs,
            marker='.',
            markersize=plot_style_config['marker_size'],
            linestyle=plot_style_config['line_style'],
            color=plot_style_config['line_color'],
           )

# --- 3. Axis Configuration ---
# Set labels
ax.set_xlabel(r'$\Delta E$ (keV)', color=plot_style_config['label_color'], labelpad=plot_style_config['label_pad'])
ax.set_ylabel(r'$E_{mid}$ (keV)', color=plot_style_config['label_color'], labelpad=plot_style_config['label_pad']) # Changed to MeV to match data
ax.set_zlabel(r'$\sigma_{tot}$ Standard Deviation (barns)', color=plot_style_config['label_color'], labelpad=plot_style_config['z_label_pad']) # Simplified label

# Set tick colors
ax.tick_params(axis='x', colors=plot_style_config['tick_color'])
ax.tick_params(axis='y', colors=plot_style_config['tick_color'])
ax.tick_params(axis='z', colors=plot_style_config['tick_color'])

# Set spine colors (the lines of the axes box)
ax.xaxis.line.set_color(plot_style_config['spine_color'])
ax.yaxis.line.set_color(plot_style_config['spine_color'])
ax.zaxis.line.set_color(plot_style_config['spine_color'])

ax.set_zlim(0,1)
ax.set_ylim(300,1900)
ax.set_xlim(1,290)

ax.set_yticks(unique_center_energies_mev)


ax.set_yticklabels([f'{val:.0f}' for val in unique_center_energies_mev]) # Format tick labels

# Axis Proportions: Make <span class="math-inline">E\_\{mid\}</span> (Y) axis appear longer than <span class="math-inline">\\Delta E</span> (X) axis
# This depends on your data ranges. (ratio_x, ratio_y, ratio_z)
# For Y to be longer, give it a larger ratio relative to X.
# Calculate appropriate ratios based on data range, or set manually.
if all_plot_widths_kev and unique_center_energies_mev and all_plot_std_devs:
    x_range = np.ptp(all_plot_widths_kev) if np.ptp(all_plot_widths_kev) > 0 else 1
    y_range = np.ptp(unique_center_energies_mev) if np.ptp(unique_center_energies_mev) > 0 else 1
    z_range = np.ptp(all_plot_std_devs) if np.ptp(all_plot_std_devs) > 0 else 1

    # Aim for Y to be visually 1.5x longer than X, Z scaled by its own data range
    # You might need to adjust these factors based on your specific data and preference
    #y_emphasis_factor = 100.5
    #ax.set_box_aspect((x_range, y_range * y_emphasis_factor, z_range))
    ax.set_box_aspect((1, 2, 1))
else: # Default if no data
    ax.set_box_aspect((1, 1, 1)) # Default: Y axis 1.5 times longer

# --- 2. Styling for "Black and White Vibe" (Panes and Grid) ---
# Remove pane fills
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Set pane edge colors to form the bounding box
ax.xaxis.pane.set_edgecolor(plot_style_config['spine_color'])
ax.yaxis.pane.set_edgecolor(plot_style_config['spine_color'])
ax.zaxis.pane.set_edgecolor(plot_style_config['spine_color'])

# Turn off grid lines
ax.grid(True)

# --- 5. View Angle ---
# Adjust view angle to be more like the example (E_mid receding to back-right)
# Original: elev=10, azim=135
# Try: elev=25, azim=-45 (or azim=315)
ax.view_init(elev=15, azim=110)

# Optional: Add a legend (can be tricky in 3D, might need to adjust its position)
# If you used labels in the ax.plot:
# handles, labels = ax.get_legend_handles_labels()
# if handles:
#     ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False, title="E_mid (MeV)")

# Adjust layout
plt.tight_layout(pad=5.0) # Add some padding
plt.show()

# --- Example Usage ---
dataset_label = 'Musgrove (1977) (200m)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

# Choose an upper energy boundary and parameters for analysis
E_upper = max(datasets[dataset_label]['EXFOR']['x-vals']) # eV (0.8 MeV)

print(E_upper)
E_upper = E_upper -1.1e5
print(E_upper)
E_upper = E_upper -0.8e5
print(E_upper)
E_upper = E_upper -0.55e5
print(E_upper)
E_upper = E_upper -1.8e5
print(E_upper)
E_upper = E_upper -1.1e5
# E_upper = E_upper -1.2e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.1e5
# E_upper = E_upper -0.9e5
# E_upper = E_upper -0.9e5
#E_upper = E_upper -1.9e5
#E_upper = E_upper -1.7e5
max_bin_width = 5e5 # eV (Test bins up to 800 keV width ending at E_upper)
step_increment = 5000 # eV (5 keV steps in width)

# Analyze standard deviation vs. bin width from the top down
std_dev_data_from_top_0p8MeV = analyze_std_dev_vs_bin_width_from_top(
    original_x, original_y, original_yerr, E_upper, max_bin_width, step_increment)

# Separate the results into width and std dev arrays
bin_widths_from_top_0p8MeV_kev = std_dev_data_from_top_0p8MeV[:, 0] / 1e3 # Convert width to keV
std_devs_from_top_0p8MeV = std_dev_data_from_top_0p8MeV[:, 1]

# Filter out NaNs for plotting (optional, but often makes line plots cleaner)
valid_indices = ~np.isnan(std_devs_from_top_0p8MeV)
plot_widths_from_top_0p8MeV_kev = bin_widths_from_top_0p8MeV_kev[valid_indices]
plot_std_devs_from_top_0p8MeV = std_devs_from_top_0p8MeV[valid_indices]


# Plotting the results for this specific upper energy boundary
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(plot_widths_from_top_0p8MeV_kev, plot_std_devs_from_top_0p8MeV, 'o-', markersize=4)

ax.set_xlabel('Full Bin Width (keV)')
ax.set_ylabel('Standard Deviation of Original Data (barns)')
ax.set_title(f'Std Dev vs. Bin Width Ending at {E_upper/1e6:.2f} MeV')

ax.grid(True, which='both', linestyle='--', color='#cccccc', alpha=0.7)
plt.tight_layout()
plt.show()

subtractions_ev = [1.4e5, 2.1e5, 2.8e5, 1.1e5, 1.2e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.1e5, 0.9e5, 0.9e5, 1.9e5, 1.7e5]
E_max = 2349100.0

bin_edges = [E_max]
for sub in subtractions_ev:
    E_max -= sub
    bin_edges.append(E_max)

bin_edges.append(0)

print(bin_edges)
bin_edges = list(reversed(bin_edges))

print(bin_edges)

label = 'Musgrove (1977) (200m)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

xvals, yvals, yerr = process_data_into_bins(bin_edges, original_x, original_y, original_yerr)


datasets[dataset_label]['Processed']['x-vals'] = xvals
datasets[dataset_label]['Processed']['xs-vals'] = yvals
datasets[dataset_label]['Processed']['yerr-vals'] = yerr

fig, ax = plt.subplots(figsize=(7, 5), dpi=200)

# Plot original data
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']
ax.errorbar(original_x, original_y, yerr=original_yerr,
              fmt='.', markersize=2, elinewidth=1, alpha=0.5, label='EXFOR')

# Plot processed (binned) data
processed_x = datasets[dataset_label]['Processed']['x-vals']
processed_y = datasets[dataset_label]['Processed']['xs-vals']
processed_yerr = datasets[dataset_label]['Processed']['yerr-vals']
ax.errorbar(processed_x, processed_y, yerr=processed_yerr,
              fmt='o', markersize=6, elinewidth=1.5, capsize=3, label='Averaged Data')


# Apply formatting (assuming you have the eV_to_MeV_formatter defined)
def eV_to_MeV_formatter(x, pos):
    """Converts tick value from eV to MeV and formats it."""
    if isinstance(x, (int, float)):
        return f'{x / 1e6:.2f}'
    return str(x)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(eV_to_MeV_formatter))
ax.set_yscale('log')
ax.set_ylim(1, 150)
ax.set_xlabel('Incident Energy (MeV)')
ax.set_ylabel('Total Cross Section (barns)')
ax.set_title(f'Original vs. Average Cross Section for {dataset_label}')
ax.grid(True, which='both', linestyle='--', color='#cccccc', alpha=0.7)
ax.legend()
plt.tight_layout()
plt.show()

dataset_label = 'Musgrove (1977) (200m)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']


subtractions_ev = [1.4e5, 2.1e5, 2.8e5, 1.1e5, 1.2e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.0e5, 1.1e5, 0.9e5, 0.9e5, 1.9e5, 1.7e5]
E_max = 2349100.0

bin_edges = [E_max]
for sub in subtractions_ev:
    E_max -= sub
    bin_edges.append(E_max)

bin_edges.append(0)

#print(bin_edges)
bin_edges = list(reversed(bin_edges))
xvals, yvals, yerr = process_data_into_bins(bin_edges, original_x, original_y, original_yerr)
trans, trans_err = calc_transmission(datasets[dataset_label])

xvals, Tvals, terr = process_data_into_bins(bin_edges, original_x, trans, trans_err)
datasets[dataset_label]['Processed']['x-vals'] = xvals
datasets[dataset_label]['Processed']['T-vals'] = Tvals
datasets[dataset_label]['Processed']['T-err'] = terr

print("Musgrove 200m (1977)")
print("UNCERTAINTY    ABSOLUTE")
print("THICKNESS      0.0827")
print("TEMPERATURE    293.0")
print("ABUNDANCES     0.9765 0.0093 0.0070 0.0056 0.0016\n")
for energy, T, terr in zip(xvals, Tvals, terr):
    print(f"{energy:.4e}    {T:.4f}   {terr:.4f}")





dataset_label = 'Musgrove (1977) (80m)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

E_max = max(original_x)
E_upper = E_max
#print(E_upper)
E_upper = E_upper -1.1e5
#print(E_upper)
E_upper = E_upper -0.8e5
#print(E_upper)
E_upper = E_upper -0.55e5
#print(E_upper)

subtractions_ev = [1.1e5, 0.8e5, 0.55e5]
bin_edges = [E_max]
for sub in subtractions_ev:
    E_max -= sub
    bin_edges.append(E_max)

bin_edges.append(0)
bin_edges = list(reversed(bin_edges))
process_data_into_bins(bin_edges, original_x, original_y, original_yerr)
xvals, Tvals, terr = process_data_into_bins(bin_edges, original_x, trans, trans_err)
datasets[dataset_label]['Processed']['x-vals'] = xvals
datasets[dataset_label]['Processed']['T-vals'] = Tvals
datasets[dataset_label]['Processed']['T-err'] = terr

print("\n# Musgrove 80m (1977)")
print("UNCERTAINTY    ABSOLUTE")
print("THICKNESS      0.0827")
print("TEMPERATURE    293.0")
print("ABUNDANCES     0.9765 0.0093 0.0070 0.0056 0.0016\n")
for energy, T, terr in zip(xvals, Tvals, terr):
    print(f"{energy:.4e}    {T:.4f}   {terr:.4e}")



dataset_label = 'Green (1973)'
bin_edges = [0.55e6, 0.60e6, 0.65e6, 0.70e6, 0.75e6, 0.80e6, 0.85e6, 0.90e6, 0.95e6, 1.00e6,
             1.05e6, 1.10e6, 1.15e6, 1.20e6, 1.25e6, 1.30e6, 1.35e6, 1.40e6, 1.45e6, 1.50e6,
             1.55e6, 1.60e6, 1.65e6, 1.70e6, 1.75e6, 1.80e6, 1.85e6, 1.90e6, 1.95e6, 2.0e6]
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

process_data_into_bins(bin_edges, original_x, original_y, original_yerr)
xvals, Tvals, terr = process_data_into_bins(bin_edges, original_x, trans, trans_err)
datasets[dataset_label]['Processed']['x-vals'] = xvals
datasets[dataset_label]['Processed']['T-vals'] = Tvals
datasets[dataset_label]['Processed']['T-err'] = terr


print("\n# Green (1973)")
print("UNCERTAINTY    ABSOLUTE")
print("THICKNESS      0.0799")
print("TEMPERATURE    294.0")
print("ABUNDANCES     0.9772  0.0107  0.0051  0.0056  0.0015\n")
for energy, T, terr in zip(xvals, Tvals, terr):
    print(f"{energy:.4e}    {T:.4f}   {terr:.4e}")

# print("MUSGROVE 80 M")
# print(datasets['Musgrove (1977) (80m)']['Processed']['x-vals'])
# print(datasets['Musgrove (1977) (80m)']['Processed']['T-vals'])
# print(datasets['Musgrove (1977) (80m)']['Processed']['T-err'])

# print("MUSGROVE 200 M")
# print(datasets['Musgrove (1977) (200m)']['Processed']['x-vals'])
# print(datasets['Musgrove (1977) (200m)']['Processed']['T-vals'])
# print(datasets['Musgrove (1977) (200m)']['Processed']['T-err'])

datasets['Guenther (1974)']['EXFOR']['x-vals']

dataset_label = 'Guenther (1974)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

# Choose an upper energy boundary and parameters for analysis
E_upper = 2.0e6 # max(datasets[dataset_label]['EXFOR']['x-vals']) # eV (0.8 MeV)

print(E_upper)
E_upper = E_upper -1.6e5
print(E_upper)
E_upper = E_upper -4.0e5
print(E_upper)
E_upper = E_upper -2.4e5
print(E_upper)
# E_upper = E_upper -2.4e5
# print(E_upper)
# E_upper = E_upper -1.4e5
# print(E_upper)
# E_upper = E_upper -1.2e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.0e5
# E_upper = E_upper -1.1e5
# E_upper = E_upper -0.9e5
# E_upper = E_upper -0.9e5
#E_upper = E_upper -1.9e5
#E_upper = E_upper -1.7e5
max_bin_width = 5e5 # eV (Test bins up to 800 keV width ending at E_upper)
step_increment = 5000 # eV (5 keV steps in width)

# Analyze standard deviation vs. bin width from the top down
std_dev_data_from_top_0p8MeV = analyze_std_dev_vs_bin_width_from_top(
    original_x, original_y, original_yerr, E_upper, max_bin_width, step_increment)

# Separate the results into width and std dev arrays
bin_widths_from_top_0p8MeV_kev = std_dev_data_from_top_0p8MeV[:, 0] / 1e3 # Convert width to keV
std_devs_from_top_0p8MeV = std_dev_data_from_top_0p8MeV[:, 1]

# Filter out NaNs for plotting (optional, but often makes line plots cleaner)
valid_indices = ~np.isnan(std_devs_from_top_0p8MeV)
plot_widths_from_top_0p8MeV_kev = bin_widths_from_top_0p8MeV_kev[valid_indices]
plot_std_devs_from_top_0p8MeV = std_devs_from_top_0p8MeV[valid_indices]


# Plotting the results for this specific upper energy boundary
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(plot_widths_from_top_0p8MeV_kev, plot_std_devs_from_top_0p8MeV, 'o-', markersize=4)

ax.set_xlabel('Full Bin Width (keV)')
ax.set_ylabel('Standard Deviation of Original Data (barns)')
ax.set_title(f'Std Dev vs. Bin Width Ending at {E_upper/1e6:.2f} MeV')

ax.grid(True, which='both', linestyle='--', color='#cccccc', alpha=0.7)
plt.tight_layout()
plt.show()

dataset_label = 'Guenther (1974)'
original_x = datasets[dataset_label]['EXFOR']['x-vals']
original_y = datasets[dataset_label]['EXFOR']['y-vals']
original_yerr = datasets[dataset_label]['EXFOR']['yerr-vals']

E_max = 2.0e6


subtractions_ev = [1.6e5, 1.8e5, 2.4e5, 2.4e5, 1.4e5]
bin_edges = [E_max]
for sub in subtractions_ev:
    E_max -= sub
    bin_edges.append(E_max)

bin_edges.append(0)
bin_edges = list(reversed(bin_edges))
process_data_into_bins(bin_edges, original_x, original_y, original_yerr)
xvals, Tvals, terr = process_data_into_bins(bin_edges, original_x, trans, trans_err)
datasets[dataset_label]['Processed']['x-vals'] = xvals
datasets[dataset_label]['Processed']['T-vals'] = Tvals
datasets[dataset_label]['Processed']['T-err'] = terr


print("Guenther (1974)")
print(datasets["Guenther (1974)"]['Processed']['x-vals'])
print(datasets["Guenther (1974)"]['Processed']['T-vals'])
print(datasets["Guenther (1974)"]['Processed']['T-err'])

datasets["Guenther (1974)"].keys()

datasets["Guenther (1974)"]['Processed']["thickness"]

