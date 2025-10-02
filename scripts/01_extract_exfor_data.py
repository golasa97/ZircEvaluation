import numpy as np
import requests
import json
import os

def extract_exfor_data(url: str):
    """
    Fetches data from a plain text EXFOR URL, extracts specified metadata,
    X, Y, YERR columns, and their units, converting to a standard output format.
    Handles both 3-column and 4-column data tables.
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
    data_column_indices = None # To store indices for (X, Y, YERR)

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
        metadata_keywords_to_check = ["REFERENCE", "AUTHOR", "TITLE", "SAMPLE", "METHOD", "DETECTOR"]
        is_new_metadata_keyword_line = False

        if data_parse_state == 0:
            for kw in metadata_keywords_to_check:
                if stripped_line.startswith(kw):
                    if current_metadata_keyword_internal and out[current_metadata_keyword_internal]:
                        out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()

                    current_metadata_keyword_internal = kw
                    value_part = stripped_line[len(kw):].strip()
                    if out[kw]:
                        out[kw] += "; " + value_part
                    else:
                        out[kw] = value_part
                    is_new_metadata_keyword_line = True
                    break

            if not is_new_metadata_keyword_line and current_metadata_keyword_internal and \
               (original_line.startswith("          ") or original_line.startswith("            ") or \
                (stripped_line and original_line.startswith(" ") and not columns[0].isupper())):
                if out[current_metadata_keyword_internal]:
                     out[current_metadata_keyword_internal] += " " + stripped_line
                else:
                     out[current_metadata_keyword_internal] = stripped_line
                is_new_metadata_keyword_line = True

        # --- Data Block Parsing State Machine ---
        if first_word == "DATA" and data_parse_state == 0:
            if current_metadata_keyword_internal:
                 out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()
                 current_metadata_keyword_internal = None
            data_parse_state = 1
            continue

        if data_parse_state == 1: # Seeking column NAME header
            # Check for 4-column format first
            if len(columns) >= 4 and "EN" in columns[0] and "EN-RSL" in columns[1] and "DATA" in columns[2] and ("ERR" in columns[3] or "DATA-ERR" in columns[3]):
                data_column_indices = (0, 2, 3) # EN, DATA, DATA-ERR
                data_parse_state = 2
            # Check for 3-column format
            elif len(columns) >= 3 and "EN" in columns[0] and "DATA" in columns[1] and ("ERR" in columns[2] or "DATA-ERR" in columns[2]):
                data_column_indices = (0, 1, 2) # EN, DATA, DATA-ERR
                data_parse_state = 2
            continue

        elif data_parse_state == 2: # Seeking column UNIT header
            try:
                float(columns[0].upper().replace('D','E'))
                data_parse_state = 3
            except (ValueError, IndexError):
                if data_column_indices and len(columns) >= max(data_column_indices) + 1:
                    x_col, y_col, yerr_col = data_column_indices
                    parsed_x_unit_original = columns[x_col].upper()
                    parsed_y_unit_original = columns[y_col].upper()
                    parsed_yerr_unit_original = columns[yerr_col].upper()
                data_parse_state = 3
                continue

        if data_parse_state == 3: # Parsing numerical data
            if first_word == "ENDDATA":
                data_parse_state = 0
                data_column_indices = None # Reset for the next data block
                if current_metadata_keyword_internal:
                     out[current_metadata_keyword_internal] = out[current_metadata_keyword_internal].strip()
                     current_metadata_keyword_internal = None
                continue

            if is_new_metadata_keyword_line:
                 continue

            if not data_column_indices: # If headers were missing, assume 3-column
                data_column_indices = (0, 1, 2)

            if len(columns) >= max(data_column_indices) + 1:
                try:
                    x_col, y_col, yerr_col = data_column_indices
                    x_val = float(columns[x_col].upper().replace('D','E'))
                    y_val = float(columns[y_col].upper().replace('D','E'))
                    yerr_val = float(columns[yerr_col].upper().replace('D','E'))

                    raw_x_values.append(x_val)
                    raw_y_values.append(y_val)
                    raw_yerr_values.append(yerr_val)
                except (ValueError, IndexError):
                    pass # Not a numerical data line, or malformed, skip

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

def convert_numpy_to_list(data):
    """
    Recursively converts numpy arrays in a dictionary or list to lists.
    """
    if isinstance(data, dict):
        return {k: convert_numpy_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_list(i) for i in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

# --- Zr-90 Data ---
datasets_zr90 = {}

datasets_zr90['Musgrove (1977) (80m)'] = {}
datasets_zr90['Musgrove (1977) (80m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=13736002')

datasets_zr90['Musgrove (1977) (200m)'] = {}
datasets_zr90['Musgrove (1977) (200m)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=13736003')

datasets_zr90['Guenther (1974)'] = {}
datasets_zr90['Guenther (1974)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=10468002')

datasets_zr90['Green (1973)'] = {}
datasets_zr90['Green (1973)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=5896&subID=10225018')

# Define processing parameters for each dataset
zr90_parameters = {
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
        'thickness': 0.0799,
        'temperature': 294,
        'abundances': [0.9772, 0.0107, 0.0051, 0.0056, 0.0015]
    },
    'Guenther (1974)': {
        'thickness': 0.0799,
        'temperature': 294,
        'abundances': [0.9772, 0.0107, 0.0051, 0.0056, 0.0015]
    },
}

# Process each dataset using the new function
for label, params in zr90_parameters.items():
    process_dataset(datasets_zr90, label, params['thickness'], params['temperature'], params['abundances'])

# --- Zr-91 Data ---
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

for label, params in zr91_parameters.items():
    process_dataset(datasets_zr91, label, params['thickness'], params['temperature'], params['abundances'])


datasets_natZr = {}

datasets_natZr['Rapp (2019) (6cm)'] = {}
datasets_natZr['Rapp (2019) (6cm)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=7762&subID=14576009')


datasets_natZr['Rapp (2019) (10cm)'] = {}
datasets_natZr['Rapp (2019) (10cm)']['EXFOR'] = extract_exfor_data('https://www-nds.iaea.org/exfor/servlet/X4sGetSubent?reqx=7762&subID=14576011')


natZr_parameters = {
    'Rapp (2019) (6cm)': {
        'thickness' : 0.25812,
        'temperature' : 294.0,
        'abundances' : [0.515, 0.112, 0.171, 0.174, 0.028]
        },
    'Rapp (2019) (10cm)': {
        'thickness' : 0.4296,
        'temperature' : 294.0,
        'abundances' : [0.515, 0.112, 0.171, 0.174, 0.028]
        }
    }

for label, params in natZr_parameters.items():
    process_dataset(datasets_natZr, label, params['thickness'], params['temperature'], params['abundances'])

# --- Save Data ---
all_datasets = {
    'zr90': datasets_zr90,
    'zr91': datasets_zr91,
    'nat-zr' : datasets_natZr
}

all_datasets_serializable = convert_numpy_to_list(all_datasets)

# Note: The script is in 'scripts/', so we go up one level to the project root.
output_dir = os.path.join(os.path.dirname(__file__), '..', 'experimental_data', 'pointwise')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for isotope, datasets in all_datasets_serializable.items():
    for label, data in datasets.items():
        # Sanitize the label to create a valid filename
        sanitized_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"{isotope}_{sanitized_label}.json"
        output_file = os.path.join(output_dir, filename)
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Data for '{label}' ({isotope}) saved to {output_file}")
