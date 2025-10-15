#!/bin/bash
set -e

# This script runs the Doppler broadening Python script on 0K cross-section files.
# It's structured to handle multiple isotopes and multiple cross-section files per isotope.

echo "--- Starting Doppler Broadening Process ---"

# Define base directories
# You might need to adjust these paths depending on your project structure.
XS_0K_DIR="UncertaintyQuantification/0KCrossSections"
XS_294K_DIR="UncertaintyQuantification/294KCrossSections"
SCRIPT_PATH="scripts/doppler_broaden_xs.py"

# --- Isotope Parameters ---
# Add other isotopes here as needed.
ZR90_MASS=89.9047
ZR91_MASS=90.9056
ZR92_MASS=91.9050
ZR94_MASS=93.9063

# --- Processing for Zr90 ---
echo "--- Processing Zr90 ---"
ISOTOPE="zr90"
MASS=${ZR90_MASS}
TEMPERATURE=294.0

INPUT_DIR="${XS_0K_DIR}/${ISOTOPE}"
OUTPUT_DIR="${XS_294K_DIR}/${ISOTOPE}"

# Create the output directory if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Check if there are files to process
if [ -z "$(find "${INPUT_DIR}" -maxdepth 1 -name '*.csv' -print -quit)" ]; then
    echo "No .csv files found in ${INPUT_DIR}. Skipping Zr90."
else
    # Loop over all 0K cross-section files for this isotope
    for infile in "${INPUT_DIR}"/*.csv; do
        # Ensure it's a file before processing
        if [ -f "$infile" ]; then
            filename=$(basename -- "$infile")
            outfile="${OUTPUT_DIR}/${filename%.csv}_${TEMPERATURE}K.csv"
            
            echo "Broadening ${infile} -> ${outfile}"
            
            python3 "${SCRIPT_PATH}" \
                "${infile}" \
                "${outfile}" \
                --mass "${MASS}" \
                --temp "${TEMPERATURE}"
        fi
    done
fi

# --- You can add processing for other isotopes below ---
# echo ""
# echo "--- Processing Zr91 ---"
# ... copy the block above and change the ISOTOPE and MASS variables ...


echo ""
echo "--- Doppler Broadening Process Finished ---"
