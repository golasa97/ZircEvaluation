set -e


avgParameterDir="UncertaintyQuantification/AvgParameters"
sampleDir="UncertaintyQuantification/SampledParameters"
xs0Kdir="UncertaintyQuantification/0KCrossSections"

zr90_mass=89.9047
zr90_radius=6.31
zr90_spin=0.0

zr91_mass=90.9056
zr91_radius=6.33
zr91_spin=2.5

zr92_mass=91.9050
zr92_radius=7.20
zr92_spin=0.0

zr94_mass=93.9063
zr94_radius=7.20
zr94_spin=0.0

echo "Generating Resonance Samples in URR"


numLadders=1

zr90_avgParameter="${avgParameterDir}/zr90_urr_parameters.csv"
zr91_avgParameter="${avgParameterDir}/zr91_urr_parameters.csv"
zr92_avgParameter="${avgParameterDir}/zr92_urr_parameters.csv"
zr94_avgParameter="${avgParameterDir}/zr94_urr_parameters.csv"


python3 scripts/call_rurr.py --param_file "UncertaintyQuantification/AvgParameters/zr90_urr_parameters.csv" --output_file "UncertaintyQuantification/0KCrossSections/zr90/zr90_xs" --mass ${zr90_mass} --radius ${zr90_radius} --nLadders ${numLadders}


#zr90_sampledir="${sampleDir}/zr90/"
#zr91_sampledir="${sampleDir}/zr91/"
#zr92_sampledir="${sampleDir}/zr92/"
#zr94_sampledir="${sampleDir}/zr94/"
#python3 scripts/sample_resonance_ladder.py ${zr90_avgParameter} --mass ${zr90_mass} --radius ${zr90_radius} --output_dir ${zr90_sampledir} --num_ladders ${numLadders}
#python3 scripts/sample_resonance_ladder.py ${zr91_avgParameter} --mass ${zr91_mass} --radius ${zr91_radius} --output_dir ${zr91_sampledir} --num_ladders ${numLadders}
#python3 scripts/sample_resonance_ladder.py ${zr92_avgParameter} --mass ${zr92_mass} --radius ${zr92_radius} --output_dir ${zr92_sampledir} --num_ladders ${numLadders}
#python3 scripts/sample_resonance_ladder.py ${zr94_avgParameter} --mass ${zr94_mass} --radius ${zr94_radius} --output_dir ${zr94_sampledir} --num_ladders ${numLadders}
#
#echo "Starting all cross-section calculations at $(date)"
#
#zr90_xs0Kdir="${xs0Kdir}/zr90"
#zr91_xs0Kdir="${xs0Kdir}/zr91"
#zr92_xs0Kdir="${xs0Kdir}/zr92"
#zr94_xs0Kdir="${xs0Kdir}/zr94"
##echo "--- Processing Zr90 ---"
#python3 scripts/calculate_xs_from_ladder.py ${zr90_sampledir} ${zr90_xs0Kdir} --mass ${zr90_mass} --radius ${zr90_radius} --spin ${zr90_spin} --emin 1e5 --emax 5e5 --global_points 5000 --local_points 10 --width_multiplier 2.0
#python3 scripts/calculate_xs_from_ladder.py ${zr91_sampledir} ${zr91_xs0Kdir} --mass ${zr91_mass} --radius ${zr91_radius} --spin ${zr91_spin} --emin 1e5 --emax 5e5 --global_points 5000 --local_points 10 --width_multiplier 2.0
#python3 scripts/calculate_xs_from_ladder.py ${zr92_sampledir} ${zr92_xs0Kdir} --mass ${zr92_mass} --radius ${zr92_radius} --spin ${zr92_spin} --emin 1e5 --emax 5e5 --global_points 5000 --local_points 10 --width_multiplier 2.0
#python3 scripts/calculate_xs_from_ladder.py ${zr94_sampledir} ${zr94_xs0Kdir} --mass ${zr94_mass} --radius ${zr94_radius} --spin ${zr94_spin} --emin 1e5 --emax 5e5 --global_points 5000 --local_points 10 --width_multiplier 2.0
##
##echo "--- Processing Zr91 ---"
##python3 scripts/calculate_xs_from_ladder.py UncertaintyQuantification/SampledParameters/zr91/ UncertaintyQuantification/0KCrossSections/zr91/ --mass 90.9056 --radius 6.33 --spin 2.5 --emin 1e5 --emax 2e6 --global_points 5000 --local_points 100 --width_multiplier 10.0
#
##echo "--- Processing Zr92 ---"
##python3 scripts/calculate_xs_from_ladder.py UncertaintyQuantification/SampledParameters/zr92/ UncertaintyQuantification/0KCrossSections/zr92/ --mass 91.9050 --radius 7.20 --spin 0.0 --emin 1e5 --emax 2e6 --global_points 5000 --local_points 20 --width_multiplier 5.0
##
##echo "--- Processing Zr94 ---"
##python3 scripts/calculate_xs_from_ladder.py UncertaintyQuantification/SampledParameters/zr92/ UncertaintyQuantification/0KCrossSections/zr94/ --mass 93.9063 --radius 7.20 --spin 0.0 --emin 1e5 --emax  2e6 --global_points 5000 --local_points 20 --width_multiplier 5.0
