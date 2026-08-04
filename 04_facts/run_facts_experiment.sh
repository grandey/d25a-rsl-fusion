#!/usr/bin/env bash

# Run an experiment in 04_facts/experiments/ using FACTS.
#
# Requirements:
#   FACTS v1.1.5 (installed using install_facts.sh) and Docker Desktop.
#
# Usage:
#   bash 04_facts/run_facts_experiment.sh ssp126
#   bash 04_facts/run_facts_experiment.sh ssp245
#   bash 04_facts/run_facts_experiment.sh ssp370
#   bash 04_facts/run_facts_experiment.sh ssp585

set -euo pipefail

# Check that one command-line argument is supplied
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 {ssp126|ssp245|ssp370|ssp585}" >&2
    exit 1
fi

# Directory containing this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FACTS version, directory, and Docker image (see install_facts.sh)
facts_tag="v1.1.5"
facts_dir="${HOME}/Github/facts_${facts_tag}"
facts_image="facts:${facts_tag}"

# Experiment paths
experiment_name="d25a.${1}"
experiment_dir="${script_dir}/experiments/${experiment_name}"
location_file="${script_dir}/../03_locations/data_locations/location.lst"
log_file="${experiment_dir}/run.log"

# Copy location.lst
echo "Copying ${location_file} to ${experiment_dir}/location.lst"
cp "${location_file}" "${experiment_dir}/location.lst"

# Run FACTS experiment
echo "Running FACTS experiment ${experiment_name}"
docker run --rm --init \
    -e PATH="/factsVe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    -v "${facts_dir}:/opt/facts" \
    -v "${script_dir}/experiments:/opt/experiments" \
    -w /opt/facts \
    "${facts_image}" \
    python3 runFACTS.py "/opt/experiments/${experiment_name}" \
    2>&1 | tee "${log_file}"
