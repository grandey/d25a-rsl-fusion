#!/usr/bin/env bash

# Download external input datasets used by d25a-rsl-fusion.
#
# Usage:
#   bash 02_input_data/download_input_data.sh

set -euo pipefail

# Directory containing this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Directory to which data are downloaded
download_dir="${script_dir}/data_downloaded"
mkdir -p "${download_dir}"

# URLs to download
urls=(
    "https://population.un.org/wup/assets/Download/Cities/WUP2025-F21-DEGURBA-Cities_Pop.xlsx"
    "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dist2coast_4deg.nc"
    "https://raw.githubusercontent.com/radical-collaboration/facts/refs/tags/v1.1.4/input_files/location.lst"
)

# Download data from URLs
for url in "${urls[@]}"; do
    echo "Downloading ${url} to ${download_dir}"
    curl --fail --location --output-dir "${download_dir}" --remote-name "${url}"
done
