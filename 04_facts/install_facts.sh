#!/usr/bin/env bash

# Install FACTS v1.1.5 in $HOME/Github/facts_v1.1.5, build Docker image, run dummy experiment, and download module data.
#
# Usage:
#   mamba activate d25a-rsl-fusion
#   bash 04_facts/install_facts.sh
#
# Requirements:
#   git, wget (included in d25a-rsl-fusion environment), and Docker Desktop.
#
# Relevant FACTS documentation:
#   https://fact-sealevel.readthedocs.io/en/latest/quickstart.html#installing-and-using-facts-on-a-gnu-linux-container

set -euo pipefail

# FACTS version
facts_tag="v1.1.5"

# Local path in which to install FACTS
facts_dir="${HOME}/Github/facts_${facts_tag}"

# Clone repository
echo "Cloning FACTS ${facts_tag} to ${facts_dir}"
git clone --depth 1 --branch "${facts_tag}" https://github.com/radical-collaboration/facts "${facts_dir}"

# Build Docker image
facts_image="facts:${facts_tag}"
docker build --no-cache --target facts-core -t "${facts_image}" -f "${facts_dir}/docker/Dockerfile" "${facts_dir}"

# Run dummy experiment
docker run --rm --init \
    -e PATH="/factsVe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    -v "${facts_dir}:/opt/facts" \
    -w /opt/facts \
    "${facts_image}" \
    python3 runFACTS.py experiments/dummy

# Download module data
echo "Downloading module data"
wget --continue -P "${facts_dir}/modules-data" -i "${facts_dir}/modules-data/modules-data.urls.txt"
