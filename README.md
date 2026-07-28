# Analysis code and data for _Worst- and best-case sea-level projections for coastal cities_ (d25a-rsl-fusion)

[![DOI](https://zenodo.org/badge/907213343.svg)](https://zenodo.org/badge/latestdoi/907213343)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey et al.,  **Worst- and best-case sea-level projections for coastal cities**, in preparation.

The manuscript serves as the primary reference.
The Zenodo archive of this repository serves as a secondary reference.

## Data files containing sea-level projections [TO REVISE]
The sea-level projections are contained in [**`data_d25a/`**](data_d25a/).  The values are in metres, relative to the IPCC AR6 baseline (1995–2014).

### Contents
1. [**`time_series/`**](data_d25a/time_series/) – NetCDF time series for 2020, 2030, ..., 2100, for the locations in [**`locations_info_d25a.csv`**](data_d25a/time_series/locations_info_d25a.csv) (grid boxes with coastal cities, alongside tide gauge locations).
2. [**`year_2100/`**](data_d25a/year_2100/) – CSV summaries for 2100.

### File-name conventions
- The prefix `rsl` refers to relative sea-level rise, `novlm` refers to geocentric sea-level rise (i.e. no VLM component), and `gmsl` refers to global mean sea-level rise. 
- In [**`year_2100/`**](data_d25a/year_2100/), `grid` refers to 1°×1° grid locations (preferred), while `gauges` refers to tide gauge locations. 
- In the [**`time_series/`**](data_d25a/time_series/), `fusion-ssp585` and `fusion-ssp126` refer to full probabilistic fusion projections under SSP5-8.5 and SSP1-2.6, while `high-end`, `high`, `central`, `low`, and `low-end` refer to the projections described in the manuscript.

#### Primary projections of relative sea-level rise by 2100
- [**`rsl_grid_megacities_2100_d25a.csv`**](data_d25a/year_2100/rsl_grid_megacities_2100_d25a.csv) – projections for coastal megacities, with projected 2050 population of at least 10 million.
- [**`rsl_grid_cities_2100_d25a.csv`**](data_d25a/year_2100/rsl_grid_cities_2100_d25a.csv) – projections for all coastal cities.

## Workflow
The following workflow can be used to reproduce and analyse the projections.

### 1. Create environment
To create a _conda_ environment with the required software dependencies, use [`01_environment/environment.yml`](01_environment/environment.yml):

```
mamba env create -f 01_environment/environment.yml
```

The analysis workflow has been developed within this `d25a-rsl-fusion` environment on _macOS 26_ (arm64). The exact package versions used for development are recorded in [`01_environment/environment-resolved.yml`](01_environment/environment-resolved.yml).

If you do not already have access to a _Jupyter_ server, you can install _JupyterLab_ in the same environment:

```
mamba activate d25a-rsl-fusion
mamba install jupyterlab
```

### 2. Download input data
To download the input data, run [`02_input_data/download_input_data.sh`](02_input_data/download_input_data.sh):

```
bash 02_input_data/download_input_data.sh
```

The script downloads the following input data to the directory `02_input_data/data_downloaded/`:
1. City populations and locations from the United Nations Department of Economic and Social Affairs Population Division's [World Urbanization Prospects 2025](https://population.un.org/wup/): [`WUP2025-F21-DEGURBA-Cities_Pop.xlsx`](https://population.un.org/wup/assets/Download/Cities/WUP2025-F21-DEGURBA-Cities_Pop.xlsx).
2. The [Distance to Nearest Coastline 0.04-Degree Grid dataset](https://www.pacioos.hawaii.edu/metadata/dist2coast_4deg.html), created by the NASA Ocean Biology Processing Group and distributed by the Pacific Islands Ocean Observing System: [`dist2coast_4deg.nc`](https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dist2coast_4deg.nc).
3. The complete list of tide-gauge and grid locations used by [FACTS v1.1.4](https://github.com/radical-collaboration/facts/tree/v1.1.4): [`location.lst`](https://raw.githubusercontent.com/radical-collaboration/facts/refs/tags/v1.1.4/input_files/location.lst).

### 3. Identify locations of interest [TODO]

### 4. Run FACTS [TODO]

### 5. Produce fusion, high-end, high, central, low, and low-end projections [TO REVISE]
[**`data_d25a.ipynb`**](data_d25a.ipynb) uses the input data to produce the fusion, high-end, high, central, low, and low-end projections, which are saved to [**`data_d25a/`**](data_d25a/).

### 6. Analyse data and produce figures [TO REVISE]
[**`figs_d25a.ipynb`**](figs_d25a.ipynb) analyses the projections and produces the figures.

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with colleagues.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
