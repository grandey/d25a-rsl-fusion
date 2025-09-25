# Analysis code and data for _Worst and best-case sea-level projections for coastal cities_ (d25a-rsl-fusion)

[![DOI](https://zenodo.org/badge/907213343.svg)](https://zenodo.org/badge/latestdoi/907213343)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey et al.,  **Worst and best-case sea-level projections for coastal cities**, in preparation.

The manuscript serves as the primary reference.
The Zenodo archive of this repository serves as a secondary reference.

## Data files containing sea-level projections
The sea-level projections are contained in [**`data_d25a/`**](data_d25a/).  The values are in metres, relative to the IPCC AR6 baseline (1995–2014).

### Contents
1. [**`time_series/`**](data_d25a/time_series/) – NetCDF time series for 2020, 2030, ..., 2100, for the locations in [**`locations_info_d25a.csv`**](data_d25a/time_series/locations_info_d25a.csv) (grid boxes with coastal cities, alongside tide gauge locations).
2. [**`year_2100/`**](data_d25a/year_2100/) – CSV summaries for 2100.

### File-name conventions
- The prefix `rsl` refers to relative sea-level rise, `novlm` refers to geocentric sea-level rise (i.e. no VLM component), and `gmsl` refers to global mean sea-level rise. 
- In [**`year_2100/`**](data_d25a/year_2100/), `grid` refers to 1°×1° grid locations (preferred), while `gauges` refers to tide gauge locations. 
- In the [**`time_series/`**](data_d25a/time_series/), `fusion-ssp585` and `fusion-ssp126` refer to full probabilistic fusion projections under SSP5-8.5 and SSP1-2.6, while `high-end`, `high`, `central`, `low`, and `low-end` refer to the projections described in the manuscript.

#### Primary projections of relative sea-level rise by 2100
- [**`rsl_grid_megacities_2100_d25a.csv`**](data_d25a/year_2100/rsl_grid_megacities_2100_d25a.csv) – projections for 48 large coastal cities.
- [**`rsl_grid_cities_2100_d25a.csv`**](data_d25a/year_2100/rsl_grid_cities_2100_d25a.csv) – projections for all coastal cities.

## Workflow
The following workflow can be used to reproduce and analyse the projections.

### 1. Create environment
To create a _conda_ environment with the necessary software dependencies, use the [**`environment.yml`**](environment.yml) file:

```
conda env create --file environment.yml
```

The analysis has been performed within this environment on _macOS 13_ (arm64).

### 2. Download input data
Input data from the [IPCC AR6 Sea Level Projections](https://doi.org/10.5281/zenodo.6382554), the [IPCC AR6 Relative Sea Level Projection Distributions](https://doi.org/10.5281/zenodo.5914932), and the [IPCC AR6 Relative Sea Level Projections without Background Component](https://doi.org/10.5281/zenodo.5967269) repositories can be downloaded as follows:

```
mkdir -p data_in/ar6
cd data_in/ar6
curl "https://zenodo.org/records/6382554/files/ar6.zip?download=1" -O
unzip ar6.zip
curl "https://zenodo.org/records/5914932/files/ar6-regional-distributions.zip?download=1" -O
unzip ar6-regional-distributions.zip
curl "https://zenodo.org/records/5967269/files/ar6-regional_novlm-distributions.zip?download=1" -O
unzip ar6-regional_novlm-distributions.zip
curl "https://zenodo.org/records/6382554/files/location_list.lst?download=1" -O
cd ../..
```

Users of these IPCC AR6 projections should note the [required acknowledgments and citations](https://doi.org/10.5281/zenodo.6382554).

The [PSMSL catalogue file](https://psmsl.org/data/obtaining/nucat.dat) identifies which country each tide gauge corresponds to:

```
mkdir -p data_in/psmsl
cd data_in/psmsl
curl "https://psmsl.org/data/obtaining/nucat.dat" -O
cd ../..
```

The United Nations Department of Economic and Social Affairs [World Urbanisation Prospects 2018](https://population.un.org/wup/downloads?tab=Urban%20Agglomerations) File 12 contains the population and locations of cities (urban agglomerations):

```
mkdir -p data_in/wup18
cd data_in/wup18
curl "https://population.un.org/wup/assets/Download/WUP2018-F12-Cities_Over_300K.xls" -O
cd ../..
```

The NASA [Distance to the Nearest Coast](https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/) dataset contains the distance to the nearest coast at 0.04-degree resolution:

```
mkdir -p data_in/nasa
cd data_in/nasa
curl "https://oceancolor.gsfc.nasa.gov/images/resources/distfromcoast/dist2coast.txt.bz2" -O
bzip2 -d dist2coast.txt.bz2
cd ../..
```

### 3. Produce data for fusion, high-end, high, central, low, and low-end projections
[**`data_d25a.ipynb`**](data_d25a.ipynb) uses the input data to produce the fusion, high-end, high, central, low, and low-end projections, which are saved to [**`data_d25a/`**](data_d25a/).

### 4. Analyse data and produce figures
[**`figs_d25a.ipynb`**](figs_d25a.ipynb) analyses the projections and produces the figures.

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with colleagues.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
