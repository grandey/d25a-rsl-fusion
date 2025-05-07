# Analysis Code for _Worst-case projections of relative sea-level rise_ (d25a-rsl-fusion)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey et al.,  **Worst-case projections of sea-level rise for cities**, in preparation.

The manuscript serves as the primary reference.
The Zenodo archive of this repository serves as a secondary reference.

## Workflow

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

### 3. Produce data for fusion, high-end, high, central, and low projections
[**`data_d25a.ipynb`**](data_d25a.ipynb) uses the input data to produce the fusion, high-end, high, central, and low projections, which are saved to [**`data_d25a/`**](data_d25a/).

### 4. Analyse data and produce figures
[**`figs_d25a.ipynb`**](figs_d25a.ipynb) analyses the projections and produces the figures.

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with colleagues.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
