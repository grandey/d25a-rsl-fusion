# Analysis Code for _High-end and low-end projections of relative sea-level rise_ (d25a-rsl-fusion)

## Usage guidelines
This repository accompanies the following manuscript:

B. S. Grandey et al.,  **High-end and low-end projections of relative sea-level rise**, in preparation.

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
Input data from the [IPCC AR6 Sea Level Projections](https://doi.org/10.5281/zenodo.6382554) and the [IPCC AR6 Relative Sea Level Projection Distributions](https://doi.org/10.5281/zenodo.5914932) repositories can be downloaded as follows:

```
mkdir data_ar6
cd data_ar6
curl "https://zenodo.org/records/6382554/files/ar6.zip?download=1" -O
unzip ar6.zip
curl "https://zenodo.org/records/5914932/files/ar6-regional-distributions.zip?download=1" -O
unzip ar6-regional-distributions.zip
curl "https://zenodo.org/records/6382554/files/location_list.lst?download=1" -O
cd ..
```

Users of these data IPCC AR6 projections should note the [required acknowledgments and citations](https://doi.org/10.5281/zenodo.6382554).

The [PSMSL catalogue file](https://psmsl.org/data/obtaining/nucat.dat) identifies which country each tide gauge corresponds to:

```
mkdir data_psmsl
cd data_psmsl
curl "https://psmsl.org/data/obtaining/nucat.dat" -O
cd ..
```

### 3. Produce data for fusion, high-end, and low-end projections
[**`d25a_data.ipynb`**](d25a_data.ipynb) uses the input data to produce the fusion, high-end, and low-end projections, which are saved to [**`data_fusion/`**](data_fusion/).

### 4. Analyse data and produce figures - NOT YET IMPLEMENTED
[**`d25a_figs.ipynb`**](d25a_figs.ipynb) analyses the projections and produces the figures.

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with colleagues.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).
We thank the projection authors for developing and making the sea level rise projections available, multiple funding agencies for supporting the development of the projections, and the NASA Sea Level Change Team for developing and hosting the IPCC AR6 Sea Level Projection Tool.
