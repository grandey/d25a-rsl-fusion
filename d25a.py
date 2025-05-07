"""
d25a:
    Functions that support the analysis contained in the d25a-rsl-fusion repository.

Author:
    Benjamin S. Grandey, 2024–2025.

Notes:
    Much of this code is based on the d23a-fusion repository, https://doi.org/10.5281/zenodo.13627262.
"""


import cartopy.crs as ccrs
from functools import cache
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy import stats
import seaborn as sns
from watermark import watermark
import xarray as xr


# Matplotlib settings
sns.set_style('whitegrid')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['axes.titleweight'] = 'bold'  # titles for subplots
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.axisbelow'] = True  # grid should be behind other elements
plt.rcParams['grid.color'] = '0.95'

# Constants
SCENARIO_LABEL_DICT = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5', 'ssp245': 'SSP2-4.5'}  # names of scenarios
SLR_LABEL_DICT = {'gmsl': 'Global mean SLR', 'rsl': 'Relative SLR', 'novlm': 'Geocentric SLR'}
AR6_DIR = Path.cwd() / 'data_in' / 'ar6'  # directory containing AR6 input data
PSMSL_DIR = Path.cwd() / 'data_in' / 'psmsl'  # directory containing PSMSL catalogue file
WUP18_DIR = Path.cwd() / 'data_in' / 'wup18'  # directory containing World Urbanisation Prospects 2018 data
DATA_DIR = Path.cwd() / 'data_d25a'  # directory containing projections produced by data_d25a.ipynb
FIG_DIR = Path.cwd() / 'figs_d25a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter
MEGACITIES_LIST = [  # 48 largest coastal cities analysed by Tay et al. (2022)
        "Abidjan", "Ahmadabad", "Alexandria", "Bangkok", "Barcelona",
        "Buenos Aires", "Chennai", "Chittagong", "Dalian", "Dar es Salaam",
        "Dhaka", "Dongguan", "Foshan", "Fukuoka", "Guangzhou",
        "Hangzhou", "Ho Chi Minh City", "Hong Kong", "Houston", "Istanbul",
        "Jakarta", "Karachi", "Kolkata", "Lagos", "Lima",
        "London", "Los Angeles", "Luanda", "Manila", "Miami",
        "Mumbai", "Nagoya", "Nanjing", "New York", "Osaka",
        "Philadelphia", "Qingdao", "Rio de Janeiro", "Saint Petersburg", "Seoul",
        "Shanghai", "Singapore", "Surat", "Suzhou", "Tianjin",
        "Tokyo", "Washington, D.C.", "Yangon"]


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,seaborn,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


@cache
def get_gauge_info(gauge='TANJONG_PAGAR'):
    """
    Get name, ID, latitude, longitude, and country of tide gauge, using location_list.lst
    (https://doi.org/10.5281/zenodo.6382554) and the PSMSL catalogue file.

    Parameters
    ----------
    gauge : int or str
        ID or name of gauge. Default is 'TANJONG_PAGAR' (equivalent to 1746).

    Returns
    -------
    gauge_info : dict
        Dictionary containing gauge_name, gauge_id, lat, lon, country.

    Notes
    -----
    This function builds on d23a-fusion by also including country information.
    """
    # Read location_list.lst into DataFrame
    in_fn = AR6_DIR / 'location_list.lst'
    in_df = pd.read_csv(in_fn, sep='\t', names=['gauge_name', 'gauge_id', 'lat', 'lon'])
    # Get data for gauge of interest
    try:
        if type(gauge) == str:
            df = in_df[in_df.gauge_name == gauge]
        else:
            df = in_df[in_df.gauge_id == gauge]
        gauge_info = dict()
        for c in ['gauge_name', 'gauge_id', 'lat', 'lon']:
            gauge_info[c] = df[c].values[0]
    except IndexError:
        raise ValueError(f"gauge='{gauge}' not found.")
    # Get country information
    psmsl_fn = PSMSL_DIR / 'nucat.dat'  # PSMSL catalogue file
    with open(psmsl_fn) as f:  # open file
        for l in f:  # loop over lines
            s_list = l.split()  # split line into strings
            try:
                if l[0:24] == '                        ' and len(s_list) >= 2:
                    country = ' '.join(s_list[1:])  # most recent country heading read
                elif s_list[0] == str(gauge_info['gauge_id']):
                    gauge_info['country'] = country  # if gauge found, save most recently-read country info
            except IndexError:
                pass
    return gauge_info


@cache
def get_coastal_loc_df():
    """
    Return AR6 projections locations with (i) data AND (ii) a gauge or a coastal city.

    Returns
    -------
    coastal_locations_df : DataFrame
        Dataframe of locations (AR6 projections location code, latitude, longitude)

    Note
    ----
    This function provides input to get_sl_qfs().
    """
    # Read example AR6 projections file
    in_dir = AR6_DIR / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / 'wf_1e' / 'ssp585'
    in_fn = in_dir / 'total-workflow.nc'
    ar6_ds = xr.open_dataset(in_fn)
    # Drop locations with missing data (including gauge locations with missing data)
    ar6_ds = ar6_ds.dropna(dim='locations', how='any')
    # Gauge locations with data
    gauge_ds = ar6_ds.sel(locations=slice(0, int(1e8)))
    gauge_loc_df = pd.DataFrame()
    gauge_loc_df['lat'] = gauge_ds['lat']
    gauge_loc_df['lon'] = gauge_ds['lon']
    gauge_loc_df['loc'] = gauge_ds['locations']
    gauge_loc_df = gauge_loc_df.set_index('loc').sort_index()  # set index and sort
    # City locations in world urbanisation prospects data
    in_fn = WUP18_DIR / 'WUP2018-F12-Cities_Over_300K.xls'
    cities_df = pd.read_excel(in_fn, header=16, usecols='A,C,E,G,H', index_col=None)
    cities_loc_df = pd.DataFrame()
    cities_loc_df['lat'] = cities_df['Latitude'].round().astype(int)  # round lat and lon
    cities_loc_df['lon'] = cities_df['Longitude'].round().astype(int)
    cities_loc_df['loc'] = ('10' + (90 - cities_loc_df['lat']).astype(str).str.zfill(3) + '0' +  # form 10MMM0NNN0
                            (cities_loc_df['lon'] % 360).astype(str).str.zfill(3) + '0').astype(int)
    cities_loc_df = cities_loc_df.drop_duplicates()  # drop duplications
    cities_loc_df = cities_loc_df.set_index('loc').sort_index()  # set index and sort
    # Intersection between city locations and AR6 projections data
    overlap = cities_loc_df.index.intersection(ar6_ds['locations'])
    cities_loc_df = cities_loc_df.loc[overlap]
    # Combine gauge and city locations into a single DataFrame
    coastal_loc_df = pd.concat([gauge_loc_df, cities_loc_df], axis=0)
    return coastal_loc_df


@cache
def get_sl_qfs(workflow='fusion_1e+2e', slr_str='rsl', scenario='ssp585'):
    """
    Return quantile functions corresponding to a probabilistic projection of sea-level rise.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), effective distribution (e.g.
        'effective_0.5'), mean (e.g. 'mean_1e+2e'), or fusion (e.g. 'fusion_1e+2e', default).
    slr_str : str
        Return global mean sea level ('gmsl'), relative sea level ('rsl'; default), or
        geocentric sea level without the background component ('novlm').
    scenario : str
        Options are 'ssp585' (default), 'ssp126', or 'ssp245'.

    Returns
    -------
    qfs_da : xarray DataArray
        DataArray of sea-level rise quantiles in m or mm/yr for different probability levels.

    Notes
    -----
    1. This function is based on get_sl_qf() in the d23a-fusion repository.
    2. In contrast to d23a.get_sl_qf(), which returns data for a specific year and location,
       get_sl_qfs() returns data for multiple years during the 21st century and multiple locations.
    """
    # Case 1: single workflow, corresponding to one of the alternative projections
    if workflow in ['wf_1e', 'wf_1f', 'wf_2e', 'wf_2f', 'wf_3e', 'wf_3f', 'wf_4']:
        # Read data
        if slr_str == 'gmsl':  # GMSL
            in_dir = AR6_DIR / 'ar6' / 'global' / 'dist_workflows' / workflow / scenario
        elif slr_str == 'rsl':  # RSL
            in_dir = AR6_DIR / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / workflow / scenario
        elif slr_str == 'novlm':  # RSL
            in_dir = (AR6_DIR / 'ar6-regional_novlm-distributions' / 'regional_novlm' / 'dist_workflows' / workflow
                      / scenario)
        else:
            raise ValueError(f"slr_str should be 'gmsl', 'rsl', or 'novlm', not '{slr_str}'.")
        in_fn = in_dir / 'total-workflow.nc'
        qfs_da = xr.open_dataset(in_fn)['sea_level_change']
        qfs_da = qfs_da.load(decode_cf=True)
        # Include only 21st century
        qfs_da = qfs_da.sel(years=slice(2000, 2100))
        # Keep only coastal locations of interest
        if slr_str != 'gmsl':
            coastal_loc_df = get_coastal_loc_df()
            qfs_da = qfs_da.sel(locations=coastal_loc_df.index)
        # Change units from mm to m
        qfs_da = qfs_da / 1000.
        qfs_da.attrs['units'] = 'm'
    # Case 2: lower or upper bound of low confidence p-box
    elif workflow in ['lower', 'upper']:
        # Contributing workflows (Kopp et al., GMD, 2023)
        wf_list = ['wf_1e', 'wf_2e', 'wf_3e', 'wf_4']
        # Get quantile function data for each of these workflows and scenarios
        qfs_da_list = []
        for wf in wf_list:
            qfs_da_list.append(get_sl_qfs(workflow=wf, slr_str=slr_str, scenario=scenario))
        concat_da = xr.concat(qfs_da_list, 'wf')
        # Find lower or upper bound
        if workflow == 'lower':
            qfs_da = concat_da.min(dim='wf')
        else:
            qfs_da = concat_da.max(dim='wf')
    # Case 3: Outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', slr_str=slr_str, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', slr_str=slr_str, scenario=scenario)
        # Derive outer bound
        qfs_da = xr.concat([lower_da.sel(quantiles=slice(0, 0.5)),  # lower bound below median
                            upper_da.sel(quantiles=slice(0.500001, 1))],  # upper bound above median
                           dim='quantiles')
        qfs_da.sel(quantiles=0.5).data[:] = np.nan  # median is undefined
    # Case 4: "effective" quantile function (Rohmer et al., 2019)
    elif 'effective' in workflow:
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', slr_str=slr_str, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', slr_str=slr_str, scenario=scenario)
        # Get constant weight w
        w = float(workflow.split('_')[-1])
        # Derive effective distribution
        qfs_da = w * upper_da + (1 - w) * lower_da
    # Case 5: "mean" quantile function
    elif 'mean' in workflow:
        # Get quantile function data for workflows and scenarios
        qfs_da_list = []
        for wf in [f'wf_{s}' for s in workflow.split('_')[-1].split('+')]:
            qfs_da_list.append(get_sl_qfs(workflow=wf, slr_str=slr_str, scenario=scenario))
        concat_da = xr.concat(qfs_da_list, dim='wf')
        # Derive mean
        qfs_da = concat_da.mean(dim='wf')
    # Case 6: fusion quantile function (Grandey et al., 2024)
    elif 'fusion' in workflow:
        # Get data for preferred workflow and outer bound of p-box
        if '+' in workflow:  # use mean for preferred workflow
            wf = f'mean_{workflow.split("_")[-1]}'
        else:  # use single workflow for preferred workflow
            wf = f'wf_{workflow.split("_")[-1]}'
        pref_da = get_sl_qfs(workflow=wf, slr_str=slr_str, scenario=scenario)
        outer_da = get_sl_qfs(workflow='outer', slr_str=slr_str, scenario=scenario)
        # Weighting function, with weights depending on probability p
        w_da = get_fusion_weights()
        # Derive fusion distribution; rely on automatic broadcasting/alignment
        qfs_da = w_da * pref_da + (1 - w_da) * outer_da
        # Correct median (which is currently nan due to nan in outer_da)
        qfs_da.sel(quantiles=0.5).data[:] = pref_da.sel(quantiles=0.5).data[:]
        # Include name and units
        qfs_da = qfs_da.rename('sea_level_change')
        qfs_da.attrs['units'] = 'm'
    # Return result
    return qfs_da


@cache
def get_fusion_weights():
    """
    Return trapezoidal weighting function for fusion.

    Returns
    -------
    w_da : xarray DataArray
        DataArray of weights for preferred workflow, with weights depending on probability.
    """
    # Get a quantile function corresponding to a projection of total sea level, to use as template
    qfs_da = get_sl_qfs(workflow='wf_1e', slr_str='gmsl', scenario='ssp585').copy()
    w_da = qfs_da.sel(years=2100).squeeze()
    # Update data to follow trapezoidal weighting function, with weights depending on probability
    da1 = w_da.sel(quantiles=slice(0, 0.169999))
    da1[:] = da1.quantiles / 0.17
    da2 = w_da.sel(quantiles=slice(0.17, 0.83))
    da2[:] = 1.
    da3 = w_da.sel(quantiles=slice(0.830001, 1))
    da3[:] = (1 - da3.quantiles) / 0.17
    w_da = xr.concat([da1, da2, da3], dim='quantiles')
    # Rename
    w_da = w_da.rename('weights')
    return w_da


def write_time_series_da(slr_str='rsl', proj_str='fusion-ssp585'):
    """
    Get and write sea-level projection time-series DataArray to NetCDF.

    Parameters
    ----------
    slr_str : str
        Return global mean sea level ('gmsl'), relative sea level ('rsl'; default), or
        geocentric sea level without the background component ('novlm').
    proj_str : str
        Probabilistic fusion under a specified scenario ('fusion-ssp585', 'fusion-ssp126'), low, central, high, or
        high-end projection.

    Returns
    -------
    out_fn : Path
        Name of written NetCDF file.
    """
    # Case 1: full probabilistic fusion projection.
    if 'fusion' in proj_str:
        scenario = proj_str.split('-')[1]
        time_series_da = get_sl_qfs(workflow='fusion_1e+2e', slr_str=slr_str, scenario=scenario).copy().squeeze()
    # Case 2: low, central, high, or high-end projection
    else:
        if proj_str == 'low':
            workflow, scenario, p = 'fusion_1e+2e', 'ssp126', 0.17
        elif proj_str == 'central':
            workflow, scenario, p = 'mean_1e+2e', 'ssp245', 0.5
        elif proj_str == 'high':
            workflow, scenario, p = 'fusion_1e+2e', 'ssp585', 0.83
        elif proj_str == 'high-end':
            workflow, scenario, p = 'fusion_1e+2e', 'ssp585', 0.95
        else:
            raise ValueError(f'Invalid proj_str: {proj_str}.')
        qfs_da = get_sl_qfs(workflow=workflow, slr_str=slr_str, scenario=scenario)
        time_series_da = qfs_da.sel(quantiles=p).squeeze()
    # Write to NetCDF file
    out_dir = DATA_DIR / 'time_series'
    if not out_dir.exists():
        out_dir.mkdir()
    out_fn = out_dir / f'{slr_str}_{proj_str}_d25a.nc'
    if slr_str == 'gmsl':
        print(f'Writing time_series/{out_fn.name}')
    else:
        print(f'Writing time_series/{out_fn.name} ({len(time_series_da.locations)} locations)')
    time_series_da.to_netcdf(out_fn)
    return time_series_da


def read_time_series_da(slr_str='rsl', proj_str='fusion-ssp585'):
    """
    Read sea-level projection time-series DataArray written by write_time_series_da().

    Parameters
    ----------
    slr_str : str
        Global mean sea level ('gmsl'), relative sea level ('rsl'; default), or
        geocentric sea level without the background component ('novlm').
    proj_str : str
        Probabilistic fusion under a specified scenario ('fusion-ssp585', 'fusion-ssp126'), low, central, high, or
        high-end projection.

    Returns
    -------
    time_series_da : DataArray
        Sea-level projection time-series DataArray.
    """
    # Input file
    in_dir = DATA_DIR / 'time_series'
    in_fn = in_dir / f'{slr_str}_{proj_str}_d25a.nc'
    print(f'Reading time_series/{in_fn.name}')
    # If input file does not yet exist, create it
    if not in_fn.exists():
        print(f'File {in_fn.name} not found; creating it now')
        _ = write_time_series_da(slr_str=slr_str, proj_str=proj_str)
    # Read data
    time_series_da = xr.open_dataset(in_fn)['sea_level_change']
    return time_series_da


def write_locations_info_df():
    """
    Get and write locations information DataFrame to CSV, including gauge information for gauges.

    Returns
    -------
    out_fn : Path
        Name of written CSV file.

    Notes
    -----
    In contrast to get_coastal_loc_df(), which provides input to get_sl_qfs(), this function provides info about
    locations returned by qfs_da().
    """
    # Create DataFrame to hold information about locations
    locations_info_df = pd.DataFrame(columns=['location', 'lat', 'lon', 'gauge_id', 'gauge_name', 'gauge_country'])
    # Loop over locations for which projections are available
    qfs_da = get_sl_qfs(workflow='fusion_1e+2e', slr_str='rsl', scenario='ssp585')
    for location in qfs_da.locations.data:
        # Get information about location, assuming it is a gauge location for now
        gauge_info = get_gauge_info(location)
        # Map to location info columns
        lat = gauge_info['lat']
        lon = gauge_info['lon']
        if location < 10000:  # remaining columns only relevant to gauge locations (which have IDs < 10000)
            gauge_id = gauge_info['gauge_id']
            gauge_name = gauge_info['gauge_name']
            gauge_country = gauge_info['country']
        else:
            gauge_id, gauge_name, gauge_country = None, None, None
        # Append to DataFrame
        locations_info_df.loc[len(locations_info_df)] = [location, lat, lon, gauge_id, gauge_name, gauge_country]
    # Index by location
    locations_info_df = locations_info_df.set_index('location')
    # Save to CSV
    out_dir = DATA_DIR / 'time_series'
    out_fn = out_dir / 'locations_info_d25a.csv'
    print(f'Writing time_series/{out_fn.name} ({len(locations_info_df)} locations)')
    locations_info_df.to_csv(out_fn)
    return out_fn


def read_locations_info_df():
    """
    Read locations information DataFrame written by write_locations_info_df().

    Returns
    -------
    locations_info_df : DataFrame
        Locations information DataFrame (location, lat, lon, gauge_id, gauge_name, gauge_country)
    """
    # Input file
    in_dir = DATA_DIR / 'time_series'
    in_fn = in_dir / 'locations_info_d25a.csv'
    print(f'Reading time_series/{in_fn.name}')
    # If input file does not yet exist, create it
    if not in_fn.exists():
        print(f'File {in_fn.name} not found; creating it now')
        _ = write_locations_info_df()
    # Read data
    locations_info_df = pd.read_csv(in_fn, index_col='location', dtype={'location': 'Int64', 'gauge_id': 'Int64'})
    return locations_info_df


def write_year_2100_df(slr_str='rsl', gauges_str='gauges', cities_str=None):
    """
    Get and write year-2100 low, central, high, and high-end projections for gauge/grid locations or cities.

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').
    gauges_str : str
        Use projections at gauges ('gauges'; default) or grid locations ('grid').
    cities_str : None or str
        Arrange projections by gauge/grid location (None; default), by city ('cities'), or by megacity ('megacities').

    Returns
    -------
    out_fn : Path
        Name of written CSV file.
    """
    # Read locations information DataFrame
    year_2100_df = read_locations_info_df().copy()
    # Select gauges or grid locations
    if gauges_str == 'gauges':
        year_2100_df = year_2100_df.loc[year_2100_df['gauge_id'].notnull()]
    elif gauges_str == 'grid':
        year_2100_df = year_2100_df.loc[year_2100_df['gauge_id'].isnull()]
    else:
        raise ValueError(f'Invalid gauges_str: {gauges_str}')
    # Get low, central, high, and high-end projections for 2100, rounded to the nearest cm
    for proj_str in ['low', 'central', 'high', 'high-end']:
        time_series_da = read_time_series_da(slr_str=slr_str, proj_str=proj_str)
        for location in year_2100_df.index:  # loop over gauges and save year-2100 projection to DataFrame
            year_2100_df.loc[location, proj_str] = time_series_da.sel(locations=location, years=2100).round(2).data
    # If cities_str is specified, arrange by city
    if cities_str is not None:
        if cities_str not in ('cities', 'megacities'):
            raise ValueError(f'Invalid cities_str: {cities_str}')
        # Read World Urbanisation Prospects 2018 data
        in_fn = WUP18_DIR / 'WUP2018-F12-Cities_Over_300K.xls'
        cities_df = pd.read_excel(in_fn, header=16, usecols='A,C,E,G,H,X', index_col=None)
        # Rename and reorder columns
        cities_df = cities_df.rename(columns={'Index': 'city_index', 'Country or area': 'city_country',
                                              'Urban Agglomeration': 'city_name', 'Latitude': 'city_lat',
                                              'Longitude': 'city_lon', 2025: 'population_2025_1000s'})
        cities_df = cities_df.set_index('city_index')
        cities_df = cities_df[['city_name', 'city_country', 'city_lat', 'city_lon', 'population_2025_1000s']]
        # If megacities, then only include cities identified by Tay et al. (2022)
        if cities_str == 'megacities':
            cities_df = cities_df.loc[cities_df['population_2025_1000s'] >= 5000]  # all have population >= 5 million
            pattern = '|'.join([rf'{re.escape(city)}' for city in MEGACITIES_LIST])
            cities_df = cities_df.loc[cities_df['city_name'].str.contains(pattern)]  # keep only Tay et al. cities
            for city_index in cities_df.index:  # add 'city_short' column containing short name of city
                city_name = cities_df.loc[city_index, 'city_name']
                for city_short in MEGACITIES_LIST:
                    if city_short in city_name:
                        cities_df.loc[city_index, 'city_short'] = city_short
                        break
        # Loop over these cities
        for index, row_ser in cities_df.iterrows():
            # Get data for nearby gauge / grid location
            lat0 = row_ser['city_lat']  # latitude of city
            lon0 = row_ser['city_lon']  # longitude of city
            # If projections are at gauges, calculate great-circle distance between city and locations
            if gauges_str == 'gauges':
                temp_df = year_2100_df.copy()  # copy projections data (from above)
                temp_df['gauge_distance_km'] = 6378 * np.arccos(
                    np.sin(np.radians(lat0)) * np.sin(np.radians(temp_df['lat'])) +
                    np.cos(np.radians(lat0)) * np.cos(np.radians(temp_df['lat'])) *
                    np.cos(np.radians(temp_df['lon'] - lon0)))
                temp_df = temp_df.sort_values(by=['gauge_distance_km']).reset_index()  # sort by distance
                # Save nearest gauge location data to DataFrame if distance <= 100 km
                if temp_df.loc[0, 'gauge_distance_km'] < 100.5:
                    for col in ['location', 'gauge_id', 'gauge_name', 'gauge_distance_km', 'lat', 'lon',
                                'low', 'central', 'high', 'high-end']:
                        cities_df.loc[index, col] = temp_df.loc[0, col]
            # If projections are at grid locations, save relevant grid location data to DataFrame
            else:
                lat0 = int(round(lat0))
                lon0 = int(round(lon0))
                temp_df = year_2100_df[(year_2100_df['lat'] == lat0) & (year_2100_df['lon'] == lon0)]
                temp_df = temp_df.reset_index()
                if not temp_df.empty:
                    for col in ['location', 'lat', 'lon', 'low', 'central', 'high', 'high-end']:
                        cities_df.loc[index, col] = temp_df.loc[0, col]
        # Keep only cities with projection data
        n_tot = len(cities_df)
        cities_df = cities_df.dropna(how='any')
        print(f'{len(cities_df)} out of {n_tot} {cities_str} have a {gauges_str} {slr_str} projection')
        # Round data
        for col in ['city_lat', 'city_lon']:  # round to 2 d.p.
            cities_df[col] = cities_df[col].round(2)
        for col in ['population_2025_1000s', 'location', 'gauge_id', 'gauge_distance_km']:  # round to nearest integer
            try:
                cities_df[col] = cities_df[col].round(0).astype('Int64')
            except KeyError:
                pass
        # cities_df replaces year_2100_df
        year_2100_df = cities_df.copy()
    # Save to CSV
    out_dir = DATA_DIR / 'year_2100'
    if not out_dir.exists():
        out_dir.mkdir()
    if cities_str:
        out_fn = out_dir / f'{slr_str}_{gauges_str}_{cities_str}_2100_d25a.csv'
    else:
        out_fn = out_dir / f'{slr_str}_{gauges_str}_2100_d25a.csv'
    print(f'Writing year_2100/{out_fn.name} ({len(year_2100_df)} locations)')
    year_2100_df.to_csv(out_fn)
    return out_fn


def read_year_2100_df(slr_str='rsl', gauges_str='gauges', cities_str=None):
    """
    Read year-2100 projections DataFrame written by write_year_2100_df().

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').
    gauges_str : str
        Use projections at gauges ('gauges'; default) or grid locations ('grid').
    cities_str : None or str
        Arrange projections by gauge/grid location (None; default), by city ('cities'), or by megacity ('megacities').

    Returns
    -------
    year_2100_df : DataFrame
        Year-2100 projections DataFrame
    """
    # Input file
    in_dir = DATA_DIR / 'year_2100'
    if cities_str:
        in_fn = in_dir / f'{slr_str}_{gauges_str}_{cities_str}_2100_d25a.csv'
    else:
        in_fn = in_dir / f'{slr_str}_{gauges_str}_2100_d25a.csv'
    print(f'Reading year_2100/{in_fn.name}')
    # If input file does not yet exist, create it
    if not in_fn.exists():
        print(f'File {in_fn.name} not found; creating it now')
        _ = write_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str=cities_str)
    # Read data
    year_2100_df = pd.read_csv(in_fn)
    return year_2100_df


def get_proj_2100_summary_df(gauges_cities_megacities='megacities'):
    """
    Return DataFrame summarising some of the key results of year-2100 projections across gauges/cities.

    Parameters
    ----------
    gauges_cities_megacities : str
        Gauges ('gauges'), cities ('cities'), or megacities ('megacities'; default).

    Returns
    -------
    summary_df : DataFrame
    """
    # Get year-2100 projections DataFrame
    proj_2100_df = read_proj_2100_df(gauges_cities_megacities=gauges_cities_megacities)
    # Keep only rsl and novlm columns
    proj_2100_df = proj_2100_df[['rsl_low', 'rsl_central', 'rsl_high', 'novlm_low', 'novlm_central', 'novlm_high']]
    # Remove rows with missing data
    proj_2100_df = proj_2100_df.dropna()
    # Get summary statistics using describe and round to 1 d.p.
    describe_df = proj_2100_df.describe().round(1)
    # Use these summary statistics to populate new DataFrame
    summary_df = pd.DataFrame(columns=describe_df.columns)
    for col in summary_df.columns:
        summary_df.loc['Median', col] = describe_df.loc['50%', col]
        summary_df.loc['IQR', col] = f'{describe_df.loc["25%", col]} to {describe_df.loc["75%", col]}'
        summary_df.loc['Range', col] = f'{describe_df.loc["min", col]} to {describe_df.loc["max", col]}'
        summary_df.loc['Count', col] = int(describe_df.loc['count', col])
    return summary_df


# @cache
# def get_cities_table_df():
#     """
#     Return table of city name, closest gauge, distance, and low-end, central, and high-end projections.
#     Also include summary statistics across all cities.
#
#     Returns
#     -------
#     table_df : DataFrame
#     """
#     # Get RSL projections and associated data for cities
#     rsl_df = get_info_high_low_exceed_df(slr_str='rsl', cities=True).copy()
#     # Calculate summary statistics
#     max_ser = rsl_df.max(numeric_only=True)
#     med_ser = rsl_df.median(numeric_only=True)
#     min_ser = rsl_df.min(numeric_only=True)
#     # Combine gauge name & ID in new column
#     rsl_df['gauge'] = rsl_df['gauge_name'] + ' (' + rsl_df.index.astype(int).astype(str) + ')'
#     # Add region headers and sort by region and high-end
#     rsl_df.loc[len(rsl_df)] = {'city_short': '—Asian megacities—', 'region': 'asia'}
#     rsl_df.loc[len(rsl_df)] = {'city_short': '—Other megacities—', 'region': 'other'}
#     rsl_df = rsl_df.sort_values(by=['region', 'high'], ascending=[True, False], na_position='first')
#     # Include summary statistics
#     rsl_df.loc[len(rsl_df)] = {'city_short': '—Statistics—'}
#     for ser, stat_str in [(max_ser, 'Maximum'), (med_ser, 'Median'), (min_ser, 'Minimum')]:
#         ser['city_short'] = stat_str
#         ser['distance'] = np.nan  # distance stats not required
#         rsl_df.loc[len(rsl_df)] = ser
#     # Rename columns
#     rsl_df = rsl_df.rename(columns={
#         'city_short': 'City', 'city_name': 'Full name', 'gauge': 'Gauge', 'distance': 'Distance, km',
#         'low': 'Low-end, m', 'central': 'Central, m', 'high': 'High-end, m'})
#     for high_low in ['low', 'central', 'high']:
#         for scenario in ['ssp126', 'ssp585']:
#             if high_low == 'central':
#                 new_col_name = f'Central under {SCENARIO_LABEL_DICT[scenario]}'
#             else:
#                 new_col_name = f'{high_low.title()}-end under {SCENARIO_LABEL_DICT[scenario]}'
#             rsl_df = rsl_df.rename(columns={f'p_ex_{high_low}_{scenario}': new_col_name})
#     # Use short name of city as index
#     rsl_df = rsl_df.set_index('City')
#     # Select columns of interest
#     columns = ['Full name', 'Gauge', 'Distance, km', 'Low-end, m', 'Central, m', 'High-end, m']
#     table_df = rsl_df[columns]
#     return table_df


@cache
def get_country_stats_df(slr_str='rsl', min_count=4):
    """
    Return country-level statistics across gauges for year-2100 projections.

    Parameters
    ----------
    slr_str : str
        RSL ('rsl'; default) or RSL without the background component ('novlm').
    min_count : int
        Minimum number of tide gauges required for country to be included. Default is 4.

    Returns
    -------
    country_stats_df : DataFrame
        DataFrame containing country, count (number of gauges), low_med, low_min, low_max, central_med, central_min,
        central_max, high_med, high_min, high_max.
    """
    # Get low, central, and high projections for 2100
    proj_2100_df = read_proj_2100_df(gauges_cities_megacities='gauges')
    # Groupby country and calculate count, median, min, and max
    count_df = proj_2100_df.groupby('gauge_country').count()
    med_df = proj_2100_df.groupby('gauge_country').median(numeric_only=True)
    min_df = proj_2100_df.groupby('gauge_country').min(numeric_only=True)
    max_df = proj_2100_df.groupby('gauge_country').max(numeric_only=True)
    # Reformat names of countries (for 'country' column, used when plotting)
    countries = []
    for s in count_df.index:
        if ', ' in s:
            s = ' '.join(s.split(', ')[::-1])
        s = s.title()
        s = s.replace("Of", "of")
        s = s.replace("The", "the")
        countries.append(s)
    # Save country-level stats to new DataFrame
    columns = ['gauge_country', 'country', 'count', 'low_med', 'low_min', 'low_max', 'central_med', 'central_min',
               'central_max', 'high_med', 'high_min', 'high_max']
    country_stats_df = pd.DataFrame(columns=columns)
    country_stats_df['gauge_country'] = count_df.index
    country_stats_df['country'] = countries
    country_stats_df['count'] = count_df[f'{slr_str}_high'].values
    for high_low_central in ['low', 'central', 'high']:
        country_stats_df[f'{high_low_central}_med'] = med_df[f'{slr_str}_{high_low_central}'].values
        country_stats_df[f'{high_low_central}_min'] = min_df[f'{slr_str}_{high_low_central}'].values
        country_stats_df[f'{high_low_central}_max'] = max_df[f'{slr_str}_{high_low_central}'].values
    # Remove countries with fewer gauges than min_count
    country_stats_df = country_stats_df.where(country_stats_df['count'] >= min_count).dropna()
    # Sort by high-end median and reindex
    country_stats_df = country_stats_df.sort_values(by='high_med')
    country_stats_df = country_stats_df.reset_index()
    return country_stats_df


def fig_fusion_ts(gauge_city='Bangkok', slr_str='rsl'):
    """
    Plot time series of median, likely range, and very likely range of sea level for (a) SSP5-8.5 and (b) SSP1-2.6.
    Also plot high-end and low-end projections.

    Parameters
    ----------
    gauge_city : int, str, or None.
        Name of gauge or city. Or ID of gauge. Default is 'Bangkok'. Ignored if slr_str is 'gmsl'.
    slr_str : str
        Global mean sea level ('gmsl'), RSL ('rsl'; default), or RSL without the background component ('novlm').

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharex=False, sharey=True, tight_layout=True)
    # Loop over scenarios and axes
    for i, (scenario, ax) in enumerate(zip(['ssp585', 'ssp126'], axs)):
        # Get fusion data
        if slr_str == 'gmsl':
            fusion_da = read_proj_ts_da(slr_str='gmsl', fusion_high_low_central='fusion',
                                        scenario=scenario).squeeze()
            city, gauge = None, None
        else:
            fusion_da = read_proj_ts_da(slr_str=slr_str, fusion_high_low_central='fusion',
                                        scenario=scenario)
            try:  # Is gauge parameter a tide gauge?
                fusion_da = fusion_da.sel(locations=get_gauge_info(gauge=gauge_city)['gauge_id']).squeeze()
                city = None
                gauge = gauge_city
            except ValueError:  # Or is it a city?
                city = gauge_city
                try:
                    cities_df = read_proj_2100_df(gauges_cities_megacities='cities')
                    gauge = cities_df.loc[cities_df['city_name'] == city, 'gauge_name'].values[0]
                except IndexError:  # Or is it a short name of a megacity?
                    mega_df = read_proj_2100_df(gauges_cities_megacities='megacities')
                    gauge = mega_df.loc[mega_df['city_short'] == city, 'gauge_name'].values[0]
                fusion_da = fusion_da.sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
        # Plot median, likely range, and very likely range
        ax.plot(fusion_da['years'], fusion_da.sel(quantiles=0.5), color='turquoise', alpha=1, label=f'Median')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.17), fusion_da.sel(quantiles=0.83),
                        color='turquoise', alpha=0.4, label='Likely range')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.83), fusion_da.sel(quantiles=0.95),
                        color='turquoise', alpha=0.1, label='Very likely range')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.05), fusion_da.sel(quantiles=0.17),
                        color='turquoise', alpha=0.1)
        # Plot high-end or low-end projection
        if scenario == 'ssp585':
            high_low = 'high'
            color = 'darkred'
        elif scenario == 'ssp126':
            high_low = 'low'
            color = 'darkgreen'
        proj_da = read_proj_ts_da(slr_str=slr_str, fusion_high_low_central=high_low, scenario=None)
        if slr_str != 'gmsl':
            proj_da = proj_da.sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
        ax.plot(proj_da['years'], proj_da, color=color, alpha=1, label=f'{high_low.title()}-end projection')
        # Customise plot
        ax.set_title(f'({chr(97+i)}) {SCENARIO_LABEL_DICT[scenario]}')
        ax.legend(loc='upper left', reverse=True)
        ax.set_xlim([2020, 2100])
        ax.set_xlabel('Year')
        if i == 0:
            if slr_str == 'gmsl':
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]}, m')
            elif city:
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]} near {city}, m')
            else:
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]} at {gauge.replace("_", " ").title()}, m')
        if i == 1:
            ax.tick_params(axis='y', labelright=True)
        if slr_str == 'gmsl':
            ax.set_ylim([0, 2])
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    return fig, axs


def fig_vlm_sensitivity_ts(megacity='Bangkok', vlm_rate_b=-3):
    """
    Plot time series of high-end, central, and low-end projections using (a) AR6 VLM and (b) assumed VLM rate.

    Parameters
    ----------
    megacity : str
        Name of megacity. Default is 'Bangkok'.
    vlm_rate_b : int or flt
        Assumed VLM rate to use in panel (b), in mm/yr. Default is -3.

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    # Identify nearest gauge
    cities_df = read_proj_2100_df(gauges_cities_megacities='megacities')
    gauge = cities_df.loc[cities_df['city_short'] == megacity, 'gauge_name'].values[0]
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5), sharex=False, sharey=True, tight_layout=True)
    # Loop over axes
    for i, ax in enumerate(axs):
        # Loop over high-end, central, and low-end projections
        for high_low_central, color in [('high', 'darkred'), ('central', 'lightblue'), ('low', 'darkgreen')]:
            # Get data
            if i == 0:
                proj_da = read_proj_ts_da(slr_str='rsl', fusion_high_low_central=high_low_central, scenario=None
                                          ).sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
            else:
                proj_da = read_proj_ts_da(slr_str='novlm', fusion_high_low_central=high_low_central,
                                          scenario=None
                                          ).sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
                proj_da = proj_da - vlm_rate_b * 1e-3 * (proj_da['years'] - 2005)  # add assumed VLM rate
            # Label
            if high_low_central == 'central':
                label = high_low_central.title()
            else:
                label = f'{high_low_central.title()}-end'
            # Plot
            ax.plot(proj_da['years'], proj_da, color=color, alpha=1, label=label)
        # Customise plot
        if i == 0:
            ax.set_title('(a) Using AR6 VLM component')
        else:
            ax.set_title(f'(b) Assuming VLM rate of {vlm_rate_b:.1f} mm/yr')
        ax.legend(loc='upper left')
        ax.set_xlim([2020, 2100])
        ax.set_xlabel('Year')
        if i == 0:
            ax.set_ylabel(f'{SLR_LABEL_DICT["rsl"]} near {megacity}, m')
        if i == 1:
            ax.tick_params(axis='y', labelright=True)
        if megacity == 'Bangkok':
            ax.set_ylim(0, 3.3)
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    return fig, axs


def fig_p_exceed_heatmap():
    """
    Plot heatmap table showing probability of GMSL exceeding the low-end, central, and high-end projections in 2100.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2), tight_layout=True)
    # For each combination of projection and scenario, calculate probability of exceeding projection
    p_exceed_df = pd.DataFrame()
    for high_low_central in ['low', 'central', 'high']:
        for scenario in ['ssp585', 'ssp126']:
            # Get and linearly interpolate quantile functions for fusion under specified scenario in 2100
            fusion_da = read_proj_ts_da(slr_str='gmsl', fusion_high_low_central='fusion', scenario=scenario)
            fusion_da = fusion_da.sel(years=2100)
            fusion_da = fusion_da.interp(quantiles=np.linspace(0, 1, 20001), method='linear')  # interval of 0.005%
            # Get high-end, low-end, or central projection
            proj_da = read_proj_ts_da(slr_str='gmsl', fusion_high_low_central=high_low_central, scenario=None)
            proj_val = proj_da.sel(years=2100).round(decimals=2).data
            # Find approximate probability of exceeding projection
            p_ex_da = (fusion_da > proj_val).mean(dim='quantiles')
            p_ex_val = p_ex_da.round(decimals=4).data[0]  # round to nearest 0.01%
            if high_low_central == 'central':
                p_exceed_df.loc[SCENARIO_LABEL_DICT[scenario], high_low_central.title()] = p_ex_val
            else:
                p_exceed_df.loc[SCENARIO_LABEL_DICT[scenario], f'{high_low_central.title()}-end'] = p_ex_val
    # Plot heatmap
    sns.heatmap(p_exceed_df, annot=True, fmt='.1%', cmap='inferno_r', vmin=0., vmax=1.,
                annot_kws={'weight': 'bold', 'fontsize': 'large'}, ax=ax)
    # Change colorbar labels to percentage
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0., 1.])
    cbar.set_ticklabels(['0%', '100%'])
    # Customise plot
    ax.tick_params(top=False, bottom=False, left=False, right=False, labeltop=True, labelbottom=False, rotation=0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize('large')
    ax.set_title(f'Probability of global mean SLR exceeding projection in 2100', y=1.35)
    return fig, ax


def fig_proj_2100_map(proj_col_str='rsl_high', gauges_cities_megacities='megacities', region=None):
    """
    Plot map of high-end, low-end, or central projection for 2100.

    Parameters
    ----------
    proj_col_str : str
        Name of projection column to plot. Default is 'rsl_high' (high-end projection of RSL).
    gauges_cities_megacities : str
        Gauges ('gauges'), cities ('cities'), or megacities ('megacities'; default).
    region : str or None
        If not None, plot data for a specific region (e.g. 'asia', 'other').

    Returns
    -------
    fig : figure
    ax : Axes
    """
    # Set up map
    fig = plt.figure(figsize=(6, 4), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, zorder=1)
    gl.bottom_labels = False
    gl.right_labels = False
    ax.coastlines(alpha=0.2, zorder=1)
    # Read projection data
    proj_df =  read_proj_2100_df(gauges_cities_megacities=gauges_cities_megacities)
    # Select only a specific region?
    if region:
        proj_df = proj_df[proj_df['region'] == region]
    # If megacities, plot location of megacities with no nearby tide gauge
    if gauges_cities_megacities == 'megacities':
        miss_df = proj_df[proj_df[proj_col_str].isnull()]
        print(f'Plotting {len(miss_df)} megacity locations with no gauge nearby.')
        plt.scatter(miss_df['city_lon'], miss_df['city_lat'], s=50, marker='^', c='0.5', zorder=2)
    # Plot projections
    proj_df = proj_df.dropna().sort_values(by=proj_col_str)
    print(f'Plotting projection for {len(proj_df)} locations.')
    cmap = plt.get_cmap('viridis', 10)
    cmap.set_over('yellow')
    cmap.set_under([0, 0, 0.1])
    if gauges_cities_megacities == 'gauges':
        plt.scatter(proj_df['gauge_lon'], proj_df['gauge_lat'], c=proj_df[proj_col_str],
                    s=10, marker='o', edgecolors='1.', linewidths=0.5, vmin=1, vmax=3, cmap=cmap, zorder=3)
    elif gauges_cities_megacities == 'cities':
        plt.scatter(proj_df['city_lon'], proj_df['city_lat'], c=proj_df[proj_col_str],
                    s=20, marker='o', edgecolors='1.', linewidths=0.5, vmin=1, vmax=3, cmap=cmap, zorder=3)
    else:
        plt.scatter(proj_df['city_lon'], proj_df['city_lat'], c=proj_df[proj_col_str],
                    s=100, marker='o', edgecolors='1.', linewidths=0.5, vmin=1, vmax=3, cmap=cmap, zorder=3)
    # Colorbar
    if proj_df[proj_col_str].min() < 1 and proj_df[proj_col_str].max() > 3:
        extend = 'both'
    elif proj_df[proj_col_str].min() < 1:
        extend = 'min'
    elif proj_df[proj_col_str].max() > 3:
        extend = 'max'
    else:
        extend = None
    if 'rsl_' in proj_col_str:
        label = f'{proj_col_str.split("_")[-1].title()}-end relative SLR in 2100, m'
    else:
        label = f'{proj_col_str.split("_")[-1].title()}-end geocentric SLR in 2100, m'
    cbar = plt.colorbar(orientation='horizontal', extend=extend, pad=0.05, shrink=0.7, label=label)
    cbar.ax.set_xticks(np.arange(1, 3.1, 0.2))
    # If megacities and only one region, annotate with city names
    if gauges_cities_megacities == 'megacities' and region:
        for index, row in proj_df.iterrows():
            city_short, lon, lat = row['city_short'], row['city_lon'], row['city_lat']
            size, weight = 'medium', 'bold'
            if city_short in ['Tianjin',]:  # on left
                plt.annotate(f'{city_short}   ', (lon, lat), va='center', ha='right', size=size, weight=weight)
            elif city_short in ['Seoul', 'Tokyo']:  # above
                plt.annotate(f'{city_short}', (lon, lat+1.5), va='bottom', ha='center', size=size, weight=weight)
            elif city_short in ['Osaka', 'Kolkata', 'Mumbai']:  # below
                plt.annotate(f'{city_short}', (lon, lat-1.5), va='top', ha='center', size=size, weight=weight)
            else:  # on right (default)
                plt.annotate(f'  {city_short}', (lon, lat), va='center', ha='left', size=size, weight=weight)
    return fig, ax


def fig_proj_2100_megacities():
    """
    Plot high-end, low-end, and central year-2100 RSL projections for megacities.

    Returns
    -------
    fig : figure
    ax: Axes
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), tight_layout=True)
    # Get RSL and no-VLM projections for megacities
    proj_df = read_proj_2100_df(gauges_cities_megacities='megacities')
    proj_df = proj_df.dropna()
    proj_df = proj_df.reset_index()
    # Sort by high-end RSL within each region
    proj_df.loc[len(proj_df)] = {'city_short': '—Asian megacities—', 'region': 'asia'}  # headers
    proj_df.loc[len(proj_df)] = {'city_short': '—Other megacities—', 'region': 'other'}
    proj_df = proj_df.sort_values(by=['region', 'rsl_high'], ascending=[False, True] )
    proj_df = proj_df.reset_index()
    # Plot data
    for col, label, color, marker, offset in [('rsl_high', 'High-end', 'darkred', 'x', 0.15),
                                              ('novlm_high', 'High-end without VLM', 'darkred', 'o', -0.15),
                                              ('rsl_central', 'Central', 'lightblue', 'x', 0.15),
                                              ('novlm_central', 'Central without VLM', 'lightblue', 'o', -0.15),
                                              ('rsl_low', 'Low-end', 'darkgreen', 'x', 0.15),
                                              ('novlm_low', 'Low-end without VLM', 'darkgreen', 'o', -0.15)]:
        # Plot GMSL data
        gmsl_da = read_proj_ts_da(slr_str='gmsl', fusion_high_low_central=col.split('_')[-1], scenario=None)
        gmsl = gmsl_da.sel(years=2100).data
        ax.axvline(gmsl, color=color, alpha=0.5, linestyle='--')
        if col == 'rsl_high':
            label2 = f'High-end global mean SLR'
        else:
            label2 = None
        ax.text(gmsl, proj_df.index.max()+0.3, label2,
                rotation=90, va='top', ha='right', color=color, alpha=0.5)
        # Plot RSL data
        ax.scatter(x=proj_df[col], y=proj_df.index+offset, color=color, label=label, marker=marker, s=20)
        # # Plot VLM component for high-end
        # if col == 'rsl_high':
        #     ax.hlines(proj_df.index, proj_df[f'novlm_high'], proj_df[col], color=color, alpha=0.7, label='VLM')
    # Legend
    ax.legend(loc='center right', title=None)
    # Tick labels etc
    ax.set_yticks(proj_df.index)
    ax.set_yticklabels(proj_df['city_short'], weight='bold')
    ax.set_ylim(proj_df.index.min() - 0.5, proj_df.index.max() + 0.5)
    ax.set_xlim(-0.4, 3.8)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    ax.tick_params(labelbottom=True, labeltop=True, labelleft=False, labelright=True,
                   bottom=False, top=False, right=False, left=False)
    ax.set_xlabel(f'{SLR_LABEL_DICT["rsl"]} in 2100, m')
    return fig, ax


def fig_country_stats(slr_str='rsl', min_count=4):
    """
    Plot country-level median, min, and max (across gauges) of high-end, low-end, and central RSL projections for 2100.

    Parameters
    ----------
    slr_str : str
        RSL ('rsl'; default) or RSL without the background component ('novlm').
    min_count : int
        Minimum number of tide gauges required to plot. Default is 4.

    Returns
    -------
    fig : figure
    (ax1, ax2) : tuple of Axes
    """
    # Create figure and axes
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 12), tight_layout=True)
    ax2 = ax1.twinx()  # twin axis, to split legend
    ax3 = ax1.twinx()  # twin axis, to split legend
    # Get country-level stats
    country_stats_df = get_country_stats_df(slr_str=slr_str, min_count=min_count)
    # Plot data
    for high_low_central, offset, color, ax in [('high', 0.2, 'darkred', ax1), ('central', 0, 'lightblue', ax2),
                                                ('low', -0.2, 'darkgreen', ax3)]:
        # Country-level RSL data
        y = country_stats_df.index + offset
        ax.scatter(x=country_stats_df[f'{high_low_central}_med'], y=y, color=color, label='Median', s=15)
        ax.hlines(y, country_stats_df[f'{high_low_central}_min'], country_stats_df[f'{high_low_central}_max'],
                  color=color, alpha=0.7, label='Range')
        # GMSL data
        gmsl_da = read_proj_ts_da(slr_str='gmsl', fusion_high_low_central=high_low_central, scenario=None)
        gmsl = gmsl_da.sel(years=2100).data
        ax.axvline(gmsl, color=color, alpha=0.5, linestyle='--')
        if high_low_central == 'high':
            label = f'High-end global mean SLR'
        else:
            label = None
        ax.text(gmsl+0.05, country_stats_df.index.min()-0.3, label,
                rotation=90, va='bottom', ha='left', color=color, alpha=0.5)
        # Legend
        if high_low_central == 'high':
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0.5), title=f'High-end')
        elif high_low_central == 'low':
            ax.legend(loc='lower left', bbox_to_anchor=(0, 0.5), title=f'Low-end')
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 0.45), title=high_low_central.title())
        # Tick labels etc
        ax.set_yticks(country_stats_df.index)
        ax.set_yticklabels(country_stats_df['country'], weight='bold')
        ax.set_ylim(country_stats_df.index.min() - 0.5, country_stats_df.index.max() + 0.5)
        ax.set_xlim(-2, 4)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        if high_low_central == 'high':
            ax.tick_params(labelbottom=True, labeltop=True, labelleft=False, labelright=True,
                           bottom=False, top=False, right=False, left=False)
            ax.set_xlabel(f'{SLR_LABEL_DICT[slr_str]} in 2100, m')
        else:
            ax.axis('off')
    return fig, (ax1, ax2)


def fig_rsl_vs_vlm():
    """
    Plot RSL vs VLM component of high-end projections for countries with the largest RSL ranges.

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), tight_layout=True)
    # Get year-2100 projection data for gauges
    proj_df = read_proj_2100_df(gauges_cities_megacities='gauges').copy()
    # Calculate VLM contribution to high-end projection as difference between total RSL and no-VLM RSL
    proj_df['vlm_high'] = proj_df['rsl_high'] - proj_df['novlm_high']
    # Identify countries with largest RSL ranges
    stats_df = get_country_stats_df(slr_str='rsl', min_count=4)
    stats_df['high_range'] = stats_df['high_max'] - stats_df['high_min']
    stats_df = stats_df.sort_values('high_range', ascending=False)
    countries = stats_df['gauge_country'][0:6]
    # Loop over countries and subplots
    for i, (country, ax) in enumerate(zip(countries, axs.flatten())):
        country_df = proj_df[proj_df['gauge_country'] == country]  # select data for country
        ax.scatter(country_df['vlm_high'], country_df['rsl_high'], marker='x', color='darkred', alpha=0.5)  # plot
        r2 = stats.pearsonr(country_df['vlm_high'], country_df['rsl_high'])[0] ** 2   # coefficienct of determination
        ax.text(0.05, 0.95, f'r$^2$ = {r2:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize='large')
        ax.set_title(f'\n({chr(97+i)}) {country.title()}')  # title
        ax.set_aspect('equal')  # fixed aspect ratio
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]  # range of y-axis
        xmid = np.mean(ax.get_xlim())  # middle of x-axis
        ax.set_xlim((xmid - yrange / 2.), (xmid + yrange / 2.))  # x-axis to cover same range as y-axis
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))  # ticks at interval of 0.5 m
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        if i in (0, 3):
            ax.set_ylabel('High-end relative SLR in 2100, m')  # y label
        if i in (3, 4, 5):
            ax.set_xlabel('VLM component, m')  # x label
    return fig, axs


# def fig_p_exceed():
#     """
#     Plot histogram showing probability of exceeding high-end projection at tide gauge locations.
#
#     Returns
#     -------
#     fig : figure
#     ax : Axes
#     """
#     # Create figure and axes
#     fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), tight_layout=True)
#     # Get high-end projection data
#     proj_df = get_info_high_low_exceed_df(slr_str='rsl')
#     # Loop over scenarios and plot
#     for scenario, binrange, color, hatch in [('ssp585', (-0.05, 5.25), 'darkred', '/'),
#                                              ('ssp126', (0, 5.2), 'green', None)]:
#         sns.histplot(proj_df[f'p_ex_high_{scenario}']*100, binwidth=0.1, binrange=binrange, stat='count',
#                      label=SCENARIO_LABEL_DICT[scenario], color=color, hatch=hatch, ax=ax)
#     # Customise axes etc
#     plt.xlim([0, 5.1])
#     ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
#     ax.yaxis.set_major_locator(plticker.MultipleLocator(base=100))
#     plt.xlabel('Probability of exceeding high-end RSL, %')
#     plt.ylabel('Number of tide gauge locations')
#     plt.legend()
#     return fig, ax


def name_save_fig(fig, fso='o', exts=('pdf', 'png'), close=False):
    """
    Name and save a figure, then increase counter.

    Parameters
    ----------
    fig : Figure
        Figure to save.
    fso : str
        Figure type. Either 'f' (main), 's' (supplement), or 'o' (other; default).
    exts : tuple
        Extensions to use. Default is ('pdf', 'png').
    close : bool
        Suppress output in notebook? Default is False.

    Returns
    -------
    fig_name : str
        Name of figure.

    Notes
    -----
    This function follows the version in the d23b-ice-dependence repository, building on d22a-mcdc and d23a-fusion.
    """
    # Name based on counter, then update counter (in preparation for next figure)
    if fso == 'f':
        fig_name = f'fig{next(F_NUM):02}'
    elif fso == 's':
        fig_name = f's{next(S_NUM):02}'
    else:
        fig_name = f'o{next(O_NUM):02}'
    # File location based on extension(s)
    for ext in exts:
        # Sub-directory
        sub_dir = FIG_DIR.joinpath(f'{fso}_{ext}')
        sub_dir.mkdir(exist_ok=True)
        # Save
        fig_path = sub_dir.joinpath(f'{fig_name}.{ext}')
        fig.savefig(fig_path)
        # Print file name and size
        fig_size = fig_path.stat().st_size / 1024 / 1024  # bytes -> MB
        print(f'Written {fig_name}.{ext} ({fig_size:.2f} MB)')
    # Suppress output in notebook?
    if close:
        plt.close()
    return fig_name
