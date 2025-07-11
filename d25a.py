"""
d25a:
    Functions that support the analysis contained in the d25a-rsl-fusion repository.

Author:
    Benjamin S. Grandey, 2024–2025.

Notes:
    Much of this code is based on the d23a-fusion repository, https://doi.org/10.5281/zenodo.13627262.
"""


import cartopy.crs as ccrs
import cartopy.feature
from functools import cache
import itertools
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
from pathlib import Path
import re
from scipy import stats, interpolate
import seaborn as sns
from watermark import watermark
import xarray as xr


# Matplotlib settings
sns.set_style('ticks')
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['axes.titleweight'] = 'bold'  # titles for subplots
plt.rcParams['figure.titleweight'] = 'bold'  # suptitle
plt.rcParams['figure.titlesize'] = 'x-large'  # suptitle
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True


# Constants
SCENARIO_LABEL_DICT = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5', 'ssp245': 'SSP2-4.5'}  # names of scenarios
SLR_LABEL_DICT = {'gmsl': 'Global mean SLR', 'rsl': 'Relative SLR', 'novlm': 'Geocentric SLR'}
AR6_DIR = Path.cwd() / 'data_in' / 'ar6'  # directory containing AR6 input data
PSMSL_DIR = Path.cwd() / 'data_in' / 'psmsl'  # directory containing PSMSL catalogue file
WUP18_DIR = Path.cwd() / 'data_in' / 'wup18'  # directory containing World Urbanisation Prospects 2018 data
NASA_DIR = Path.cwd() / 'data_in' / 'nasa'  # directory containing the NASA Distance to the Nearest Coast data
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
def get_coastal_cities_df():
    """
    Return cities within 120 km of the coast, using World Urbanisation Prospects 2018 city data.

    Returns
    -------
    cities_df : DataFrame
        Dataframe of city_index, city_name, city_country, city_lat, city_lon, population_2025_1000s, coast_distance_km

    Note
    ----
    The coast distance is based on linear interpolation of the Distance to the Nearest Coast data, using the lat and lon
    of each city. The extent of the city is not explicitly considered. The threshold of 120 km is chosen so that all
    the megacities of Tay et al. (2022) are included: Nanjing is calculated to be 120 km here.
    """
    # Read World Urbanisation Prospects 2018 data
    in_fn = WUP18_DIR / 'WUP2018-F12-Cities_Over_300K.xls'
    cities_df = pd.read_excel(in_fn, header=16, usecols='A,C,E,G,H,X', index_col=None)
    # Rename and reorder columns
    cities_df = cities_df.rename(columns={'Index': 'city_index', 'Country or area': 'city_country',
                                          'Urban Agglomeration': 'city_name', 'Latitude': 'city_lat',
                                          'Longitude': 'city_lon', 2025: 'population_2025_1000s'})
    cities_df = cities_df.set_index('city_index')
    cities_df = cities_df[['city_name', 'city_country', 'city_lat', 'city_lon', 'population_2025_1000s']]
    # Read Distance to the Nearest Coast data (https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/)
    in_fn = NASA_DIR / 'dist2coast.txt'
    dist_df = pd.read_csv(in_fn, sep='\t', names=['lon', 'lat', 'distance'])
    # Grid as Numpy array and create interpolator
    dist_pivot = dist_df.pivot(index='lat', columns='lon', values='distance')
    interp = interpolate.RegularGridInterpolator((dist_pivot.index, dist_pivot.columns), dist_pivot.values,
                                                 method='linear', bounds_error=True, fill_value=np.nan)
    # Interpolate distance using lat-lon coords of cities
    cities_coords = cities_df[['city_lat', 'city_lon']].to_numpy()
    cities_df['coast_distance_km'] = interp(cities_coords)
    cities_df['coast_distance_km'] = cities_df['coast_distance_km'].round().astype(int)
    # Keep only cities within 120 km of coast and round
    cities_df = cities_df[cities_df['coast_distance_km'] <= 120]
    return cities_df


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
    cities_df = get_coastal_cities_df().copy()
    cities_loc_df = pd.DataFrame()
    cities_loc_df['lat'] = cities_df['city_lat'].round().astype(int)  # round lat and lon
    cities_loc_df['lon'] = cities_df['city_lon'].round().astype(int)
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
        Probabilistic fusion under a specified scenario ('fusion-ssp585', 'fusion-ssp126'), low-end, low, central,
        high, or high-end projection.

    Returns
    -------
    out_fn : Path
        Name of written NetCDF file.
    """
    # Case 1: full probabilistic fusion projection.
    if 'fusion' in proj_str:
        scenario = proj_str.split('-')[1]
        time_series_da = get_sl_qfs(workflow='fusion_1e+2e', slr_str=slr_str, scenario=scenario).copy().squeeze()
    # Case 2: low-end, low, central, high, or high-end projection
    else:
        if proj_str == 'low-end':
            workflow, scenario, p = 'fusion_1e+2e', 'ssp126', 0.05
        elif proj_str == 'low':
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
        Probabilistic fusion under a specified scenario ('fusion-ssp585', 'fusion-ssp126'), low-end, low, central,
        high, or high-end projection.

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
    Get and write year-2100 low-end, low, central, high, and high-end projections for gauge/grid locations or cities.

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
    # Get low-end, low, central, high, and high-end projections for 2100, rounded to the nearest cm
    for proj_str in ['low-end', 'low', 'central', 'high', 'high-end']:
        time_series_da = read_time_series_da(slr_str=slr_str, proj_str=proj_str)
        for location in year_2100_df.index:  # loop over gauges and save year-2100 projection to DataFrame
            year_2100_df.loc[location, proj_str] = time_series_da.sel(locations=location, years=2100).round(2).data
    # If cities_str is specified, arrange by city
    if cities_str is not None:
        if cities_str not in ('cities', 'megacities'):
            raise ValueError(f'Invalid cities_str: {cities_str}')
        # Get coastal cities data
        cities_df = get_coastal_cities_df().copy()
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
        # If megacities, then identify region based on country
        if cities_str == 'megacities':
            for city_index in cities_df.index:
                country = cities_df.loc[city_index, 'city_country']
                if country in ['Japan', 'Republic of Korea', 'China', 'China, Hong Kong SAR']:
                    region = 'East Asia'
                elif country in ['Philippines', 'Viet Nam', 'Thailand', 'Myanmar', 'Indonesia', 'Singapore']:
                    region = 'Southeast Asia'
                elif country in ['Bangladesh', 'India', 'Pakistan']:
                    region = 'South Asia'
                elif country in ["Côte d'Ivoire", 'Egypt', 'Angola', 'Nigeria', 'United Republic of Tanzania']:
                    region = 'Africa'
                elif country in ['Spain', 'United Kingdom', 'Russian Federation', 'Turkey']:
                    region = 'Europe'
                elif country in ['United States of America',]:
                    region = 'North America'
                elif country in ['Argentina', 'Brazil', 'Peru']:
                    region = 'South America'
                else:
                    region = 'Unidentified'
                cities_df.loc[city_index, 'city_region'] = region
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
                                'low-end', 'low', 'central', 'high', 'high-end']:
                        cities_df.loc[index, col] = temp_df.loc[0, col]
            # If projections are at grid locations, save relevant grid location data to DataFrame
            else:
                lat0 = int(round(lat0))
                lon0 = int(round(lon0))
                temp_df = year_2100_df[(year_2100_df['lat'] == lat0) & (year_2100_df['lon'] == lon0)]
                temp_df = temp_df.reset_index()
                if not temp_df.empty:
                    for col in ['location', 'lat', 'lon', 'low-end', 'low', 'central', 'high', 'high-end']:
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


def get_year_2100_summary_df(slr_str='rsl', gauges_str='gauges', cities_str=None):
    """
    Return DataFrame summarising some of the key results of year-2100 projections across locations.

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
    summary_df : DataFrame
    """
    # Get year-2100 projections DataFrame
    year_2100_df = read_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str=cities_str)
    # Keep only low-end, low, central, high, and high-end projections columns
    year_2100_df = year_2100_df[['low-end', 'low', 'central', 'high', 'high-end']]
    # Remove rows with missing data
    year_2100_df = year_2100_df.dropna()
    # Get summary statistics using describe and round to 1 d.p.
    describe_df = year_2100_df.describe().round(1)
    # Use these summary statistics to populate new DataFrame
    summary_df = pd.DataFrame(columns=describe_df.columns)
    for col in summary_df.columns:
        summary_df.loc['Median, m', col] = describe_df.loc['50%', col]
        summary_df.loc['IQR, m', col] = f'{describe_df.loc["25%", col]} to {describe_df.loc["75%", col]}'
        summary_df.loc['Range, m', col] = f'{describe_df.loc["min", col]} to {describe_df.loc["max", col]}'
        summary_df.loc['Count', col] = int(describe_df.loc['count', col])
    # Calculate percentage of locations where projection is greater than global mean SLR
    gmsl_dict = dict()  # dictionary to hold global mean of each projection
    for proj_str in ['low-end', 'low', 'central', 'high', 'high-end']:
        gmsl = read_time_series_da(slr_str='gmsl', proj_str=proj_str).sel(years=2100).data
        gmsl_dict[proj_str] = gmsl
    perc_exceed_ser = year_2100_df.gt(pd.Series(gmsl_dict)).mean() * 100  # % of locations that exceed global mean SLR
    summary_df.loc['Proportion above global mean SLR, %'] = perc_exceed_ser.round().astype(int)
    # Calculate correlation with high-end projection
    r_ser = pd.Series()  # dictionary to hold correlation of each projection with high-end projection
    for proj_str in ['low-end', 'low', 'central', 'high', 'high-end']:
        r = year_2100_df[proj_str].corr(year_2100_df['high-end'])
        r_ser[proj_str] = r
    summary_df.loc['Correlation with high-end projection'] = r_ser.round(2)
    return summary_df


@cache
def get_country_stats_df(slr_str='rsl', min_count=4):
    """
    Return country-level statistics across gauges for year-2100 projections.

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').
    min_count : int
        Minimum number of gauges required for country to be included. Default is 4.

    Returns
    -------
    country_stats_df : DataFrame
        DataFrame containing country, count, low_med, low_min, low_max, central_med, central_min, central_max,
        high_med, high_min, high_max, high-end_med, high-end_min, high-end_max
    """
    # Get low-end, low, central, high, and high-end projections for 2100
    year_2100_df = read_year_2100_df(slr_str=slr_str, gauges_str='gauges', cities_str=None)
    # Groupby country and calculate count, median, min, and max
    count_df = year_2100_df.groupby('gauge_country').count()
    med_df = year_2100_df.groupby('gauge_country').median(numeric_only=True)
    min_df = year_2100_df.groupby('gauge_country').min(numeric_only=True)
    max_df = year_2100_df.groupby('gauge_country').max(numeric_only=True)
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
    columns = ['gauge_country', 'country', 'count', 'low-end_med', 'low-end_min', 'low-end_max',
               'low_med', 'low_min', 'low_max', 'central_med', 'central_min', 'central_max',
               'high_med', 'high_min', 'high_max', 'high-end_med', 'high-end_min', 'high-end_max']
    country_stats_df = pd.DataFrame(columns=columns)
    country_stats_df['gauge_country'] = count_df.index
    country_stats_df['country'] = countries
    country_stats_df['count'] = count_df['high-end'].values
    for proj_str in ['low-end', 'low', 'central', 'high', 'high-end']:
        country_stats_df[f'{proj_str}_med'] = med_df[proj_str].values
        country_stats_df[f'{proj_str}_min'] = min_df[proj_str].values
        country_stats_df[f'{proj_str}_max'] = max_df[proj_str].values
    # Remove countries with fewer gauges than min_count
    country_stats_df = country_stats_df.where(country_stats_df['count'] >= min_count).dropna()
    # Sort by high-end median and reindex
    country_stats_df = country_stats_df.sort_values(by='high-end_med')
    country_stats_df = country_stats_df.reset_index()
    return country_stats_df


@cache
def get_gmsl_df():
    """
    Return year-2100 global mean SLR projections and the probability of exceeding the projections.

    Returns
    -------
    gmsl_df : DataFrame
        DataFrame containing definition, gmsl_2100, p_ssp126, p_ssp585 for low-end, low, central, high, and high-end
        projections.
    """
    # Create DataFrame
    gmsl_df = pd.DataFrame()
    # Definitions
    gmsl_df.loc['high-end', 'definition'] = '95th %ile under SSP5-8.5'
    gmsl_df.loc['high', 'definition'] = '83rd %ile under SSP5-8.5'
    gmsl_df.loc['central', 'definition'] = '50th %ile under SSP2-4.5'
    gmsl_df.loc['low', 'definition'] = '17th %ile under SSP1-2.6'
    gmsl_df.loc['low-end', 'definition'] = '5th %ile under SSP1-2.6'
    # Year-2100 global mean SLR projection
    for proj_str in gmsl_df.index:
        gmsl_2100 = read_time_series_da(slr_str='gmsl', proj_str=proj_str).sel(years=2100).data
        gmsl_df.loc[proj_str, 'gmsl_2100'] = gmsl_2100
    # Probability of exceeding under SSP1-2.6 and SSP5-8.5
    for proj_str in gmsl_df.index:
        for scenario in ['ssp126', 'ssp585']:
            # Get and linearly interpolate quantile functions for fusion under scenario in 2100
            fusion_da = read_time_series_da(slr_str='gmsl', proj_str=f'fusion-{scenario}').sel(years=2100)
            fusion_da = fusion_da.interp(quantiles=np.linspace(0, 1, 200001), method='linear')  # interval of 0.0005%
            # Find probability of exceeding projection, expressed as %
            p = (fusion_da > gmsl_df.loc[proj_str, 'gmsl_2100']).mean(dim='quantiles').data
            gmsl_df.loc[proj_str, f'p_{scenario}'] = p.round(5) * 100
    return gmsl_df


def fig_fusion_time_series(slr_str='rsl', gauges_str='gauges', loc_str='TANJONG_PAGAR'):
    """
    Plot time series of median, likely range, and very likely range of sea level for (a) SSP5-8.5 and (b) SSP1-2.6.
    Also plot low-end, low, high, and high-end projections.

    Parameters
    ----------
    slr_str : str
        Global mean sea level ('gmsl'), relative sea level ('rsl'; default), or
        geocentric sea level without the background component ('novlm').
    gauges_str : str
        Use projections at gauges ('gauges'; default) or grid locations ('grid').
    loc_str : str or None.
        Name of gauge or city. Default is 'TANJONG_PAGAR'. Ignored if slr_str is 'gmsl'.

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    print(f'Called fig_fusion_time_series({slr_str}, {gauges_str}, {loc_str})')
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharex=False, sharey=True, tight_layout=True)
    # Loop over scenarios and axes
    for i, (scenario, ax) in enumerate(zip(['ssp585', 'ssp126'], axs)):
        # Get fusion data
        if slr_str == 'gmsl':
            fusion_da = read_time_series_da(slr_str=slr_str, proj_str=f'fusion-{scenario}').squeeze()
            city, location = None, None
        else:
            fusion_da = read_time_series_da(slr_str=slr_str, proj_str=f'fusion-{scenario}')
            try:  # Is gauge parameter a tide gauge?
                fusion_da = fusion_da.sel(locations=get_gauge_info(gauge=loc_str)['gauge_id']).squeeze()
                city = None
                location = get_gauge_info(gauge=loc_str)['gauge_id']
            except ValueError:  # Or is it a city?
                city = loc_str
                try:
                    year_2100_df = read_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str='cities')
                    location = year_2100_df.loc[year_2100_df['city_name'] == city, 'location'].values[0]
                except IndexError:  # Or is it a short name of a megacity?
                    year_2100_df = read_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str='megacities')
                    location = year_2100_df.loc[year_2100_df['city_short'] == city, 'location'].values[0]
                fusion_da = fusion_da.sel(locations=location).squeeze()
            # Print location info
            if i == 0:
                if gauges_str == 'gauges':
                    gauge_name = get_gauge_info(gauge=location)['gauge_name']
                    print(f'City is {city}, location is {location} ({gauge_name})')
                else:
                    print(f'City is {city}, location is {location}')
        # Plot median, likely range, and very likely range
        color = 'orange'
        ax.plot(fusion_da['years'], fusion_da.sel(quantiles=0.5), color=color, alpha=1,
                label=f'{SCENARIO_LABEL_DICT[scenario]} median')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.17), fusion_da.sel(quantiles=0.83),
                        color=color, alpha=0.4, label='Likely range')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.83), fusion_da.sel(quantiles=0.95),
                        color=color, alpha=0.1, label='Very likely range')
        ax.fill_between(fusion_da['years'], fusion_da.sel(quantiles=0.05), fusion_da.sel(quantiles=0.17),
                        color=color, alpha=0.1)
        # Plot low, high, and/or high-end projection
        if scenario == 'ssp126':
            proj_str_list = ['low', 'low-end']
            color_list = ['green', 'darkgreen']
            linestyle_list = [':', '--']
        elif scenario == 'ssp585':
            proj_str_list = ['high', 'high-end']
            color_list = ['red', 'darkred']
            linestyle_list = [':', '--']
        for proj_str, color, linestyle in zip(proj_str_list, color_list, linestyle_list):
            proj_da = read_time_series_da(slr_str=slr_str, proj_str=proj_str)
            if slr_str != 'gmsl':
                proj_da = proj_da.sel(locations=location).squeeze()
            ax.plot(proj_da['years'], proj_da, color=color, linestyle=linestyle, alpha=1, linewidth=2,
                    label=proj_str.capitalize())
        # Customise plot
        ax.set_title(f'({chr(97+i)}) {SCENARIO_LABEL_DICT[scenario]}')
        ax.legend(loc='upper left', reverse=False)
        ax.set_xlim([2020, 2100])
        ax.tick_params(axis='x', pad=10)
        ax.set_xlabel('Year')
        if i == 0:
            if slr_str == 'gmsl':
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]}, m')
            elif city:
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]} near {city}, m')
            else:
                ax.set_ylabel(f'{SLR_LABEL_DICT[slr_str]} at {gauge_name.replace("_", " ").title()}, m')
        if i == 1:
            ax.tick_params(axis='y', labelright=True)
        if slr_str == 'gmsl':
            ax.set_ylim([0, 2])
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
        ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=1))
    return fig, axs


def fig_year_2100_map(slr_str='rsl', gauges_str='grid', proj_str='high-end', diff=True, vmin=-0.5, vmax=0.5):
    """
    Plot map of low, central, high, or high-end projection for 2100.

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').
    gauges_str : str
        Use projections at grid locations ('grid'; default) or gauges ('gauges').
    proj_str : str
        'low-end', 'low', 'central', 'high', or 'high-end' (default) projection.
    diff : bool
        If true (default), subtract global mean SLR.
    vmin : int, float, or None
        Minimum for colorbar. Default is -0.5.
    vmax : int, float, or None
        Maximum for colorbar. Default is 0.5.

    Returns
    -------
    fig : figure
    ax : Axes
    """
    # Set up map
    fig = plt.figure(figsize=(12, 5), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, zorder=1, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    ax.add_feature(cartopy.feature.LAND, zorder=2)
    # Read projection data
    year_2100_df =  read_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str=None).copy()
    # Subtract global mean?
    if diff:
        gmsl_df = get_gmsl_df()
        year_2100_df[proj_str] = year_2100_df[proj_str] - gmsl_df['gmsl_2100'][proj_str]
    # Color map to use
    if diff:
        cmap = plt.get_cmap('seismic', 20)
        cmap.set_over([0.3, 0, 0])
        cmap.set_under([0, 0, 0.15])
    else:
        cmap = plt.get_cmap('viridis', 10)
        cmap.set_over('yellow')
        cmap.set_under([0, 0, 0.1])
    # Plot projection
    year_2100_df = year_2100_df.sort_values(by=proj_str)  # plot larger values last
    print(f'Plotting projection for {len(year_2100_df)} locations.')
    if gauges_str == 'grid':
        marker = 's'
    else:
        marker = 'o'
    plt.scatter(year_2100_df['lon'], year_2100_df['lat'], c=year_2100_df[proj_str],
                s=5, marker=marker, edgecolors='1.', linewidths=0.2, vmin=vmin, vmax=vmax, cmap=cmap, zorder=3)
    # Annotate megacities
    if gauges_str == 'grid':
        # Replot projection for megacities using a larger marker, so that they are visible above other points
        megacities_df = read_year_2100_df(slr_str=slr_str, gauges_str=gauges_str, cities_str='megacities').copy()
        if diff:
            megacities_df[proj_str] = megacities_df[proj_str] - gmsl_df['gmsl_2100'][proj_str]
        megacities_df = megacities_df.sort_values(by='population_2025_1000s')  # plot larger cities last
        plt.scatter(megacities_df['lon'], megacities_df['lat'], c=megacities_df[proj_str],
                    s=20, marker='o', edgecolors='1.', linewidths=0.5, vmin=vmin, vmax=vmax, cmap=cmap, zorder=4)
        # Label megacities
        offset_dict = {  # manually tune position of labels
            'Tokyo': (30, 10), 'Nagoya': (40, 0), 'Osaka': (40, -11), 'Fukuoka': (58, -20), 'Seoul': (50, 30),
            'Dalian': (25, 55), 'Tianjin': (0, 55), 'Qingdao': (10, 70),
            'Nanjing': (-30, 80), 'Shanghai': (-55, 72), 'Suzhou': (-65, 62), 'Hangzhou': (-75, 55),
            'Dongguan': (-97, 80), 'Foshan': (-102, 70), 'Guangzhou': (-115, 60), 'Hong Kong': (-120, 50),
            'Dhaka': (-65, 35), 'Chittagong': (-80, 30), 'Karachi': (-20, 10),
            'Manila': (30, 15), 'Ho Chi Minh City': (80, 7),
            'Jakarta': (20, -25), 'Singapore': (-5, -32), 'Bangkok': (-15, -50), 'Yangon': (-17, -45),
            'Kolkata': (-20, -70), 'Chennai': (-25, -60),
            'Mumbai': (-25, -60), 'Surat': (-35, -50), 'Ahmadabad': (-50, -38),
            'Dar es Salaam': (-10, -10), 'Luanda': (-25, -15), 'Lagos': (-15, -33), 'Abidjan': (-20, -15),
            'Alexandria': (-5, -35), 'Istanbul': (-20, -40), 'Barcelona': (0, -25),
            'London': (10, 30), 'Saint Petersburg': (15, -15),
            'New York': (50, 5), 'Philadelphia': (50, -5), 'Washington, D.C.': (45, -15), 'Miami': (35, -5),
            'Houston': (-10, 20), 'Los Angeles': (-20, -20),
            'Lima': (-20, -15), 'Buenos Aires': (20, -20), 'Rio de Janeiro': (20, -20)
        }
        for _, row in megacities_df.iterrows():
            city_short = row['city_short']
            try:
                offset = offset_dict[city_short]
            except KeyError:
                offset = (-20, -15)
                print(f'{city_short}: using default offset {offset}; {row["lon"]}, {row["lat"]}')
            ax.annotate(city_short, xy=(row['lon'], row['lat']),
                        xytext=offset, textcoords='offset points', ha='center', va='center', fontsize='small',
                        bbox=dict(boxstyle='round,pad=0', fc='none', ec='none'),  # reduce text padding
                        arrowprops=dict(arrowstyle='-', color='k', lw=0.5, alpha=0.5), zorder=5)
    # Colorbar
    if year_2100_df[proj_str].min() < vmin and year_2100_df[proj_str].max() > vmax:
        extend = 'both'
    elif year_2100_df[proj_str].min() < vmin:
        extend = 'min'
    elif year_2100_df[proj_str].max() > vmax:
        extend = 'max'
    else:
        extend = None
    if slr_str == 'rsl' and gauges_str == 'gauges':
        label = f'{proj_str.capitalize()} relative SLR at gauges in 2100'
    elif slr_str == 'rsl' and gauges_str == 'grid':
        label = f'{proj_str.capitalize()} relative SLR near cities in 2100'
    elif slr_str == 'novlm' and gauges_str == 'grid':
        label = f'{proj_str.capitalize()} geocentric SLR near cities in 2100'
    if diff:
        label += ' minus global mean SLR, m'
    else:
        label += ', m'
    cbar = plt.colorbar(orientation='horizontal', extend=extend, pad=0.05, shrink=0.6, label=label)
    cbar.ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    return fig, ax


def fig_year_2100_megacities(slr_str='rsl'):
    """
    Plot high-end, high, central, and low year-2100 SLR projections for megacities.

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').

    Returns
    -------
    fig : figure
    ax: Axes
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(7, 9), tight_layout=True)
    # Get SLR projections for megacities
    year_2100_df = read_year_2100_df(slr_str=slr_str, gauges_str='grid', cities_str='megacities')
    year_2100_df = year_2100_df.dropna()
    year_2100_df = year_2100_df.reset_index()
    # Sort by high-end projection
    year_2100_df = year_2100_df.sort_values(by='high-end', ascending=True)
    year_2100_df = year_2100_df.reset_index()
    # Show range, low-end to high-end
    ax.hlines(y=year_2100_df.index, xmin=year_2100_df['low-end'], xmax=year_2100_df['high-end'],
              colors='0.9', zorder=1)
    # Plot high-end etc
    for proj_str, color, marker in [('high-end', 'darkred', '^'),
                                    ('high', 'red', '2'),
                                    ('central', 'lightblue', 's'),
                                    ('low', 'green', '1'),
                                    ('low-end', 'darkgreen', 'v')]:
        # Plot GMSL data if high-end
        if proj_str == 'high-end':
            gmsl_da = read_time_series_da(slr_str='gmsl', proj_str=proj_str)
            gmsl = gmsl_da.sel(years=2100).data
            ax.axvline(gmsl, color=color, alpha=0.5, linestyle='--')
            label = f'High-end global mean SLR'
            ax.text(gmsl, year_2100_df.index.max()+0.3, label, rotation=90, va='top', ha='right',
                    color=color, alpha=0.5)
        # Plot SLR data for cities
        ax.scatter(x=year_2100_df[proj_str], y=year_2100_df.index, color=color, marker=marker, s=20,
                   label=proj_str.capitalize())
    # Legend
    ax.legend(loc='lower right', title=None)
    # Shorten some country names and combine with short city names to use as y-axis labels
    country_short_map = {'China, Hong Kong SAR': 'China', 'Russian Federation': 'Russia',
                         'United Kingdom': 'UK', 'United Republic of Tanzania': 'Tanzania',
                         'United States of America': 'USA'}
    year_2100_df['city_country'] = year_2100_df['city_country'].map(country_short_map
                                                                    ).fillna(year_2100_df['city_country'])
    yticklabels = year_2100_df['city_short'] + ', ' + year_2100_df['city_country']
    # Tick labels etc
    ax.set_yticks(year_2100_df.index)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(year_2100_df.index.min() - 0.5, year_2100_df.index.max() + 0.5)
    ax.set_xlim(-0.5, 3.2)
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.set_xlabel(f'{SLR_LABEL_DICT[slr_str]} in 2100, m')
    return fig, ax


def fig_rsl_vs_novlm(proj_str='high-end', gauges_str='grid', cities_str='megacities', lims=(1.7, 2.7)):
    """
    Plot relative SLR vs geocentric SLR globally across megacities (or cities/locations).

    Parameters
    ----------
    proj_str : str
        'low-end', 'low', 'central', 'high', or 'high-end' (default) projection.
    gauges_str : str
        Use projections at grid locations ('grid'; default) or gauges ('gauges').
    cities_str : None or str
        Arrange projections by gauge/grid location (None), city ('cities'), or megacity ('megacities'; default).
    lims : None or tuple
        x- and y-axis limits. Default is (1.7, 2.7).

    Returns
    -------
    fig : figure
    ax : Axes
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    # Get year-2100 projections of relative and geocentric SLR
    rsl_df = read_year_2100_df(slr_str='rsl', gauges_str=gauges_str, cities_str=cities_str).sort_values('location')
    novlm_df = read_year_2100_df(slr_str='novlm', gauges_str=gauges_str, cities_str=cities_str).sort_values('location')
    # Include geocentric SLR projection in 1st DataFrame, so that we can work with a single DataFrame below
    rsl_df[f'{proj_str}_novlm'] = novlm_df[proj_str]
    # If megacities, plotting includes colour (region), size (population), and labelled points (some specific points)
    if cities_str == 'megacities':
        # Plot each region separately, to simplify link between colour and label in legend
        for region_str in ['East Asia', 'Southeast Asia', 'South Asia',  # manually specify preferred order of regions
                           'Africa', 'Europe', 'North America', 'South America']:
            temp_df = rsl_df[rsl_df['city_region'] == region_str]  # select data for region
            temp_df = temp_df.sort_values('population_2025_1000s', ascending=False)  # plot smaller points last
            s = temp_df['population_2025_1000s'] * 0.002  # size depends on population
            ax.scatter(x=temp_df[f'{proj_str}_novlm'], y=temp_df[proj_str], s=s, label=region_str, alpha=0.5)  # plot
        # Legend
        ax.legend(loc='lower right', title=None, fontsize='medium')
        # Label some specific cities
        for city_str in ['Tokyo', 'Manila', 'Houston']:
            temp_ser = rsl_df[rsl_df['city_short'] == city_str].iloc[0]  # select row for city
            ax.text(temp_ser[f'{proj_str}_novlm'], temp_ser[proj_str], f'  {city_str}', va='center', ha='left')
    # If not megacities, plotting is simpler
    else:
        ax.scatter(x=rsl_df[f'{proj_str}_novlm'], y=rsl_df[proj_str], s=1, alpha=0.5)
        ax.set_title(f'cities_str = {cities_str}')  # clarify value of cities_str
    # Label axes
    if gauges_str == 'gauges':
        mod_str = 'at gauges '  # modifier string to clarify whether projections are at gauges
    else:
        mod_str = ''
    ax.set_xlabel(f'{proj_str.capitalize()} geocentric SLR {mod_str} in 2100, m')
    ax.set_ylabel(f'{proj_str.capitalize()} relative SLR {mod_str} in 2100, m')
    # Set equal axis limits
    ax.set_aspect('equal')
    if lims is None:
        lims = ax.get_ylim()  # relative SLR generally covers larger range than geocentric SLR
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[0], lims[1])
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    ax.yaxis.set_minor_locator(plticker.MultipleLocator(base=0.1))
    # Plot x = y reference line
    ax.plot(lims, lims, linestyle='-', color='k', alpha=0.1, linewidth=1, zorder=0)
    return fig, ax


def fig_country_stats(slr_str='rsl', min_count=4, high_end_only=True):
    """
    Plot country-level median, min, and max (across gauges) of high-end, high, central, and low projections for 2100.

    Parameters
    ----------
    slr_str : str
        Relative sea level ('rsl'; default) or geocentric sea level without the background component ('novlm').
    min_count : int
        Minimum number of gauges required for country to be included. Default is 4.
    high_end_only : bool
        If True, show only high-end results

    Returns
    -------
    fig : figure
    axs : tuple of Axes
    """
    # Create figure and axes
    if high_end_only:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10), tight_layout=True)
        axs = (ax1,)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 14), tight_layout=True)
        axs = (ax1, ax1.twinx(), ax1.twinx(), ax1.twinx())  # twin axis, to split legend
    # Get country-level stats
    country_stats_df = get_country_stats_df(slr_str=slr_str, min_count=min_count)
    # Plot data
    if high_end_only:
        proj_str_list = ['high-end',]
        color_list = ['darkred',]
        offset_list = [0,]
    else:
        proj_str_list = ['high-end', 'high', 'central', 'low']
        color_list = ['darkred', 'red', 'lightblue', 'darkgreen']
        offset_list = [0.2, 0.05, -0.1, -0.25]
    for proj_str, color, offset, ax in zip(proj_str_list, color_list, offset_list, axs):
        # Country-level RSL data
        y = country_stats_df.index + offset
        ax.scatter(x=country_stats_df[f'{proj_str}_med'], y=y, color=color, label='Median', s=15)
        ax.hlines(y, country_stats_df[f'{proj_str}_min'], country_stats_df[f'{proj_str}_max'],
                  color=color, alpha=0.7, label='Range')
        # GMSL data
        gmsl_da = read_time_series_da(slr_str='gmsl', proj_str=proj_str)
        gmsl = gmsl_da.sel(years=2100).data
        ax.axvline(gmsl, color=color, alpha=0.5, linestyle='--')
        if proj_str == 'high-end':
            label = f'High-end global mean SLR'
        else:
            label = None
        ax.text(gmsl+0.05, country_stats_df.index.min()-0.3, label,
                rotation=90, va='bottom', ha='left', color=color, alpha=0.5)
        # Legend
        if proj_str == 'high-end':
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0.55), title=proj_str.capitalize())
        elif proj_str == 'high':
            ax.legend(loc='upper right', bbox_to_anchor=(1, 0.45), title=proj_str.capitalize())
        elif proj_str == 'central':
            ax.legend(loc='lower left', bbox_to_anchor=(0, 0.55), title=proj_str.capitalize())
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 0.45), title=proj_str.capitalize())
        # Tick labels etc
        ax.set_yticks(country_stats_df.index)
        ax.set_yticklabels(country_stats_df['country'])
        ax.set_ylim(country_stats_df.index.min() - 0.5, country_stats_df.index.max() + 0.5)
        if high_end_only:
            ax.set_xlim(-0.5, 4)
        else:
            ax.set_xlim(-2, 4)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        if proj_str == 'high-end':
            ax.tick_params(labelbottom=True, labeltop=True, labelleft=False, labelright=True, right=True)
            if high_end_only:
                ax.set_xlabel(f'High-end {SLR_LABEL_DICT[slr_str].split()[0].lower()} SLR in 2100, m')
            else:
                ax.set_xlabel(f'{SLR_LABEL_DICT[slr_str]} in 2100, m')
        else:
            ax.axis('off')
    return fig, axs


def fig_rsl_vs_vlm():
    """
    Plot relative SLR vs VLM component of high-end projections for countries with the largest relative SLR ranges.

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(2, 3, figsize=(10, 7), tight_layout=True)
    # Get year-2100 projections of relative and geocentric SLR for gauges
    rsl_df = read_year_2100_df(slr_str='rsl', gauges_str='gauges', cities_str=None)
    novlm_df = read_year_2100_df(slr_str='novlm', gauges_str='gauges', cities_str=None)
    # Calculate VLM contribution to high-end projection as difference between total RSL and no-VLM RSL
    rsl_df['high-end_vlm'] = rsl_df['high-end'] - novlm_df['high-end']
    # Identify countries with largest RSL ranges
    stats_df = get_country_stats_df(slr_str='rsl', min_count=4)
    stats_df['high-end_range'] = stats_df['high-end_max'] - stats_df['high-end_min']
    stats_df = stats_df.sort_values('high-end_range', ascending=False)
    countries = stats_df['gauge_country'][0:6]
    # Loop over countries and subplots
    for i, (country, ax) in enumerate(zip(countries, axs.flatten())):
        country_df = rsl_df[rsl_df['gauge_country'] == country]  # select data for country
        ax.scatter(country_df['high-end_vlm'], country_df['high-end'], marker='x', color='darkred', alpha=0.5)  # plot
        r2 = stats.pearsonr(country_df['high-end_vlm'], country_df['high-end'])[0] ** 2   # coeff of determination
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
