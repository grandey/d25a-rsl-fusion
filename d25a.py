"""
d25a:
    Functions that support the analysis contained in the d25a-rsl-fusion repository.

Author:
    Benjamin S. Grandey, 2024–2025.

Notes:
    Much of this code is based on Grandey et al.'s d23a-fusion repository, https://doi.org/10.5281/zenodo.13627262.
"""


from functools import cache
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
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
AR6_DIR = Path.cwd() / 'data_ar6'  # directory containing AR6 input data
PSML_DIR = Path.cwd() / 'data_psmsl'  # directory containing PSMSL catalogue file
DATA_DIR = Path.cwd() / 'data_d25a'  # directory containing projections produced by data_d25a.ipynb
FIG_DIR = Path.cwd() / 'figs_d25a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,seaborn,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


# Functions used by data_d25a.ipynb

@cache
def get_gauge_info(gauge='TANJONG_PAGAR'):
    """
    Get name, ID, latitude, longitude, and country of tide gauge, using location_list.lst
    (https://doi.org/10.5281/zenodo.6382554) and PSMSL catalogue file.

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
    psmsl_fn = PSML_DIR / 'nucat.dat'  # PSMSL catalogue file
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
def get_sl_qfs(workflow='fusion_1e+2e', gmsl=False, rate=False, scenario='ssp585'):
    """
    Return quantile functions corresponding to a probabilistic projection of sea-level rise.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), effective distribution (e.g.
        'effective_0.5'), mean (e.g. 'mean_1e+2e'), or fusion (e.g. 'fusion_1e+2e', default).
    gmsl : bool
        If True, return global mean sea level. If False (default), return relative sea level at gauge locations.
    rate : bool
        If True, return rate of change. If False (default), return sea-level rise.
    scenario : str
        Options are 'ssp585' (default) or 'ssp126'.

    Returns
    -------
    qfs_da : xarray DataArray
        DataArray of sea-level rise quantiles in m or mm/yr for different probability levels.

    Notes
    -----
    1. This function is based on get_sl_qf() in the d23a-fusion repository.
    2. In contrast to d23a.get_sl_qf() which returns data for a specific year and location and year,
       get_sl_qfs() returns data for multiple years during the 21st century and gauge locations (if gmsl is not True).
    """
    # Case 1: single workflow, corresponding to one of the alternative projections
    if workflow in ['wf_1e', 'wf_1f', 'wf_2e', 'wf_2f', 'wf_3e', 'wf_3f', 'wf_4']:
        # Read data
        if rate:
            if gmsl:  # GMSL rate is not available in ar6.zip
                raise ValueError('rate=True is incompatible with gauge=None.')
            else:  # RSL rate
                in_dir = (AR6_DIR / 'ar6-regional-distributions' / 'regional' / 'dist_workflows_rates' / workflow
                          / scenario)
            in_fn = in_dir / 'total-workflow_rates.nc'
            qfs_da = xr.open_dataset(in_fn)['sea_level_change_rate']
        else:
            if gmsl:  # GMSL
                in_dir = AR6_DIR / 'ar6' / 'global' / 'dist_workflows' / workflow / scenario
            else:  # RSL
                in_dir = AR6_DIR / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / workflow / scenario
            in_fn = in_dir / 'total-workflow.nc'
            qfs_da = xr.open_dataset(in_fn)['sea_level_change']
        # Include only 21st century
        qfs_da = qfs_da.sel(years=slice(2000, 2100))
        # Exclude grid locations
        if not gmsl:
            qfs_da = qfs_da.sel(locations=slice(0, int(1e8)))
        # Change units from mm to m
        if not rate:
            qfs_da = qfs_da / 1000.
            qfs_da.attrs['units'] = 'm'
    # Case 2: lower or upper bound of low confidence p-box
    elif workflow in ['lower', 'upper']:
        # Contributing workflows (Kopp et al., GMD, 2023)
        if not rate:
            wf_list = ['wf_1e', 'wf_2e', 'wf_3e', 'wf_4']
        else:
            wf_list = ['wf_1f', 'wf_2f', 'wf_3f', 'wf_4']
        # Get quantile function data for each of these workflows and scenarios
        qfs_da_list = []
        for wf in wf_list:
            qfs_da_list.append(get_sl_qfs(workflow=wf, gmsl=gmsl, rate=rate, scenario=scenario))
        concat_da = xr.concat(qfs_da_list, 'wf')
        # Find lower or upper bound
        if workflow == 'lower':
            qfs_da = concat_da.min(dim='wf')
        else:
            qfs_da = concat_da.max(dim='wf')
    # Case 3: Outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', gmsl=gmsl, rate=rate, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', gmsl=gmsl, rate=rate, scenario=scenario)
        # Derive outer bound
        qfs_da = xr.concat([lower_da.sel(quantiles=slice(0, 0.5)),  # lower bound below median
                            upper_da.sel(quantiles=slice(0.500001, 1))],  # upper bound above median
                           dim='quantiles')
        qfs_da.sel(quantiles=0.5).data[:] = np.nan  # median is undefined
    # Case 4: "effective" quantile function (Rohmer et al., 2019)
    elif 'effective' in workflow:
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', gmsl=gmsl, rate=rate, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', gmsl=gmsl, rate=rate, scenario=scenario)
        # Get constant weight w
        w = float(workflow.split('_')[-1])
        # Derive effective distribution
        qfs_da = w * upper_da + (1 - w) * lower_da
    # Case 5: "mean" quantile function
    elif 'mean' in workflow:
        # Get quantile function data for workflows and scenarios
        qfs_da_list = []
        for wf in [f'wf_{s}' for s in workflow.split('_')[-1].split('+')]:
            qfs_da_list.append(get_sl_qfs(workflow=wf, gmsl=gmsl, rate=rate, scenario=scenario))
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
        pref_da = get_sl_qfs(workflow=wf, gmsl=gmsl, rate=rate, scenario=scenario)
        outer_da = get_sl_qfs(workflow='outer', gmsl=gmsl, rate=rate, scenario=scenario)
        # Weighting function, with weights depending on probability p
        w_da = get_fusion_weights()
        # Derive fusion distribution; rely on automatic broadcasting/alignment
        qfs_da = w_da * pref_da + (1 - w_da) * outer_da
        # Correct median (which is currently nan due to nan in outer_da)
        qfs_da.sel(quantiles=0.5).data[:] = pref_da.sel(quantiles=0.5).data[:]
        # Include name and units
        qfs_da = qfs_da.rename('sea_level_change')
        if not rate:
            qfs_da.attrs['units'] = 'm'
        else:
            qfs_da.attrs['units'] = 'mm/year'
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
    qfs_da = get_sl_qfs(workflow='wf_1e', gmsl=True, rate=False, scenario='ssp585').copy()
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


# Functions used by figs_d25a.ipynb

@cache
def read_fusion_high_low(fusion_high_low='fusion', gmsl=False, scenario='ssp585'):
    """
    Read quantile function data produced by data_d25a.ipynb.

    Parameters
    ----------
    fusion_high_low : str
        Choose whether to read full fusion ('fusion'), high-end ('high'), or low-end ('low') projection.
    scenario : str or None
        If reading fusion, options are 'ssp585' or 'ssp126'. Ignored for high-end or low-end.
    gmsl : bool
        If True, return global mean sea level. If False (default), return relative sea level at gauge locations.

    Returns
    -------
    qfs_da : xarray DataArray
        DataArray of sea-level rise quantiles in m for different probability levels.
    """
    # File to read
    if fusion_high_low == 'fusion':
        if gmsl:
            in_fn = DATA_DIR / f'gmsl_fusion_{scenario}_d25a.nc'
        else:
            in_fn = DATA_DIR / f'rsl_fusion_{scenario}_d25a.nc'
    elif fusion_high_low in ['high', 'low']:
        if gmsl:
            in_fn = DATA_DIR / f'gmsl_{fusion_high_low}_d25a.nc'
        else:
            in_fn = DATA_DIR / f'rsl_{fusion_high_low}_d25a.nc'
    # Read data
    qfs_da = xr.open_dataset(in_fn)['sea_level_change']
    return qfs_da

