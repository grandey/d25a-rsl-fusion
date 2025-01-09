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
AR6_DIR = Path.cwd() / 'data_ar6'  # directory containing AR6 input data
PSML_DIR = Path.cwd() / 'data_psmsl'  # directory containing PSMSL catalogue file
DATA_DIR = Path.cwd() / 'data_d25a'  # directory containing projections produced by data_d25a.ipynb
FIG_DIR = Path.cwd() / 'figs_d25a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter
SSP_LABEL_DICT = {'ssp126': 'SSP1-2.6', 'ssp585': 'SSP5-8.5'}  # names of scenarios


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,seaborn,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


# Functions used by data_d25a.ipynb

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
def get_sl_qfs(workflow='fusion_1e+2e', gmsl_rsl_novlm='rsl', scenario='ssp585'):
    """
    Return quantile functions corresponding to a probabilistic projection of sea-level rise.

    Parameters
    ----------
    workflow : str
        AR6 workflow (e.g. 'wf_1e'), p-box bound ('lower', 'upper', 'outer'), effective distribution (e.g.
        'effective_0.5'), mean (e.g. 'mean_1e+2e'), or fusion (e.g. 'fusion_1e+2e', default).
    gmsl_rsl_novlm : str
        Return global mean sea level ('gmsl'), relative sea level (RSL) at gauge locations ('rsl'; default), or
        RSL without the background component ('novlm').
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
       get_sl_qfs() returns data for multiple years during the 21st century and gauge locations.
    """
    # Case 1: single workflow, corresponding to one of the alternative projections
    if workflow in ['wf_1e', 'wf_1f', 'wf_2e', 'wf_2f', 'wf_3e', 'wf_3f', 'wf_4']:
        # Read data
        if gmsl_rsl_novlm == 'gmsl':  # GMSL
            in_dir = AR6_DIR / 'ar6' / 'global' / 'dist_workflows' / workflow / scenario
        elif gmsl_rsl_novlm == 'rsl':  # RSL
            in_dir = AR6_DIR / 'ar6-regional-distributions' / 'regional' / 'dist_workflows' / workflow / scenario
        elif gmsl_rsl_novlm == 'novlm':  # RSL
            in_dir = (AR6_DIR / 'ar6-regional_novlm-distributions' / 'regional_novlm' / 'dist_workflows' / workflow
                      / scenario)
        else:
            raise ValueError(f"gmsl_rsl_vlm should be 'gmsl', 'rsl', or 'novlm', not '{gmsl_rsl_novlm}'.")
        in_fn = in_dir / 'total-workflow.nc'
        qfs_da = xr.open_dataset(in_fn)['sea_level_change']
        # Include only 21st century
        qfs_da = qfs_da.sel(years=slice(2000, 2100))
        # Exclude grid locations
        if gmsl_rsl_novlm != 'gmsl':
            qfs_da = qfs_da.sel(locations=slice(0, int(1e8)))
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
            qfs_da_list.append(get_sl_qfs(workflow=wf, gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario))
        concat_da = xr.concat(qfs_da_list, 'wf')
        # Find lower or upper bound
        if workflow == 'lower':
            qfs_da = concat_da.min(dim='wf')
        else:
            qfs_da = concat_da.max(dim='wf')
    # Case 3: Outer bound of p-box
    elif workflow == 'outer':
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
        # Derive outer bound
        qfs_da = xr.concat([lower_da.sel(quantiles=slice(0, 0.5)),  # lower bound below median
                            upper_da.sel(quantiles=slice(0.500001, 1))],  # upper bound above median
                           dim='quantiles')
        qfs_da.sel(quantiles=0.5).data[:] = np.nan  # median is undefined
    # Case 4: "effective" quantile function (Rohmer et al., 2019)
    elif 'effective' in workflow:
        # Get data for lower and upper p-box bounds
        lower_da = get_sl_qfs(workflow='lower', gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
        upper_da = get_sl_qfs(workflow='upper', gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
        # Get constant weight w
        w = float(workflow.split('_')[-1])
        # Derive effective distribution
        qfs_da = w * upper_da + (1 - w) * lower_da
    # Case 5: "mean" quantile function
    elif 'mean' in workflow:
        # Get quantile function data for workflows and scenarios
        qfs_da_list = []
        for wf in [f'wf_{s}' for s in workflow.split('_')[-1].split('+')]:
            qfs_da_list.append(get_sl_qfs(workflow=wf, gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario))
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
        pref_da = get_sl_qfs(workflow=wf, gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
        outer_da = get_sl_qfs(workflow='outer', gmsl_rsl_novlm=gmsl_rsl_novlm, scenario=scenario)
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
    qfs_da = get_sl_qfs(workflow='wf_1e', gmsl_rsl_novlm='gmsl', scenario='ssp585').copy()
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
def read_fusion_high_low(fusion_high_low='fusion', gmsl_rsl_novlm='rsl', scenario='ssp585'):
    """
    Read projection data produced by data_d25a.ipynb.

    Parameters
    ----------
    fusion_high_low : str
        Choose whether to read full fusion ('fusion'), high-end ('high'), or low-end ('low') projection.
    gmsl_rsl_novlm : str
        Global mean sea level ('gmsl'), RSL ('rsl'; default), or RSL without the background component ('novlm').
    scenario : str or None
        If reading fusion, options are 'ssp585' or 'ssp126'. Ignored for high-end or low-end.

    Returns
    -------
    proj_da : xarray DataArray
        DataArray of sea-level projection.
    """
    # File to read
    if fusion_high_low == 'fusion':
        in_fn = DATA_DIR / f'{gmsl_rsl_novlm}_fusion_{scenario}_d25a.nc'
    elif fusion_high_low in ['high', 'low']:
        in_fn = DATA_DIR / f'{gmsl_rsl_novlm}_{fusion_high_low}_d25a.nc'
    # Read data
    proj_da = xr.open_dataset(in_fn)['sea_level_change']
    return proj_da


@cache
def get_info_high_low_exceed_df(rsl_novlm='rsl'):
    """
    Return gauge info, high-end low-end projection for 2100, and the probabilities of exceeding these.

    Parameters
    ----------
    rsl_novlm : str
        RSL ('rsl'; default) or RSL without the background component ('novlm').

    Returns
    -------
    proj_df : DataFrame
        DataFrame containing gauge_id, gauge_name, lat, lon, country, high, low, p_ex_high_ssp585, p_ex_high_ssp126,
        p_ex_low_ssp585, p_ex_low_ssp126
    """
    # Read gauge info
    proj_df = pd.read_csv(DATA_DIR / 'gauge_info_d25a.csv').set_index('gauge_id')
    # Read high-end and low-end projections for the year 2100
    for high_low in ['high', 'low']:
        temp_da = read_fusion_high_low(fusion_high_low=high_low, gmsl_rsl_novlm=rsl_novlm, scenario=None)
        temp_ser = temp_da.sel(years=2100).rename({'locations': 'gauge_id'}).to_series().rename(high_low)
        proj_df = pd.merge(proj_df, temp_ser, on='gauge_id')
    # Probability of exceeding high-end and low-end projections under different scenarios
    for high_low in ['high', 'low']:
        for scenario in ['ssp585', 'ssp126']:
            # Get and linearly interpolate quantile functions for fusion under specified scenario in 2100
            fusion_da = read_fusion_high_low(fusion_high_low='fusion', gmsl_rsl_novlm=rsl_novlm, scenario=scenario
                                             ).sel(years=2100)
            fusion_da = fusion_da.interp(quantiles=np.linspace(0, 1, 20001), method='linear')  # interval of 0.005%
            # Get high-end or low-end projection
            high_low_da = read_fusion_high_low(fusion_high_low=high_low, gmsl_rsl_novlm=rsl_novlm, scenario=None)
            # Find approximate probability of exceeding high-end or low-end projection
            p_ex_da = (fusion_da > high_low_da).mean(dim='quantiles')
            p_ex_da = p_ex_da.round(decimals=4)  # round to nearest 0.01%
            p_ex_ser = p_ex_da.sel(years=2100).rename({'locations': 'gauge_id'}).to_series()
            p_ex_ser = p_ex_ser.rename(f'p_ex_{high_low}_{scenario}')
            proj_df = pd.merge(proj_df, p_ex_ser, on='gauge_id')
    return proj_df


@cache
def get_country_stats_df(rsl_novlm='rsl'):
    """
    Return country-level statistics across gauges for high- and low-end projections for 2100.

    Parameters
    ----------
    rsl_novlm : str
        RSL ('rsl'; default) or RSL without the background component ('novlm').

    Returns
    -------
    country_stats_df : DataFrame
        DataFrame containing country, count (number of gauges), high_med, high_min, high_max, low_med, low_min, low_max.
    """
    # Get high-end and low-end projections for 2100
    proj_df = get_info_high_low_exceed_df(rsl_novlm=rsl_novlm)
    # Groupby country and calculate count, median, min, and max
    count_df = proj_df.groupby('country').count()
    med_df = proj_df.groupby('country').median(numeric_only=True)
    min_df = proj_df.groupby('country').min(numeric_only=True)
    max_df = proj_df.groupby('country').max(numeric_only=True)
    # Save country-level stats to new DataFrame
    columns = ['country', 'count', 'high_med', 'high_min', 'high_max', 'low_med', 'low_min', 'low_max']
    country_stats_df = pd.DataFrame(columns=columns)
    country_stats_df['country'] = count_df.index
    country_stats_df['count'] = count_df['high'].values
    for high_low in ['high', 'low']:
        country_stats_df[f'{high_low}_med'] = med_df[high_low].values
        country_stats_df[f'{high_low}_min'] = min_df[high_low].values
        country_stats_df[f'{high_low}_max'] = max_df[high_low].values
    return country_stats_df


def fig_fusion_timeseries(gauge=None):
    """
    Plot time series of median, likely range, and very likely range of sea level for (a) SSP5-8.5 and (b) SSP1-2.6.
    Also plot high-end and low-end projections.

    Parameters
    ----------
    gauge : int, str, or None.
        ID or name of gauge. If None (default), then use global mean.

    Returns
    -------
    fig : figure
    axs : array of Axes
    """
    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3), sharex=False, sharey=True, tight_layout=True)
    # Loop over scenarios and axes
    for i, (scenario, ax) in enumerate(zip(['ssp585', 'ssp126'], axs)):
        # Plot median, likely range, and very likely range of fusion
        if gauge is None:
            qfs_da = read_fusion_high_low(fusion_high_low='fusion', gmsl_rsl_novlm='gmsl', scenario=scenario).squeeze()
        else:
            qfs_da = read_fusion_high_low(fusion_high_low='fusion', gmsl_rsl_novlm='rsl', scenario=scenario)
            qfs_da = qfs_da.sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
        ax.plot(qfs_da['years'], qfs_da.sel(quantiles=0.5), color='turquoise', alpha=1, label=f'Median')
        ax.fill_between(qfs_da['years'], qfs_da.sel(quantiles=0.17), qfs_da.sel(quantiles=0.83), color='turquoise',
                        alpha=0.4, label='Likely range')
        ax.fill_between(qfs_da['years'], qfs_da.sel(quantiles=0.83), qfs_da.sel(quantiles=0.95), color='turquoise',
                        alpha=0.1, label='Very likely range')
        ax.fill_between(qfs_da['years'], qfs_da.sel(quantiles=0.05), qfs_da.sel(quantiles=0.17), color='turquoise',
                        alpha=0.1)
        # Plot high-end or low-end projection
        if scenario == 'ssp585':
            high_low = 'high'
            color = 'darkred'
        elif scenario == 'ssp126':
            high_low = 'low'
            color = 'darkgreen'
        if gauge is None:
            proj_da = read_fusion_high_low(fusion_high_low=high_low, gmsl_rsl_novlm='gmsl', scenario=None)
        else:
            proj_da = read_fusion_high_low(fusion_high_low=high_low, gmsl_rsl_novlm='rsl', scenario=None)
            proj_da = proj_da.sel(locations=get_gauge_info(gauge=gauge)['gauge_id']).squeeze()
        ax.plot(proj_da['years'], proj_da, color=color, alpha=1,
                label=f'{high_low.title()}-end projection')
        # Customise plot
        ax.set_title(f'({chr(97+i)}) {SSP_LABEL_DICT[scenario]}')
        ax.legend(loc='upper left')
        ax.set_xlim([2020, 2100])
        ax.set_xlabel('Year')
        if i == 0:
            if gauge is None:
                ax.set_ylabel('GMSL, m')
            else:
                ax.set_ylabel(f'RSL at {gauge.replace("_", " ").title()}, m')
        if i == 1:
            ax.tick_params(axis='y', labelright=True)
    return fig, axs


def fig_high_map(high_low='high'):
    """
    Plot map of high-end or low-end projection.

    Parameters
    ----------
    high_low : str
        Choose whether to plot high-end ('high'; default) or low-end ('low') projection.

    Returns
    -------
    fig : figure
    ax : Axes
    """
    # Set up map
    fig = plt.figure(figsize=(5, 3), tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, zorder=0)
    gl.bottom_labels = False
    gl.right_labels = False
    ax.coastlines(zorder=1)
    # Read and plot projection data
    proj_df = get_info_high_low_exceed_df(rsl_novlm='rsl')
    cmap = plt.get_cmap('viridis', 10)
    cmap.set_over('orange')
    plt.scatter(proj_df['lon'], proj_df['lat'], c=proj_df[high_low], s=10, marker='o', edgecolors='1.',
                linewidths=0.5, vmin=1, vmax=3, cmap=cmap, zorder=2)
    cbar = plt.colorbar(orientation='horizontal', extend='both', pad=0.1,
                        label=f'{high_low.title()}-end RSL in 2100, m')
    cbar.ax.set_xticks(np.arange(1, 3.1, 0.2))
    return fig, ax


def fig_country_stats(rsl_novlm='rsl', min_count=4):
    """
    Plot country-level median, min, and max of high-end and low-end projections for 2100.

    Parameters
    ----------
    rsl_novlm : str
        RSL ('rsl'; default) or RSL without the background component ('novlm').
    min_count : int
        Minimum number of tide gauges required to plot. Default is 4.

    Returns
    -------
    fig : figure
    (ax1, ax2) : tuple of Axes
    """
    # Create figure and axes
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 10), tight_layout=True)
    ax2 = ax1.twinx()  # twin axis, to split legend
    # Get country-level stats
    country_stats_df = get_country_stats_df(rsl_novlm=rsl_novlm)
    # Select only countries that meet the min_count requirement
    country_stats_df = country_stats_df.where(country_stats_df['count'] >= min_count).dropna()
    # Sort by median and reindex
    country_stats_df = country_stats_df.sort_values(by='high_med')
    country_stats_df = country_stats_df.reset_index()
    # Plot data
    for high_low, offset, color, ax in [('high', 0.15, 'darkred', ax1), ('low', -0.15, 'darkgreen', ax2)]:
        # Country-level RSL data
        y = country_stats_df.index + offset
        ax.scatter(x=country_stats_df[f'{high_low}_med'], y=y, color=color, label='Median')
        ax.hlines(y, country_stats_df[f'{high_low}_min'], country_stats_df[f'{high_low}_max'], color=color, alpha=0.7,
                  label='Range')
        # GMSL data
        gmsl = read_fusion_high_low(fusion_high_low=high_low, gmsl_rsl_novlm='gmsl', scenario=None).sel(years=2100).data
        ax.axvline(gmsl, color=color, alpha=0.5, linestyle='--')
        ax.text(gmsl+0.05, country_stats_df.index.min()-0.3, f'{high_low.title()}-end GMSL',
                rotation=90, va='bottom', ha='left', color=color, alpha=0.5)
        # Legend
        if high_low == 'high':
            ax.legend(loc='lower right', bbox_to_anchor=(1, 0.5), title=f'{high_low.title()}-end')
        else:
            ax.legend(loc='lower left', bbox_to_anchor=(0, 0.5), title=f'{high_low.title()}-end')
        # Tick labels etc
        ax.set_yticks(country_stats_df.index)
        ax.set_yticklabels(country_stats_df['country'].str.title())
        ax.set_ylim(country_stats_df.index.min() - 0.5, country_stats_df.index.max() + 0.5)
        ax.set_xlim(-2, 4)
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        if high_low == 'high':
            ax.tick_params(labelbottom=True, labeltop=True, labelleft=False, labelright=True,
                           bottom=False, top=False, right=False, left=False)
            if rsl_novlm == 'rsl':
                ax.set_xlabel('RSL in 2100, m')
            elif rsl_novlm == 'novlm':
                ax.set_xlabel('RSL without VLM component in 2100, m')
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
    # Get high-end projection data and merge into single DataFrame
    rsl_df = get_info_high_low_exceed_df(rsl_novlm='rsl')
    novlm_df = get_info_high_low_exceed_df(rsl_novlm='novlm')
    merged_df = pd.merge(rsl_df, novlm_df, how='inner', on='gauge_name', suffixes=('_rsl', '_novlm'))
    # Calculate VLM contribution to high-end projection as difference between total RSL and no-VLM RSL
    merged_df['high_vlm'] = merged_df['high_rsl'] - merged_df['high_novlm']
    # Identify countries with largest RSL ranges
    stats_df = get_country_stats_df(rsl_novlm='rsl')
    stats_df['high_range'] = stats_df['high_max'] - stats_df['high_min']
    stats_df = stats_df.sort_values('high_range', ascending=False)
    countries = stats_df['country'][0:6]
    # Loop over countries and subplots
    for i, (country, ax) in enumerate(zip(countries, axs.flatten())):
        country_df = merged_df[merged_df['country_rsl'] == country]  # select data for country
        ax.scatter(country_df['high_vlm'], country_df['high_rsl'], marker='x', color='darkred', alpha=0.5)  # plot
        r2 = stats.pearsonr(country_df['high_vlm'], country_df['high_rsl'])[0] ** 2   # coefficienct of determination
        ax.text(0.05, 0.95, f'r$^2$ = {r2:.2f}', ha='left', va='top', transform=ax.transAxes, fontsize='large')
        ax.set_title(f'\n({chr(97+i)}) {country.title()}')  # title
        ax.set_aspect('equal')  # fixed aspect ratio
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]  # range of y-axis
        xmid = np.mean(ax.get_xlim())  # middle of x-axis
        ax.set_xlim((xmid - yrange / 2.), (xmid + yrange / 2.))  # x-axis to cover same range as y-axis
        ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))  # ticks at interval of 0.5 m
        ax.yaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        if i in (0, 3):
            ax.set_ylabel('High-end RSL in 2100, m')  # y label
        if i in (3, 4, 5):
            ax.set_xlabel('VLM component, m')  # x label
    return fig, axs


def fig_p_exceed():
    """
    Plot histogram showing probability of exceeding high-end projection at tide gauge locations.

    Returns
    -------
    fig : figure
    ax : Axes
    """
    # Create figure and axes
    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5), tight_layout=True)
    # Get high-end projection data
    proj_df = get_info_high_low_exceed_df(rsl_novlm='rsl')
    # Loop over scenarios and plot
    for scenario, binrange, color in [('ssp585', (-0.05, 5.25), 'red'), ('ssp126', (0, 5.2), 'blue')]:
        sns.histplot(proj_df[f'p_ex_high_{scenario}']*100, binwidth=0.1, binrange=binrange, stat='count',
                     label=SSP_LABEL_DICT[scenario], color=color, ax=ax)
    # Customise axes etc
    plt.xlim([0, 5.1])
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=100))
    plt.xlabel('Probability of exceeding high-end RSL, %')
    plt.ylabel('Number of tide gauge locations')
    plt.legend()
    return fig, ax


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
