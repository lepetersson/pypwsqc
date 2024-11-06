"""A collection of functions for flagging problematic time steps."""


from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr
import pandas as pd


def fz_filter(
    pws_data: npt.NDArray[np.float_], reference: npt.NDArray[np.float_], nint: int = 6
) -> npt.NDArray[np.float_]:
    """Flag faulty zeros based on a reference time series.

    This function applies the FZ filter from the R package PWSQC.
    The flag 1 means, a faulty zero has been detected. The flag -1
    means that no flagging was done because evaluation cannot be
    performed for the first `nint` values.

    Note that this code here is derived from the Python translation,
    done by Niek van Andel, of the original R code from Lotte de Vos.
    The Python code stems from here https://github.com/NiekvanAndel/QC_radar.
    Also note that the correctness of the Python code has not been
    verified and not all feature of the R implementation might be there.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring PWS data.
    nint : optional
        The number of subsequent data points which have to be zero, while
        the reference has values larger than zero, to set the flag for
        this data point to 1.

    Returns
    -------
    npt.NDArray
        time series of flags
    """
    ref_array = np.zeros(np.shape(pws_data))
    ref_array[np.where(reference > 0)] = 1

    sensor_array = np.zeros(np.shape(pws_data))
    sensor_array[np.where(pws_data > 0)] = 1
    sensor_array[np.where(pws_data == 0)] = 0

    fz_array = np.ones(np.shape(pws_data), dtype=np.float_) * -1

    for i in np.arange(nint, np.shape(pws_data)[0]):
        if sensor_array[i] > 0:
            fz_array[i] = 0
        elif fz_array[i - 1] == 1:
            fz_array[i] = 1
        # TODO: check why `< nint + 1` is used here.
        #       should `nint`` be scaled with a selectable factor?
        elif (np.sum(sensor_array[i - nint : i + 1]) > 0) or (
            np.sum(ref_array[i - nint : i + 1]) < nint + 1
        ):
            fz_array[i] = 0
        else:
            fz_array[i] = 1

    # fz_array.data[nbrs_not_nan < nstat] = -1
    return fz_array


def hi_filter(
    pws_data: npt.NDArray[np.float_],
    nbrs_not_nan: npt.NDArray[np.float_],
    reference: npt.NDArray[np.float_],
    hi_thres_a: npt.NDArray[np.float_],
    hi_thres_b: npt.NDArray[np.float_],
    n_stat=npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """High Influx filter.

    This function applies the HI filter from the R package PWSQC,
    flagging unrealistically high rainfall amounts.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    The function returns an array with zeros, ones or -1 per time step
    and station.
    The flag 0 means that no high influx has been detected.
    The flag 1 means that high influx has been detected.
    The flag -1 means that no flagging was done because not enough
    neighbouring stations are reporting rainfall to make a reliable
    evaluation.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    nbrs_not_nan
        Number of neighbouring stations reporting rainfall
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring stations
    hi_thres_a
        threshold for median rainfall of neighbouring stations [mm]
    hi_thres_b
        upper rainfall limit [mm]
    n_stat
        threshold for number of neighbours reporting rainfall

    Returns
    -------
    npt.NDArray
        time series of flags
    """
    condition1 = (reference < hi_thres_a) & (pws_data > hi_thres_b)
    condition2 = (reference >= hi_thres_a) & (
        pws_data > reference * hi_thres_b / hi_thres_a
    )

    hi_array = (condition1 | condition2).astype(int)
    return xr.where(nbrs_not_nan < n_stat, -1, hi_array)


def so_filter_one_station(da_station, da_neighbors, window_length):

    # rolling pearson correlation
    s_station = da_station.to_series()
    s_neighbors = da_neighbors.to_series()
    corr = s_station.rolling(window_length, min_periods= 1).corr(s_neighbors)
    ds = xr.Dataset.from_dataframe(pd.DataFrame({'corr': corr}))

    # create dataframe of neighboring stations
    df = da_neighbors.to_dataframe()
    df = df["rainfall"].unstack("id")

    # boolean arrays - True if a rainy time step, False if 0 or NaN.
    rainy_timestep_at_nbrs = (df > 0)
    
    # rolling sum of number of rainy timesteps in last mint period, per neighbor. 
    wet_timesteps_last_mint_period = rainy_timestep_at_nbrs.rolling(mint, min_periods=1).sum()
    
    # per time step and neighbor, does the nbr have more than mmatch wet time steps in the last mint period? (true/false)
    enough_matches_per_nbr = (wet_timesteps_last_mint_period > mmatch)
    
    # summing how many neighbors that have enough matches per time step
    nr_nbrs_with_enough_matches = enough_matches_per_nbr.sum(axis = 1)

    ds['matches'] = xr.DataArray.from_series(nr_nbrs_with_enough_matches)
    
    return ds

def so_filter(
ds_pws: npt.NDArray[np.float_],
distance_matrix: npt.NDArray[np.float_],
mint: npt.NDArray[np.float_],
mmatch: npt.NDArray[np.float_],
gamma: npt.NDArray[np.float_],
n_stat=npt.NDArray[np.float_],
max_distance = npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:

    """Station Outlier filter.

    This function applies the SO filter from the R package PWSQC,
    flagging nonsensical rainfall measurements for a specific location.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    In its original implementation, any interval with at least `mrain` 
    intervals of nonzero rainfall measurements is evaluated. 
    In this implementation, only a fixed rolling window of `mint` 
    intervals is evaluated. 

    The function returns an array with zeros, ones or -1 per time step
    and station.
    The flag 0 means that no station outlier has been detected.
    The flag 1 means that a station outlier has been detected.
    The flag -1 means that no flagging was done because not enough
    neighbouring stations are reporting rainfall to make a reliable
    evaluation or that the previous mint time steps was dry.

    Parameters
    ----------
    ds_pws
        xarray data set
    nbrs_not_nan
        Number of neighbouring stations reporting rainfall
    mint
        length of (rolling) window for correlation calculation
        [timesteps]
    mmatch
        threshold for matching rainy intervals in evaluation period
        [timesteps]
    gamma
        threshold for rolling median pearson correlation [-]
    n_stat
        threshold for number of neighbours reporting rainfall
    max_distance
        considered range around each station [m]

    Returns
    -------
    npt.NDArray
        time series of flags
    """
    for i in range(len(ds_pws.id)):
    
        ds_station = ds_pws.isel(id=i)
        pws_id = ds_station.id.values

        # picking stations within max_distnance, excluding itself, for the whole duration of the time series
        neighbor_ids = distance_matrix.id.data[(distance_matrix.sel(id=pws_id) < max_distance) & (distance_matrix.sel(id=pws_id) > 0)]

        #create data set for neighbors
        #ds_neighbors = ds_pws.sel(id=neighbor_ids).sel(time = slice('2016-05-01T00:05:00','2016-05-01T06:05:00'))
        ds_neighbors = ds_pws.sel(id=neighbor_ids)

        # if there are no observations in the time series, filter cannot be applied to the whole time series
        if ds_pws.rainfall.sel(id=pws_id).isnull().all():
            ds_pws.so_flag[i, :] = -1
            ds_pws.median_corr_nbrs[i,:] = -1
            continue

        # if there are not enough stations nearby, filter cannot be applied to the whole time series
        elif (len(neighbor_ids) < n_stat):
            ds_pws.so_flag[i, :] = -1
            ds_pws.median_corr_nbrs[i,:] = -1
            continue 
            
        else: 

        # run so-filter
            ds_so_filter = so_filter_one_station(ds_station.rainfall, ds_neighbors.rainfall, window_length=mint)

            median_correlation = ds_so_filter.corr.median(dim='id', skipna = True)
            ds_pws.median_corr_nbrs[i] = median_correlation
            
            so_array = (median_correlation < gamma).astype(int)
            
        # filter can not be applied if less than n_stat neighbors have enough matches
            ds_pws.so_flag[i] = xr.where(ds_so_filter.matches < n_stat, -1, so_array)

        # Set so_flag to -1 up to first valid index
            first_valid_time = first_non_nan_index[i].item()
            ds_pws["so_flag"][i, :first_valid_time] = -1 

        # disregard warm up period
            ds_pws.so_flag[i, first_valid_time:(first_valid_time+mint)] = -1

    return ds_pws
