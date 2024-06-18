"""A collection of functions for flagging problematic time steps."""


from __future__ import annotations

import numpy as np
import xarray as xr


def fz_filter(
    pws_data: xr.DataArray,
    nbrs_not_nan: xr.DataArray,
    reference: xr.DataArray,
    nint: int,
    n_stat=int,
) -> xr.DataArray:
    """Faulty Zeros filter.

    This function applies the FZ filter from the R package PWSQC,
    flagging erroneous observations of zero rainfall.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    The function returns an array with zeros, ones or -1 per time step
    and station.

    The flag 0 means that no faulty zero has been detected.
    The flag 1 means that faulty zero has been detected.
    The flag -1 means that no flagging was done because not enough
    neighbouring stations are reporting rainfall to make a reliable
    evaluation.

    Parameters
    ----------
    pws_data (xr.DataArray)
        The rainfall time series of the PWS that should be flagged
    nbrs_not_nan (xr.DataArray)
        Number of neighbouring stations reporting rainfall
    reference (xr.DataArray)
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring stations
    nint (integer)
        The number of subsequent data points which have to be zero, while
        the reference has values larger than zero, to set the flag for
        this data point to 1.
    n_stat (integer)
        Threshold for number of neighbours reporting rainfall

    Returns
    -------
    xr.DataArray
        time series of flags
    """
    # initialize
    sensor_array = np.empty_like(pws_data)
    ref_array = np.empty_like(pws_data)
    fz_array = np.empty_like(pws_data)

    # Wet timestep at each station
    sensor_array[np.where(pws_data > 0)] = 1

    # Dry timestep at each station
    sensor_array[np.where(pws_data == 0)] = 0

    # Wet timesteps of the reference
    ref_array[np.where(reference > 0)] = 1

    for i in np.arange(len(pws_data.id.data)):
        for j in np.arange(len(pws_data.time.data)):
            if j < nint:
                fz_array[i, j] = -1
            elif sensor_array[i, j] > 0:
                fz_array[i, j] = 0
            elif fz_array[i, j - 1] == 1:
                fz_array[i, j] = 1
            elif (np.sum(sensor_array[i, j - nint : j + 1]) > 0) or (
                np.sum(ref_array[i, j - nint : j + 1]) < nint + 1
            ):
                fz_array[i, j] = 0
            else:
                fz_array[i, j] = 1

    fz_array = fz_array.astype(int)
    return xr.where(nbrs_not_nan < n_stat, -1, fz_array)


def hi_filter(
    pws_data: xr.DataArray,
    nbrs_not_nan: xr.DataArray,
    reference: xr.DataArray,
    hi_thres_a: int,
    hi_thres_b: int,
    n_stat=int,
) -> xr.DataArray:
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
    pws_data (xr.DataArray)
        The rainfall time series of the PWS that should be flagged
    nbrs_not_nan (xr.DataArray)
        Number of neighbouring stations reporting rainfall
    reference (xr.DataArray)
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring stations
    hi_thres_a (integer)
        Threshold for median rainfall of neighbouring stations [mm]
    hi_thres_b (integer)
        Upper rainfall limit [mm]
    n_stat (integer)
        Threshold for number of neighbours reporting rainfall

    Returns
    -------
    xr.DataArray
        time series of flags
    """
    condition1 = (reference < hi_thres_a) & (pws_data > hi_thres_b)
    condition2 = (reference >= hi_thres_a) & (
        pws_data > reference * hi_thres_b / hi_thres_a
    )

    hi_array = (condition1 | condition2).astype(int)
    return xr.where(nbrs_not_nan < n_stat, -1, hi_array)
