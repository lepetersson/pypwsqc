"""A collection of functions for flagging problematic time steps."""


from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr


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
    # initialize
    sensor_array = np.zeros(np.shape(pws_data))
    ref_array = np.zeros(np.shape(pws_data))
    fz_array = np.ones(np.shape(pws_data), dtype=np.float_) * -1

    # Wet timestep at each station
    sensor_array[np.where(pws_data > 0)] = 1

    # Dry timestep at each station
    sensor_array[np.where(pws_data == 0)] = 0

    # Wet timesteps of the reference
    ref_array[np.where(reference > 0)] = 1

    for i in np.arange(np.shape(pws_data)[0]):
        for j in np.arange(nint, np.shape(pws_data.time)[0]):
            if sensor_array[i, j] > 0:
                fz_array[i, j] = 0
            elif fz_array[i, j - 1] == 1:
                fz_array[i, j] = 1
            elif (np.sum(sensor_array[i, j - nint : j + 1]) > 0) or (
                np.sum(ref_array[i, j - nint : j + 1]) < nint + 1
            ):
                fz_array[i, j] = 0
            else:
                fz_array[i, j] = 1
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
