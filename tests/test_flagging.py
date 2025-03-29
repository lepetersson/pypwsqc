from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

import pypwsqc


def test_fz_filter():
    # fmt: off
    pws_data = np.array(
      [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
       0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.101, 0.   ,
       0.   ])

    reference = np.array(
      [0.101     , 0.25136087, 0.1010425 , 0.        , 0.05012029,
       0.101     , 0.101     , 0.101     , 0.303     , 0.20048115,
       0.202     , 0.202     , 0.202     , 0.303     , 0.202     ,
       0.202     , 0.202     , 0.202     , 0.101     , 0.05012029,
       0.101     , 0.10062029, 0.0505    , 0.        , 0.        ,
       0.101     , 0.101     , 0.101     , 0.0505    , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ])

    expected = np.array(
        [-1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0])
    # fmt: on
    result = pypwsqc.flagging.fz_filter(pws_data, reference, nint=6)
    np.testing.assert_almost_equal(expected, result)

    # the same test as above but with different `nint`
    # fmt: off
    expected = np.array(
    [-1., -1., -1., -1., -1., -1.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,
      0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0])
    # fmt: on
    result = pypwsqc.flagging.fz_filter(pws_data, reference, nint=7)
    np.testing.assert_almost_equal(expected, result)


def test_hi_filter():
    # for max_distance = 10e3
    # fmt: off
    pws_data = np.array([0.       , 0.       , 0.101    , 0.       , 0.101    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 1.3130001,
       3.7370002, 0.404    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ])

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
       26, 26, 25, 25, 23, 27, 25, 23])

    reference = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.])

    expected = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.])

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        pws_data=pws_data,
        nbrs_not_nan=nbrs_not_nan,
        reference=reference,
        hi_thres_a=0.4,
        hi_thres_b=10,
        n_stat=5,
    )
    np.testing.assert_almost_equal(expected, result)

    # the same test as above but with different `hi_thres_b`
    # fmt: off
    expected = np.array([0.       , 0.       , 0.   , 0.       , 0.    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.,
       1, 0.    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ])

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        pws_data=pws_data,
        nbrs_not_nan=nbrs_not_nan,
        reference=reference,
        hi_thres_a=0.4,
        hi_thres_b=3,
        n_stat=5,
    )
    np.testing.assert_almost_equal(expected, result)

    # running test again with different IO

    # fmt: off
    pws_data = np.array([0, 0, 0, 0, 15, 0, 15, 0])

    nbrs_not_nan = np.array([2, 2, 2, 2, 12, 12, 12, 12])

    reference = np.array([0.1, 0.2, 0.35, 0.2, 0.1, 0.3, 0.2, 0.2])

    expected = np.array([-1, -1, -1, -1, 1, 0, 1, 0])

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        pws_data=pws_data,
        nbrs_not_nan=nbrs_not_nan,
        reference=reference,
        hi_thres_a=0.4,
        hi_thres_b=10,
        n_stat=5,
    )
    np.testing.assert_almost_equal(expected, result)


def test_so_filter():
    # reproduce the flags for Ams16, 2017-08-12 to 2018-10-15

    ds_pws = xr.open_dataset("test_dataset.nc")
    expected = xr.open_dataarray("expected_array.nc")
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    evaluation_period = 8064
    pws_id = "ams16"

    ds_pws["so_flag"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )
    ds_pws["median_corr_nbrs"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    result = pypwsqc.flagging.so_filter(
        ds_pws=ds_pws,
        distance_matrix=distance_matrix,
        evaluation_period=evaluation_period,
        mmatch=200,
        gamma=0.15,
        n_stat=5,
        max_distance=10e3,
    )
    result_flags = result.so_flag.isel(time=slice(evaluation_period, None)).sel(
        id=pws_id
    )

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())
