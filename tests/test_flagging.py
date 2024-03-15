from __future__ import annotations

import numpy as np

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
