import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from dc.network.ppa.profiles import FlatProfile, MonthlyProfile


def test_flat():
    profile = FlatProfile(0.5)

    results = profile.profile(pd.date_range("2024-01-01", "2024-03-01", freq="1h"))

    assert_array_equal(results.values, 0.5)


def test_monthly(sns):
    profile = MonthlyProfile(np.arange(12) / 12.0)

    results = profile.profile(pd.date_range("2024-01-01", periods=12, freq="1S"))

    assert_array_equal(results, np.arange(12) / 12)


def test_diurnal(sns):
    profile = FlatProfile(0.5)

    results = profile.profile(sns)
