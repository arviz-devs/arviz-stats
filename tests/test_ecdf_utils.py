from numpy.testing import assert_array_almost_equal
from scipy.special._ufuncs import _hypergeom_cdf

from .helpers import importorskip

azb = importorskip("arviz_base")
from arviz_stats.ecdf_utils import hypergeometric_cdf


def test_hypergeom_cdf():
    suc = 8000
    pop = 200
    draws = 1000
    for x_vals in [0, 1, [2, 3]]:
        expected_cdf = _hypergeom_cdf(x_vals, pop, draws, suc)
        calculated_cdf = hypergeometric_cdf(x_vals, draws, suc, pop)
        assert_array_almost_equal(calculated_cdf, expected_cdf)
