import numpy as np
import pytest
import arviz_base as azb
import arviz_stats

@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    return azb.from_dict({"posterior": {"mu": rng.normal(size=(4, 100))}})

def test_thin_accessor_integer_factor(sample_data):
    # DataArray accessor
    da = sample_data.posterior["mu"]
    assert da.azstats.thin(factor=2).sizes["draw"] == 50

    # Dataset accessor
    ds = sample_data.posterior.ds
    assert ds.azstats.thin(factor=2).sizes["draw"] == 50

    # DataTree accessor
    assert sample_data.azstats.thin(factor=2).posterior.sizes["draw"] == 50
