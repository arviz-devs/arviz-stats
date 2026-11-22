import arviz_base as azb
import numpy as np
import xarray as xr

import arviz_stats as azs


def test_diagnose_precomputed_stats():
    rng = np.random.default_rng(0)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(4, 100))},
            "sample_stats": {"energy": rng.normal(size=(4, 100))},
        }
    )
    bfmi_da = xr.DataArray([0.1, 0.15, 0.12, 0.14], dims=["chain"], coords={"chain": range(4)})
    ess_ds = xr.Dataset({"mu": xr.DataArray([10], dims=["mu"])})
    rhat_ds = xr.Dataset({"mu": xr.DataArray([1.5], dims=["mu"])})
    has_errors, _ = azs.diagnose(
        dt,
        stats={"bfmi": bfmi_da, "ess": ess_ds, "rhat": rhat_ds},
        return_diagnostics=True,
        show_diagnostics=False,
    )
    assert has_errors is True
