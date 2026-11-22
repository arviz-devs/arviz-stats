import numpy as np

from .helpers import importorskip

azb = importorskip("arviz_base")
xr = importorskip("xarray")

import arviz_stats as azs  # pylint: disable=unused-import, wrong-import-position


def test_diagnose_precomputed_stats():
    rng = np.random.default_rng(0)
    dt = azb.from_dict(
        {
            "posterior": {"mu": rng.normal(size=(4, 100))},
            "sample_stats": {"energy": rng.normal(size=(4, 100))},
        }
    )

    bfmi_da = xr.DataArray([0.8, 0.9, 0.85, 0.88], dims=["chain"], coords={"chain": range(4)})
    ess_ds = xr.Dataset({"mu": xr.DataArray([200], dims=["mu"])})
    rhat_ds = xr.Dataset({"mu": xr.DataArray([1.1], dims=["mu"])})

    has_errors, res = azs.diagnose(
        dt,
        stats={"bfmi": bfmi_da, "ess": ess_ds, "rhat": rhat_ds},
        return_diagnostics=True,
        show_diagnostics=False,
    )

    assert has_errors is True
    assert "mu" in res["rhat"]["bad_params"]
