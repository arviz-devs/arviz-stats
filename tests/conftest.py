import pytest
from arviz_base.testing import datatree as _datatree
from arviz_base.testing import datatree_binary as _datatree_binary
from arviz_base.testing import fake_dt as _fake_dt


@pytest.fixture(scope="session")
def datatree():
    """Fixture for a general DataTree."""
    return _datatree()


@pytest.fixture(scope="session")
def datatree_binary():
    """Fixture for a DataTree with binary data."""
    return _datatree_binary()


@pytest.fixture(scope="session")
def fake_dt():
    """Fixture for a fake posterior."""
    return _fake_dt()
