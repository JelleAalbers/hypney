from typing import Any, Optional
import pytest

import hypney.utils.eagerpy as ep_utils


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--backend")


@pytest.fixture(scope="session")
def tensorlib(request: Any):
    backend: Optional[str] = request.config.option.backend
    if backend is None:
        backend = "numpy"
    result = ep_utils.tensorlib(backend)
    if backend == "jax":
        return result.numpy
    return result
