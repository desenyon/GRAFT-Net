"""Package metadata test — TDD Task 1."""

from graft_net import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
