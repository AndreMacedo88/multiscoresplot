"""Smoke tests to verify the package is importable."""


def test_import():
    import multiscoresplot

    assert hasattr(multiscoresplot, "__version__")


def test_version_string():
    from multiscoresplot import __version__

    assert isinstance(__version__, str)
    assert __version__ == "1.0.2"
