import importlib

import pytest

from acoupipe.datasets import ir


def test_get_ir_falls_back_to_pyroomacoustics(monkeypatch):
    sentinel = object()

    def raise_gpurir_import_error(*args, **kwargs):
        raise ImportError('gpuRIR is not available.')

    monkeypatch.setattr(ir, 'get_ir_gpurir', raise_gpurir_import_error)
    monkeypatch.setattr(ir, 'get_ir_pyroom_acoustics', lambda *args, **kwargs: sentinel)

    result = ir.get_ir(None, None, None, None, None)

    assert result is sentinel


def test_load_pyroomacoustics_reports_optional_dependency(monkeypatch):
    import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == 'pyroomacoustics':
            raise ImportError('missing optional dependency')
        return import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)

    with pytest.raises(ImportError, match='unsupported developer-only feature'):
        ir._load_pyroomacoustics()


def test_load_gpurir_reports_manual_install(monkeypatch):
    import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == 'gpuRIR':
            raise ImportError('missing optional dependency')
        return import_module(name, package)

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)

    with pytest.raises(ImportError, match='optional GPU IR backend'):
        ir._load_gpurir()
