
import importlib


def _have_module(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec is not None

TF_FLAG = _have_module("tensorflow")
PYROOMACOUSTICS = _have_module("pyroomacoustics")
HAVE_GPURIR = _have_module("gpuRIR")

