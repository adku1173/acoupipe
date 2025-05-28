import importlib.util

TF_FLAG = importlib.util.find_spec('tensorflow') is not None
