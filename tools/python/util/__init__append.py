# appended to the __init__.py in the onnxruntime module's 'tools' folder from /tools/python/util/__init__append.py
import importlib.util

have_torch = importlib.util.find_spec("torch")
if have_torch:
    from .pytorch_export_helpers import infer_input_info  # noqa
