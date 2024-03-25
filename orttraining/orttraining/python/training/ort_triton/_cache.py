# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# from torch/_inductor/codecache.py
import base64
import functools
import getpass
import hashlib
import os
import sys
import tempfile
from types import ModuleType
from typing import Tuple


@functools.lru_cache(None)
def _cache_dir():
    return f"{tempfile.gettempdir()}/ort_triton_{getpass.getuser()}"


def _code_hash(code):
    return "c" + base64.b32encode(hashlib.sha256(code.encode("utf-8")).digest())[:51].decode("utf-8").lower()


def _get_code_path(source_code, ext, extra):
    basename = _code_hash(source_code + extra)
    subdir = os.path.join(_cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{ext}")
    return basename, subdir, path


def _write_atomic(path: str, source_code: str):
    # use a temp file for thread safety
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as f:
        f.write(source_code)
    os.rename(tmp_path, path)


def _write(source_code, ext, extra=""):
    basename, subdir, path = _get_code_path(source_code, ext, extra)
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(path):
        _write_atomic(path, source_code)
    return basename, path


class PyCodeCache:
    cache = dict()  # noqa: RUF012
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, source_code) -> ModuleType:
        key, path = _write(source_code, "py")
        if key not in cls.cache:
            with open(path) as f:
                code = compile(f.read(), path, "exec")
                mod = ModuleType(f"{__name__}.{key}")
                mod.__file__ = path
                mod.key = key
                exec(code, mod.__dict__, mod.__dict__)
                sys.modules[mod.__name__] = mod
                # another thread might set this first
                cls.cache.setdefault(key, mod)
        return cls.cache[key]


class ModuleCache:
    cache = dict()  # noqa: RUF012
    clear = staticmethod(cache.clear)

    @classmethod
    def load(cls, key_func, mod_func, *args) -> Tuple[str, ModuleType]:
        key = key_func(*args)
        if key not in cls.cache:
            func_name, mod = mod_func(*args)
            cls.cache[key] = (func_name, mod)
        return cls.cache[key]
