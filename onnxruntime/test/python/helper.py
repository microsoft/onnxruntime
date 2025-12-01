import os
import sys


def get_name(name):
    if os.path.exists(name):
        return name
    rel = os.path.join("testdata", name)
    if os.path.exists(rel):
        return rel
    this = os.path.dirname(__file__)
    data = os.path.join(this, "..", "testdata")
    res = os.path.join(data, name)
    if os.path.exists(res):
        return res
    raise FileNotFoundError(f"Unable to find '{name}' or '{rel}' or '{res}'")


def get_shared_library_filename_for_platform(base_name):
    if sys.platform.startswith("win"):
        return base_name + ".dll"

    if sys.platform.startswith("darwin"):
        return "lib" + base_name + ".dylib"

    # Else, assume linux
    return "lib" + base_name + ".so"
