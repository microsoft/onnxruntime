import os
import platform
import re
import site


class TemporaryEnv:
    def __init__(self, updates):
        self.original_env = os.environ.copy()
        os.environ.update(updates)

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ.clear()
        os.environ.update(self.original_env)


def get_nvidia_dll_paths() -> str:
    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[0]
    nvidia_path = os.path.join(site_packages_path, "nvidia")

    # Collect all directories under site-packages/nvidia that contain .dll files (for Windows)
    dll_paths = []
    for root, files in os.walk(nvidia_path):
        if any(file.endswith(".dll") for file in files):
            dll_paths.append(root)
    return os.pathsep.join(dll_paths)


def get_nvidia_so_paths() -> str:
    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[0]
    nvidia_path = os.path.join(site_packages_path, "nvidia")

    # Collect all directories under site-packages/nvidia that contain .so files (for Linux)
    so_paths = []
    # Regular expression to match `.so` optionally followed by `.` and digits
    pattern = re.compile(r"\.so(\.\d+)?$")
    for root, files in os.walk(nvidia_path):
        for file in files:
            if pattern.search(file):
                so_paths.append(root)
    return os.pathsep.join(so_paths)


def setup_temp_env_for_ort_cuda():
    # Determine platform and set up the environment accordingly
    if platform.system() == "Windows":  # Windows
        nvidia_dlls_path = get_nvidia_dll_paths()
        if nvidia_dlls_path:
            return TemporaryEnv({"PATH": nvidia_dlls_path + os.pathsep + os.environ.get("PATH")})
        else:
            return TemporaryEnv({"PATH": os.environ.get("PATH")})
    elif platform.system() == "Linux":
        nvidia_so_paths = get_nvidia_so_paths()
        if nvidia_so_paths:
            return TemporaryEnv({"LD_LIBRARY_PATH": nvidia_so_paths + os.pathsep + os.environ.get("LD_LIBRARY_PATH")})
        else:
            return TemporaryEnv({"LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH")})
    else:
        return None
