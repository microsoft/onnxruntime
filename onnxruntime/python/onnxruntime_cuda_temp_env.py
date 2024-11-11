import ctypes
import os
import platform


class CudaTemporaryEnv:
    def __init__(self, cuda_lib_paths):
        self.loaded_libs = []
        if platform.system() == "Windows":
            for path in cuda_lib_paths:
                self.loaded_libs.append(os.add_dll_directory(path))
        elif platform.system() == "Linux":
            for path in cuda_lib_paths:
                self.loaded_libs.append(ctypes.CDLL(path))
        else:
            pass

    def __exit__(self, exc_type, exc_value, traceback):
        if platform.system() == "Windows":
            for loaded_lib in self.loaded_libs:
                loaded_lib.close()
        elif platform.system() == "Linux":
            for loaded_lib in self.loaded_libs:
                handle = loaded_lib._handle
                # Load system dynamic linking library, ctypes.CDLL(None), to access dlclose
                ctypes.CDLL(None).dlclose(handle)
        else:
            pass


def get_nvidia_lib_paths():
    import site

    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[0]
    nvidia_path = os.path.join(site_packages_path, "nvidia")
    # Collect all directories under site-packages/nvidia that contain .dll files (for Windows)
    lib_paths = []
    if platform.system() == "Windows":  # Windows
        for root, _, files in os.walk(nvidia_path):
            if any(file.endswith(".dll") for file in files):
                lib_paths.append(root)
    elif platform.system() == "Linux":
        import re

        pattern = re.compile(r"\.so(\.\d+)?$")
        for root, _, files in os.walk(nvidia_path):
            for file in files:
                if pattern.search(file):
                    lib_paths.append(root)
    else:
        pass
    return CudaTemporaryEnv(lib_paths)
