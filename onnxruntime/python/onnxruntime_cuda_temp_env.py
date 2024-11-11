import ctypes
import os
import platform
import site


def load_nvidia_libs():
    # Get the site-packages path where nvidia packages are installed
    site_packages_path = site.getsitepackages()[0]
    nvidia_path = os.path.join(site_packages_path, "nvidia")
    # Traverse the directory and subdirectories
    if platform.system() == "Windows":  #
        # Collect all directories under site-packages/nvidia that contain .dll files (for Windows)
        for root, _, files in os.walk(nvidia_path):
            # Add the current directory to the DLL search path
            with os.add_dll_directory(root):
                # Find all .dll files in the current directory
                dll_files = [f for f in files if f.lower().endswith(".dll")]

                for dll in dll_files:
                    dll_path = os.path.join(root, dll)
                    try:
                        # Load the DLL
                        _ = ctypes.CDLL(dll_path)
                        print(f"Loaded {dll_path}")
                    except OSError as e:
                        print(f"Failed to load {dll_path}: {e}")
    elif platform.system() == "Linux":
        import re

        # Regular expression to match .so files with optional versioning (e.g., .so, .so.1, .so.2.3)
        so_pattern = re.compile(r"\.so(\.\d+)*$")

        # Traverse the directory and subdirectories
        for root, _, files in os.walk(nvidia_path):
            for file in files:
                # Check if the file matches the .so pattern
                if so_pattern.search(file):
                    so_path = os.path.join(root, file)
                    try:
                        # Load the shared library
                        _ = ctypes.CDLL(so_path)
                        print(f"Loaded {so_path}")
                    except OSError as e:
                        print(f"Failed to load {so_path}: {e}")

    else:
        print(f"Unsupported platform to load nvidia libraries: {platform.system()}")
