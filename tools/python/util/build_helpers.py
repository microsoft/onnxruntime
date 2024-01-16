#  // Copyright (c) Microsoft Corporation. All rights reserved.
#  // Licensed under the MIT License.
import contextlib
import sys
import shutil
import os
from .UsageError import UsageError
from .BuildError import BuildError
from .platform_helpers import is_windows, is_macOS


def version_to_tuple(version: str) -> tuple:
    v = []
    for s in version.split("."):
        with contextlib.suppress(ValueError):
            v.append(int(s))
    return tuple(v)


def check_python_version():
    required_minor_version = 7
    if (sys.version_info.major, sys.version_info.minor) < (3, required_minor_version):
        raise UsageError(
            f"Invalid Python version. At least Python 3.{required_minor_version} is required. "
            f"Actual Python version: {sys.version}"
        )


def str_to_bool(s):
    """Convert string to bool (in argparse context)."""
    if s.lower() not in ["true", "false"]:
        raise ValueError("Need bool; got %r" % s)
    return {"true": True, "false": False}[s.lower()]


def is_reduced_ops_build(args):
    return args.include_ops_by_config is not None


def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise BuildError(f"Failed to resolve executable path for '{command_or_path}'.")
        return os.path.abspath(executable_path)
    else:
        return None


def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def use_dev_mode(args):
    if args.compile_no_warning_as_error:
        return False
    if args.use_acl:
        return False
    if args.use_armnn:
        return False
    if args.ios and is_macOS():
        return False
    SYSTEM_COLLECTIONURI = os.getenv("SYSTEM_COLLECTIONURI")  # noqa: N806
    if SYSTEM_COLLECTIONURI and SYSTEM_COLLECTIONURI != "https://dev.azure.com/onnxruntime/":
        return False
    return True


def add_default_definition(definition_list, key, default_value):
    for x in definition_list:
        if x.startswith(key + "="):
            return definition_list
    definition_list.append(key + "=" + default_value)


def normalize_arg_list(nested_list):
    return [i for j in nested_list for i in j] if nested_list else []


def number_of_parallel_jobs(args):
    return os.cpu_count() if args.parallel == 0 else args.parallel


def number_of_nvcc_threads(args):
    if args.nvcc_threads >= 0:
        return args.nvcc_threads

    nvcc_threads = 1
    
    try:
        import psutil
    except ImportError:
        print(
            "Failed to import psutil. Please `pip install psutil` for better estimation of nvcc threads. Use "
            "nvcc_threads=1"
        )
   
   available_memory = psutil.virtual_memory().available
        if isinstance(available_memory, int) and available_memory > 0:
            if available_memory > 60 * 1024 * 1024 * 1024:
                # When available memory is large enough, chance of OOM is small.
                nvcc_threads = 4
            else:
                # NVCC need a lot of memory to compile 8 flash attention cu files in Linux or 4 cutlass fmha cu files
                # in Windows. Here we select number of threads to ensure each thread has enough memory (>= 4 GB). For
                # example, Standard_NC4as_T4_v3 has 4 CPUs and 28 GB memory. When parallel=4 and nvcc_threads=2,
                # total nvcc threads is 4 * 2, which is barely able to build in 28 GB memory so we will use
                # nvcc_threads=1.
                memory_per_thread = 4 * 1024 * 1024 * 1024
                fmha_cu_files = 4 if is_windows() else 16
                fmha_parallel_jobs = min(fmha_cu_files, number_of_parallel_jobs(args))
                nvcc_threads = max(1, int(available_memory / (memory_per_thread * fmha_parallel_jobs)))
                print(
                    f"nvcc_threads={nvcc_threads} to ensure memory per thread >= 4GB for available_memory="
                    f"{available_memory} and fmha_parallel_jobs={fmha_parallel_jobs}"
                )
    except ImportError:
        print(
            "Failed to import psutil. Please `pip install psutil` for better estimation of nvcc threads. Use "
            "nvcc_threads=1"
        )

    return nvcc_threads


def setup_cann_vars(args):
    cann_home = ""

    if args.use_cann:
        cann_home = args.cann_home if args.cann_home else os.getenv("ASCEND_HOME_PATH")

        cann_home_valid = cann_home is not None and os.path.exists(cann_home)

        if not cann_home_valid:
            raise BuildError(
                "cann_home paths must be specified and valid.",
                f"cann_home='{cann_home}' valid={cann_home_valid}.",
            )

    return cann_home


def setup_tensorrt_vars(args):
    tensorrt_home = ""
    if args.use_tensorrt:
        tensorrt_home = args.tensorrt_home if args.tensorrt_home else os.getenv("TENSORRT_HOME")
        tensorrt_home_valid = tensorrt_home is not None and os.path.exists(tensorrt_home)
        if not tensorrt_home_valid:
            raise BuildError(
                "tensorrt_home paths must be specified and valid.",
                f"tensorrt_home='{tensorrt_home}' valid={tensorrt_home_valid}.",
            )

        # Set maximum workspace size in byte for
        # TensorRT (1GB = 1073741824 bytes).
        os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "1073741824"

        # Set maximum number of iterations to detect unsupported nodes
        # and partition the models for TensorRT.
        os.environ["ORT_TENSORRT_MAX_PARTITION_ITERATIONS"] = "1000"

        # Set minimum subgraph node size in graph partitioning
        # for TensorRT.
        os.environ["ORT_TENSORRT_MIN_SUBGRAPH_SIZE"] = "1"

        # Set FP16 flag
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

    return tensorrt_home


def setup_migraphx_vars(args):
    migraphx_home = None

    if args.use_migraphx:
        print(f"migraphx_home = {args.migraphx_home}")
        migraphx_home = args.migraphx_home or os.getenv("MIGRAPHX_HOME") or None

        migraphx_home_not_valid = migraphx_home and not os.path.exists(migraphx_home)

        if migraphx_home_not_valid:
            raise BuildError(
                "migraphx_home paths must be specified and valid.",
                f"migraphx_home='{migraphx_home}' valid={migraphx_home_not_valid}.",
            )
    return migraphx_home or ""


def setup_cuda_vars(args):
    cuda_home = ""
    cudnn_home = ""

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else os.getenv("CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv("CUDNN_HOME")

        cuda_home_valid = cuda_home is not None and os.path.exists(cuda_home)
        cudnn_home_valid = cudnn_home is not None and os.path.exists(cudnn_home)

        if not cuda_home_valid or (not is_windows() and not cudnn_home_valid):
            raise BuildError(
                "cuda_home and cudnn_home paths must be specified and valid.",
                "cuda_home='{}' valid={}. cudnn_home='{}' valid={}".format(
                    cuda_home, cuda_home_valid, cudnn_home, cudnn_home_valid
                ),
            )

    return cuda_home, cudnn_home
