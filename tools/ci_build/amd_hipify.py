# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import subprocess


def hipify(hipify_perl_path, src_file_path, dst_file_path):
    dir_name = os.path.dirname(dst_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    # Run hipify-perl first, capture output
    s = subprocess.run([hipify_perl_path, src_file_path], stdout=subprocess.PIPE, text=True, check=False).stdout

    # Additional exact-match replacements.
    # Order matters for all of the following replacements, reglardless of appearing in logical sections.
    s = s.replace("kCudaExecutionProvider", "kRocmExecutionProvider")
    s = s.replace("CUDAStreamType", "HIPStreamType")
    s = s.replace("kCudaStreamDefault", "kHipStreamDefault")
    s = s.replace("kCudaStreamCopyIn", "kHipStreamCopyIn")
    s = s.replace("kCudaStreamCopyOut", "kHipStreamCopyOut")
    s = s.replace("kTotalCudaStreams", "kTotalHipStreams")
    # these should be "hip" but it's easier to just use rocm to avoid complicated file renaming
    s = s.replace("CudaGraph", "RocmGraph")
    s = s.replace("CUDAGraph", "ROCMGraph")
    s = s.replace("cuda_graph", "rocm_graph")
    s = s.replace("RegisterCudaContribKernels", "RegisterRocmContribKernels")
    s = s.replace("cudaEvent", "hipEvent")
    s = s.replace("CreateCudaAllocator", "CreateRocmAllocator")
    s = s.replace("CudaErrString", "RocmErrString")
    s = s.replace("CudaAsyncBuffer", "RocmAsyncBuffer")
    s = s.replace("CudaKernel", "RocmKernel")
    s = s.replace("CudaStream", "RocmStream")
    s = s.replace("ToCudaType", "ToHipType")
    s = s.replace("CudaT", "HipT")
    s = s.replace("CUDA_LONG", "HIP_LONG")
    s = s.replace("CUDA_RETURN_IF_ERROR", "HIP_RETURN_IF_ERROR")
    s = s.replace("CUDA_KERNEL_ASSERT", "HIP_KERNEL_ASSERT")
    s = s.replace("CUDA_CALL", "HIP_CALL")
    s = s.replace("SliceCuda", "SliceRocm")
    s = s.replace("thrust::cuda", "thrust::hip")
    s = s.replace("CudaCall", "RocmCall")
    s = s.replace("cuda", "rocm")
    # s = s.replace('Cuda', 'Rocm')
    s = s.replace("CUDA", "ROCM")
    s = s.replace("GPU_WARP_SIZE = 32", "GPU_WARP_SIZE = 64")
    s = s.replace("std::exp", "expf")
    s = s.replace("std::log", "logf")
    s = s.replace("WaitCudaNotificationOnDevice", "WaitRocmNotificationOnDevice")
    s = s.replace("hipHostAlloc", "hipHostMalloc")
    s = s.replace(
        "#include <cub/device/device_radix_sort.cuh>",
        "#include <hipcub/hipcub.hpp>\n#include <hipcub/backend/rocprim/device/device_radix_sort.hpp>",
    )
    s = s.replace(
        '#include "cub/device/device_radix_sort.cuh"',
        "#include <hipcub/hipcub.hpp>\n#include <hipcub/backend/rocprim/device/device_radix_sort.hpp>",
    )
    s = s.replace(
        "#include <cub/device/device_segmented_radix_sort.cuh>",
        "#include <hipcub/backend/rocprim/device/device_segmented_radix_sort.hpp>",
    )
    s = s.replace(
        "#include <cub/device/device_reduce.cuh>", "#include <hipcub/backend/rocprim/device/device_reduce.hpp>"
    )
    s = s.replace(
        "#include <cub/device/device_run_length_encode.cuh>",
        "#include <hipcub/backend/rocprim/device/device_run_length_encode.hpp>",
    )
    s = s.replace("#include <cub/device/device_scan.cuh>", "#include <hipcub/backend/rocprim/device/device_scan.hpp>")
    s = s.replace(
        "#include <cub/iterator/counting_input_iterator.cuh>",
        "#include <hipcub/backend/rocprim/iterator/counting_input_iterator.hpp>",
    )
    s = s.replace(
        "#include <cub/iterator/discard_output_iterator.cuh>",
        "#include <hipcub/backend/rocprim/iterator/discard_output_iterator.hpp>",
    )
    s = s.replace("#include <cub/util_allocator.cuh>", "#include <hipcub/util_allocator.hpp>")
    s = s.replace('#include "cub/util_allocator.cuh"', "#include <hipcub/util_allocator.hpp>")
    s = s.replace("#include <cub/util_type.cuh>", "#include <hipcub/backend/rocprim/util_type.hpp>")
    s = s.replace('#include "cub/util_type.cuh"', "#include <hipcub/backend/rocprim/util_type.hpp>")
    s = s.replace("#include <cub/device/device_partition.cuh>", "#include <hipcub/device/device_partition.hpp>")
    s = s.replace("#include <math_constants.h>", "#include <limits>")
    s = s.replace("#include <library_types.h>", "")  # Doesn't exist
    s = s.replace("typedef half MappedType", "typedef __half MappedType")

    # CUBLAS -> HIPBLAS
    s = s.replace("CUBLAS", "HIPBLAS")
    s = s.replace("Cublas", "Hipblas")
    s = s.replace("cublas", "hipblas")
    # deprecated cublas symbol doesn't exist in hipblas, map to new symbol
    s = s.replace("HIPBLAS_GEMM_DEFAULT_TENSOR_OP", "HIPBLAS_GEMM_DEFAULT")

    # Undefined ROCMRT constants -> std::numeric_limits
    s = s.replace("ROCMRT_INF_F", "std::numeric_limits<float>::infinity()")

    # compatible layer
    s = s.replace("rocblas_gemm_strided_batched_ex", "_compat_rocblas_gemm_strided_batched_ex")
    s = s.replace("RocblasMathModeSetter", "CompatRocblasMathModeSetter")

    # CURAND -> HIPRAND
    s = s.replace("CURAND", "HIPRAND")
    s = s.replace("Curand", "Hiprand")
    s = s.replace("curand", "hiprand")

    # NCCL -> RCCL
    # s = s.replace('NCCL_CALL', 'RCCL_CALL')
    s = s.replace("#include <nccl.h>", "#include <rccl/rccl.h>")

    # CUDNN -> MIOpen
    s = s.replace("CUDNN", "MIOPEN")
    s = s.replace("Cudnn", "Miopen")
    s = s.replace("cudnn", "miopen")
    # hipify seems to have a bug for MIOpen, cudnn.h -> hipDNN.h, cudnn -> hipdnn
    s = s.replace("#include <hipDNN.h>", "#include <miopen/miopen.h>")
    s = s.replace("hipdnn", "miopen")
    s = s.replace("HIPDNN_STATUS_SUCCESS", "miopenStatusSuccess")
    s = s.replace("HIPDNN", "MIOPEN")
    s = s.replace("MIOPEN_BATCHNORM_SPATIAL", "miopenBNSpatial")
    s = s.replace("MIOPEN_BATCHNORM_PER_ACTIVATION", "miopenBNPerActivation")
    s = s.replace("MIOPEN_LRN_CROSS_CHANNEL", "miopenLRNCrossChannel")
    s = s.replace("MIOPEN_POOLING_MAX", "miopenPoolingMax")
    s = s.replace("MIOPEN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING", "miopenPoolingAverageInclusive")
    s = s.replace("MIOPEN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING", "miopenPoolingAverage")

    # CUSPARSE -> HIPSPARSE
    s = s.replace("CUSPARSE", "HIPSPARSE")

    # CUFFT -> HIPFFT
    s = s.replace("CUFFT", "HIPFFT")
    s = s.replace("cufftXtMakePlanMany", "hipfftXtMakePlanMany")
    s = s.replace("cufftXtExec", "hipfftXtExec")

    # Undo where above hipify steps went too far.
    s = s.replace("id, ROCM", "id, CUDA")  # cuda_execution_provider.cc
    s = s.replace("ROCM error executing", "HIP error executing")
    s = s.replace("ROCM_PINNED", "CUDA_PINNED")
    s = s.replace("rocm_err", "hip_err")
    s = s.replace("RegisterHipTrainingKernels", "RegisterRocmTrainingKernels")
    s = s.replace("ROCM_VERSION", "CUDA_VERSION")  # semantically different meanings, cannot hipify
    s = s.replace("__ROCM_ARCH__", "__CUDA_ARCH__")  # semantically different meanings, cannot hipify
    # "std::log" above incorrectly changed "std::logic_error" to "logfic_error"
    s = s.replace("logfic_error", "std::logic_error")

    # Deletions
    s = s.replace('#include "device_atomic_functions.h"', "")  # HIP atomics in main hip header already

    # Fix warnings due to incorrect header paths, intentionally after all other hipify steps.
    s = s.replace("#include <hiprand_kernel.h>", "#include <hiprand/hiprand_kernel.h>")
    s = s.replace("#include <rocblas.h>", "#include <rocblas/rocblas.h>")
    s = s.replace("#include <hipblas.h>", "#include <hipblas/hipblas.h>")
    s = s.replace("#include <hipfft.h>", "#include <hipfft/hipfft.h>")
    s = s.replace('#include "hipfft.h"', "#include <hipfft/hipfft.h>")
    s = s.replace('#include "hipfftXt.h"', "#include <hipfft/hipfftXt.h>")

    # Fix onnxruntime/contrib_ops/rocm/transformers. They include cpu headers which use "cuda" in their names.
    s = s.replace("rocm_device_prop_", "cuda_device_prop_")
    s = s.replace("rocm_device_arch_", "cuda_device_arch_")

    s = s.replace("HipTuningContext", "RocmTuningContext")

    # We want hipfft, which needs hipDataType etc, but only do this for files that have "fft" in their names
    # And we do this last, undoing or fixing hipify mistakes.
    if "fft" in src_file_path:
        s = s.replace("rocblas_datatype", "hipDataType")
        s = s.replace("hipDataType_f32_c", "HIP_C_32F")
        s = s.replace("hipDataType_f32_r", "HIP_R_32F")
        s = s.replace("hipDataType_f64_c", "HIP_C_64F")
        s = s.replace("hipDataType_f64_r", "HIP_R_64F")
        s = s.replace("hipDataType_f16_c", "HIP_C_16F")
        s = s.replace("hipDataType_f16_r", "HIP_R_16F")

    with open(dst_file_path, "w") as f:
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hipify_perl", required=True)
    parser.add_argument("--output", "-o", help="output file")
    parser.add_argument("src", help="src")
    args = parser.parse_args()

    hipify(args.hipify_perl, args.src, args.output)
