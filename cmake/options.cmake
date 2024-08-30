
# Options
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_ENABLE_PYTHON "Enable python buildings" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
# Enable ONNX Runtime CUDA EP's internal unit tests that directly access the EP's internal functions instead of through
# OpKernels. When the option is ON, we will have two copies of GTest library in the same process. It is not a typical
# use. If you hit any problem with that, please do not report it to GTest. Turn OFF the following build option instead.
cmake_dependent_option(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS "Build with CUDA unit tests" OFF "onnxruntime_USE_CUDA;onnxruntime_BUILD_UNIT_TESTS" OFF)

option(onnxruntime_USE_CUDA_NHWC_OPS "Build CUDA with NHWC op support" OFF)
option(onnxruntime_CUDA_MINIMAL "Build CUDA without any operations apart from memcpy ops. Usefuel for a very minial TRT build" OFF)
option(onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO "When building with CUDA support, generate device code line number information." OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_COREML "Build with CoreML support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_QNN "Build with QNN support" OFF)
option(onnxruntime_USE_SNPE "Build with SNPE support" OFF)
option(onnxruntime_USE_RKNPU "Build with RKNPU support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NEURAL_SPEED "Build with Neural Speed support" OFF)
option(onnxruntime_USE_JSEP "Build with JavaScript implemented kernels support" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_BUILD_CSHARP "Build C# library" OFF)
option(onnxruntime_BUILD_OBJC "Build Objective-C library" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_LLVM "Build TVM with LLVM" OFF)
option(onnxruntime_USE_VSINPU "Build with VSINPU support" OFF)

cmake_dependent_option(onnxruntime_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "onnxruntime_USE_CUDA" OFF)
option(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON)

option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_AVX "Use AVX instructions" OFF)
option(onnxruntime_USE_AVX2 "Use AVX2 instructions" OFF)
option(onnxruntime_USE_AVX512 "Use AVX512 instructions" OFF)

option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_BUILD_APPLE_FRAMEWORK "Build a macOS/iOS framework" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)

option(onnxruntime_USE_VITISAI "Build with Vitis-AI" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_USE_TENSORRT_BUILTIN_PARSER "Use TensorRT builtin parser" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_GCOV_COVERAGE "Compile with options necessary to run code coverage" OFF)
option(onnxruntime_DONT_VECTORIZE "Do not vectorize operations in Eigen" OFF)

option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
cmake_dependent_option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB "Build dump debug information about node inputs and outputs with support for sql database." OFF "onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS" OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_MIGRAPHX "Build with AMDMIGraphX support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_USE_ACL_1902 "Build with ACL version 1902 support" OFF)
option(onnxruntime_USE_ACL_1905 "Build with ACL version 1905 support" OFF)
option(onnxruntime_USE_ACL_1908 "Build with ACL version 1908 support" OFF)
option(onnxruntime_USE_ACL_2002 "Build with ACL version 2002 support" OFF)
option(onnxruntime_USE_ACL_2308 "Build with ACL version 2308 support" OFF)
option(onnxruntime_USE_ARMNN "Build with ArmNN support" OFF)
option(onnxruntime_ARMNN_RELU_USE_CPU "Use the CPU implementation for the Relu operator for the ArmNN EP" ON)
option(onnxruntime_ARMNN_BN_USE_CPU "Use the CPU implementation for the Batch Normalization operator for the ArmNN EP" ON)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
cmake_dependent_option(onnxruntime_USE_MIMALLOC "Override new/delete and arena allocator with mimalloc" OFF "WIN32;NOT onnxruntime_USE_CUDA;NOT onnxruntime_USE_OPENVINO" OFF)
option(onnxruntime_USE_CANN "Build with CANN support" OFF)
option(onnxruntime_USE_ROCM "Build with AMD GPU support" OFF)
option(onnxruntime_USE_TVM "Build with TVM support" OFF)
option(onnxruntime_TVM_CUDA_RUNTIME "Build TVM with CUDA support" OFF)
option(onnxruntime_TVM_USE_LLVM "Build TVM with LLVM. Set customized path to llvm-config.exe here if need" OFF)
option(onnxruntime_TVM_USE_HASH "Build ipp-crypto library for support hash algorithm. It is defined for TVM only")
option(onnxruntime_USE_XNNPACK "Build with XNNPACK support. Provides an alternative math library on ARM, WebAssembly and x86." OFF)
option(onnxruntime_USE_WEBNN "Build with WebNN support. Enable hardware acceleration in web browsers." OFF)

# Options related to reducing the binary size produced by the build
# XNNPACK EP requires the internal NHWC contrib ops to be available, so this option must be OFF when onnxruntime_USE_XNNPACK is ON
cmake_dependent_option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF "NOT onnxruntime_USE_XNNPACK" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
option(onnxruntime_DISABLE_SPARSE_TENSORS "Disable sparse tensors data types" OFF)
option(onnxruntime_DISABLE_OPTIONAL_TYPE "Disable optional type" OFF)
option(onnxruntime_DISABLE_FLOAT8_TYPES "Disable float 8 types" OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
cmake_dependent_option(onnxruntime_DISABLE_RTTI "Disable RTTI" ON "NOT onnxruntime_ENABLE_PYTHON;NOT onnxruntime_USE_CUDA" OFF)
# For now onnxruntime_DISABLE_EXCEPTIONS will only work with onnxruntime_MINIMAL_BUILD, more changes (ONNX, non-CPU EP, ...) are required to run this standalone
cmake_dependent_option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." ON "onnxruntime_MINIMAL_BUILD;NOT onnxruntime_ENABLE_PYTHON" OFF)
# Even when onnxruntime_DISABLE_ABSEIL is ON, ONNX Runtime still needs to link to abseil.
option(onnxruntime_DISABLE_ABSEIL "Do not use Abseil data structures in ONNX Runtime source code. Redefine Inlined containers to STD containers." OFF)

option(onnxruntime_EXTENDED_MINIMAL_BUILD "onnxruntime_MINIMAL_BUILD with support for execution providers that compile kernels." OFF)
option(onnxruntime_MINIMAL_BUILD_CUSTOM_OPS "Add custom operator kernels support to a minimal build." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
option(onnxruntime_DISABLE_EXTERNAL_INITIALIZERS "Don't allow models to load external data" OFF)

#A special option just for debugging and sanitize check. Please do not enable in option in retail builds.
#The option has no effect on Windows.
option(onnxruntime_USE_VALGRIND "Build with valgrind hacks" OFF)

# A special build option only used for gathering code coverage info
option(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE "Run model tests even in debug mode" OFF)

# options for security fuzzing
# build configuration for fuzz testing is in onnxruntime_fuzz_test.cmake
option(onnxruntime_FUZZ_TEST "Enable Fuzz testing" OFF)

# training options
option(onnxruntime_ENABLE_NVTX_PROFILE "Enable NVTX profile." OFF)
option(onnxruntime_ENABLE_MEMORY_PROFILE "Enable memory profile." OFF)
option(onnxruntime_ENABLE_TRAINING "Enable full training functionality. Includes ORTModule and ORT Training APIs" OFF)
option(onnxruntime_ENABLE_TRAINING_APIS "Enable ort training apis." OFF)
option(onnxruntime_ENABLE_TRAINING_OPS "Include training operators but no training session support." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_ENABLE_CPU_FP16_OPS "Build with advanced instruction sets" ON)
option(onnxruntime_USE_NCCL "Build with NCCL support" OFF)
option(onnxruntime_USE_MPI "Build with MPI support" OFF)

# WebAssembly options
option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "Enable this option to create WebAssembly static library" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "Enable this option to create WebAssembly byte codes with multi-threads support" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "Enable this option to turn on exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING "Enable this option to turn on api exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING "Enable this option to turn on exception throwing even if the build disabled exceptions support" OFF)
option(onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER "Enable this option to run tests in browser instead of Node.js" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO "Enable this option to turn on DWARF format debug info" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_PROFILING "Enable this option to turn on WebAssembly profiling and preserve function names" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL "Enable this option to allow WebAssembly to output optimized model" OFF)

# Enable bitcode for iOS
option(onnxruntime_ENABLE_BITCODE "Enable bitcode for iOS only" OFF)

# build Pytorch's LazyTensor support
cmake_dependent_option(onnxruntime_ENABLE_LAZY_TENSOR "Enable ORT as a LazyTensor backend in Pytorch." ON "onnxruntime_ENABLE_TRAINING" OFF)

# build separate library of schemas of (custom) ops used by ORT (for ONNX to MLIR translation)
option(onnxruntime_BUILD_OPSCHEMA_LIB "Build op schema library" ON)

# option to enable custom operators in onnxruntime-extensions
option(onnxruntime_USE_EXTENSIONS "Build with onnxruntime-extensions for more features" OFF)

# option to enable custom operators in onnxruntime-extensions with a custom path
option(onnxruntime_EXTENSIONS_OVERRIDDEN, "Enable onnxruntime-extensions overridden" OFF)

# Enable registering custom op schemas from shared libraries in python environment.
option(onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS "Enable registering user defined custom op schemas dynamically" OFF)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE "" CACHE STRING "Specify custom protoc executable to build ONNX")

# pre-build python path
option(onnxruntime_PREBUILT_PYTORCH_PATH "Path to pytorch installation dir")

# external transformer src path
option(onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH "Path to external transformer src dir")

option(onnxruntime_ENABLE_CUDA_PROFILING "Enable CUDA kernel profiling" OFF)
option(onnxruntime_ENABLE_ROCM_PROFILING "Enable ROCM kernel profiling" OFF)

option(onnxruntime_ENABLE_CPUINFO "Enable cpuinfo" ON)

# ATen fallback support
option(onnxruntime_ENABLE_ATEN "Enable ATen fallback" OFF)

# Triton support
option(onnxruntime_ENABLE_TRITON "Enable Triton" OFF)

# composable kernel is managed automatically, unless user want to explicitly disable it, it should not be manually set
option(onnxruntime_USE_COMPOSABLE_KERNEL "Enable composable kernel for ROCm EP" ON)
cmake_dependent_option(onnxruntime_USE_COMPOSABLE_KERNEL_CK_TILE "Enable ck_tile for composable kernel" ON "onnxruntime_USE_COMPOSABLE_KERNEL" OFF)
option(onnxruntime_USE_ROCBLAS_EXTENSION_API "Enable rocblas tuning for ROCm EP" OFF)
option(onnxruntime_USE_TRITON_KERNEL "Enable triton compiled kernel" OFF)
option(onnxruntime_BUILD_KERNEL_EXPLORER "Build Kernel Explorer for testing and profiling GPU kernels" OFF)

option(onnxruntime_BUILD_CACHE "onnxruntime build with cache" OFF)
# https://zeux.io/2010/11/22/z7-everything-old-is-new-again/
cmake_dependent_option(MSVC_Z7_OVERRIDE "replacing /Zi and /ZI with /Z7 when using MSVC with CCache" ON "onnxruntime_BUILD_CACHE; MSVC" OFF)

option(onnxruntime_USE_AZURE "Build with azure inferencing support" OFF)
option(onnxruntime_USE_LOCK_FREE_QUEUE "Build with lock-free task queue for threadpool." OFF)
