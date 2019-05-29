# Minimum CMake required
cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 11)

set(THIRD_PARTY_ROOT ${HOROVOD_ROOT}/third_party)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -B /data/anaconda/envs/ort/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=strict-aliasing -Wno-error=unused-parameter -Wno-error=ignored-qualifiers -Wno-error=maybe-uninitialized")

add_definitions(-DEIGEN_MPL2_ONLY=1 -DHAVE_CUDA=1 -DHAVE_NCCL=1 -DHOROVOD_GPU_ALLREDUCE='N' -DHOROVOD_GPU_ALLGATHER='M' -DHOROVOD_GPU_BROADCAST='M')

file(GLOB_RECURSE horovod_common_src
    "${HOROVOD_ROOT}/horovod/common/fusion_buffer_manager.cc"
    "${HOROVOD_ROOT}/horovod/common/half.cc" 
    "${HOROVOD_ROOT}/horovod/common/message.cc" 
    "${HOROVOD_ROOT}/horovod/common/common.cc" 
    "${HOROVOD_ROOT}/horovod/common/mpi_context.cc" 
    "${HOROVOD_ROOT}/horovod/common/operations.cc" 
    "${HOROVOD_ROOT}/horovod/common/parameter_manager.cc" 
    "${HOROVOD_ROOT}/horovod/common/response_cache.cc" 
    "${HOROVOD_ROOT}/horovod/common/timeline.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/mpi_cuda_operations.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/cuda_operations.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/collective_operations.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/mpi_operations.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/operation_manager.cc" 
    "${HOROVOD_ROOT}/horovod/common/ops/nccl_operations.cc"
    "${HOROVOD_ROOT}/horovod/common/optim/bayesian_optimization.cc" 
    "${HOROVOD_ROOT}/horovod/common/optim/gaussian_process.cc" 
    "${HOROVOD_ROOT}/horovod/common/logging.cc"
    )

add_library(horovod ${horovod_common_src})

target_compile_options(horovod PRIVATE ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(horovod PRIVATE ${THIRD_PARTY_ROOT}/eigen ${THIRD_PARTY_ROOT}/lbfgs/include ${THIRD_PARTY_ROOT}/boost/assert/include ${THIRD_PARTY_ROOT}/boost/config/include ${THIRD_PARTY_ROOT}/boost/core/include ${THIRD_PARTY_ROOT}/boost/detail/include ${THIRD_PARTY_ROOT}/boost/iterator/include ${THIRD_PARTY_ROOT}/boost/lockfree/include ${THIRD_PARTY_ROOT}/boost/mpl/include ${THIRD_PARTY_ROOT}/boost/parameter/include ${THIRD_PARTY_ROOT}/boost/predef/include ${THIRD_PARTY_ROOT}/boost/preprocessor/include ${THIRD_PARTY_ROOT}/boost/static_assert/include ${THIRD_PARTY_ROOT}/boost/type_traits/include ${THIRD_PARTY_ROOT}/boost/utility/include ${THIRD_PARTY_ROOT}/flatbuffers/include /usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent /usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include /usr/lib/openmpi/include /usr/lib/openmpi/include/openmpi /usr/local/cuda/include ${MPI_CXX_INCLUDE_PATH})
