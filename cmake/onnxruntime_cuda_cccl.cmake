# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# CCCL (libcu++/CUB/Thrust) header handling shared by the in-tree CUDA provider
# (onnxruntime_providers_cuda.cmake) and the CUDA plugin EP (onnxruntime_providers_cuda_plugin.cmake).
# Both compile host C++ translation units that include CUTLASS headers, which pull in <cuda/std/...>,
# so both need the same include path handling.

include_guard(GLOBAL)

# Work around a CUDA 13.3 cudafe++ (EDG front-end) regression that mis-parses CCCL's
# global-qualified partial specializations, e.g. in <cub/device/device_transform.cuh>:
#   template <typename T>
#   struct ::cuda::proclaims_copyable_arguments<...> : ::cuda::std::true_type {};
# nvcc fails with "global qualification of class name is invalid before ':' token".
# The fix is to write the specialization with the namespace reopened instead of using a
# global-qualified name. We cannot edit the (often read-only) toolkit headers, so generate
# corrected copies of the affected headers into the build tree and place that directory
# ahead of the toolkit cccl include path. This is a no-op on toolkits whose headers do not
# contain the offending pattern (e.g. once NVIDIA fixes it), so it is safe to keep enabled.
function(ort_cuda133_patch_cccl_header src dst)
  if (NOT EXISTS "${src}")
    return()
  endif()
  file(READ "${src}" _content)
  set(_orig "${_content}")
  # <cub/device/device_transform.cuh>
  string(REPLACE
    "template <typename T>\nstruct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};"
    "_CCCL_BEGIN_NAMESPACE_CUDA\ntemplate <typename T>\nstruct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>> : ::cuda::std::true_type\n{};\n_CCCL_END_NAMESPACE_CUDA"
    _content "${_content}")
  # <cub/device/dispatch/tuning/tuning_transform.cuh>
  string(REPLACE
    "template <>\nstruct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::transform::always_true_predicate>\n    : ::cuda::std::true_type\n{};"
    "_CCCL_BEGIN_NAMESPACE_CUDA\ntemplate <>\nstruct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::transform::always_true_predicate>\n    : ::cuda::std::true_type\n{};\n_CCCL_END_NAMESPACE_CUDA"
    _content "${_content}")
  if (NOT _content STREQUAL _orig)
    get_filename_component(_dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(WRITE "${dst}" "${_content}")
  elseif (EXISTS "${dst}")
    # The toolkit header no longer matches the offending pattern (e.g. after a CUDA
    # upgrade in an existing build tree). Remove any previously generated copy so a
    # stale patched header does not keep shadowing the toolkit header.
    file(REMOVE "${dst}")
  endif()
endfunction()

# Give ${target} everything it needs to compile against the CCCL headers of a CUDA 13 toolkit:
#
#  * Handle the CUDA 13.0 CCCL header directory move: libcu++, CUB and Thrust moved from
#    <toolkit>/include to <toolkit>/include/cccl, so <cuda/std/utility> - reached from the CUTLASS
#    headers that host .cc files include - is no longer on the default include path of the host
#    compiler. nvcc adds it by itself, so this only matters for targets that compile host C++.
#  * On CUDA 13.3, generate the patched CCCL headers described above into ${CMAKE_BINARY_DIR}
#    (and remove stale ones), then put that directory first on the include path.
#
# Note that this generates files in the build tree as a side effect, not just include flags.
# It must be called for every target that compiles CUDA or CUTLASS-including host sources;
# targets that inherit $<TARGET_PROPERTY:...,INCLUDE_DIRECTORIES> from such a target are
# covered by the parent call, but only because that generator expression is evaluated after
# configuration - see the call in onnxruntime_providers_cuda_plugin.cmake.
function(ort_configure_cuda_cccl target)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    return()
  endif()

  foreach(inc_dir ${CUDAToolkit_INCLUDE_DIRS})
    if (EXISTS "${inc_dir}/cccl")
      # The UNIX guard is not a statement about MSVC being unaffected: the cudafe++ regression
      # is simply untested on Windows, where no CUDA 13.3 build has been run. If a Windows
      # CUDA 13.3 build hits the same "global qualification of class name is invalid" error,
      # dropping UNIX from this condition is expected to be all that is needed.
      if (UNIX AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.3 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.4)
        # Generate cudafe++-parseable copies of the CCCL headers that contain global-qualified
        # partial specializations (see ort_cuda133_patch_cccl_header above) and put the fixed
        # directory ahead of the toolkit cccl include so the corrected headers win.
        set(_ort_cccl_fix_dir "${CMAKE_BINARY_DIR}/cccl_cuda13_fix")
        ort_cuda133_patch_cccl_header(
          "${inc_dir}/cccl/cub/device/device_transform.cuh"
          "${_ort_cccl_fix_dir}/cub/device/device_transform.cuh")
        ort_cuda133_patch_cccl_header(
          "${inc_dir}/cccl/cub/device/dispatch/tuning/tuning_transform.cuh"
          "${_ort_cccl_fix_dir}/cub/device/dispatch/tuning/tuning_transform.cuh")
        if (EXISTS "${_ort_cccl_fix_dir}/cub/device/device_transform.cuh" OR
            EXISTS "${_ort_cccl_fix_dir}/cub/device/dispatch/tuning/tuning_transform.cuh")
          target_include_directories(${target} BEFORE PRIVATE "${_ort_cccl_fix_dir}")
        endif()
      endif()

      # Add the cccl subdirectory to the include path so <cuda/std/utility> can be found
      target_include_directories(${target} PRIVATE "${inc_dir}/cccl")
    endif()
  endforeach()
endfunction()
