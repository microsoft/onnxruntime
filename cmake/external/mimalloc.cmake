
set(mimalloc_root_dir ${PROJECT_SOURCE_DIR}/external/mimalloc)

if (onnxruntime_USE_MIMALLOC)
    if(onnxruntime_USE_CUDA OR onnxruntime_USE_OPENVINO) 
        message(WARNING "Ignoring directive to substitute mimalloc in for the default allocator on unimplemented targets")
    elseif (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU")
        # Some of the non-windows targets see strange runtime failures
        message(WARNING "Ignoring request to link to mimalloc - only windows supported")
    else()
        add_definitions(-DUSE_MIMALLOC) # used in ONNXRuntime
    endif()
endif()
include_directories(${mimalloc_root_dir}/include)

option(MI_OVERRIDE "" OFF)
option(MI_BUILD_TESTS "" OFF)

add_subdirectory(${mimalloc_root_dir} EXCLUDE_FROM_ALL)
set_target_properties(mimalloc-static PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
  set_target_properties(mimalloc-static PROPERTIES COMPILE_FLAGS "/wd4389 /wd4201 /wd4244 /wd4565")
endif()
