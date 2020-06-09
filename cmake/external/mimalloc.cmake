set(mimalloc_root_dir ${PROJECT_SOURCE_DIR}/external/mimalloc)

if(onnxruntime_USE_MIMALLOC_STL_ALLOCATOR)
  add_definitions(-DUSE_MIMALLOC_STL_ALLOCATOR) # used in ONNXRuntime
endif()
if(onnxruntime_USE_MIMALLOC_ARENA_ALLOCATOR)
  add_definitions(-DUSE_MIMALLOC_ARENA_ALLOCATOR) # used in ONNXRuntime
endif()
include_directories(${mimalloc_root_dir}/include)

option(MI_OVERRIDE "" OFF)
option(MI_BUILD_TESTS "" OFF)

add_subdirectory(${mimalloc_root_dir} EXCLUDE_FROM_ALL)
set_target_properties(mimalloc-static PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if (WIN32)
  set_target_properties(mimalloc-static PROPERTIES COMPILE_FLAGS "/wd4389 /wd4201 /wd4244 /wd4565")
endif()
