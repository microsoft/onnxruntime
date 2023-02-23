add_definitions(-DUSE_MIMALLOC)

set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(MI_DEBUG_FULL OFF CACHE BOOL "" FORCE)
set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
onnxruntime_fetchcontent_makeavailable(mimalloc)