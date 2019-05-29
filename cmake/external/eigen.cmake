include (ExternalProject)

if (onnxruntime_USE_PREINSTALLED_EIGEN)
    add_library(eigen INTERFACE)
    file(TO_CMAKE_PATH ${eigen_SOURCE_PATH} eigen_INCLUDE_DIRS)
    target_include_directories(eigen INTERFACE ${eigen_INCLUDE_DIRS})
else ()
    set(eigen_INCLUDE_DIRS  "${PROJECT_SOURCE_DIR}/external/eigen")
endif()
