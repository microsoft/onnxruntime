# Copyright(C) Xilinx Inc.
# Licensed under the MIT License

include(ExternalProject)

#set(TBB_TEST OFF CACHE INTERNAL "")
#set(TBBMALLOC_BUILD OFF CACHE INTERNAL "")
#set(TBBMALLOC_PROXY_BUILD OFF CACHE INTERNAL "")
#set(TBB_TEST OFF CACHE INTERNAL "")
#set(TBB_OUTPUT_DIR_BASE intel-tbb)

ExternalProject_Add(intel_tbb
                    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
                    GIT_TAG v2021.8.0
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/tbb-src
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/tbb-build
                    CMAKE_ARGS -DTBB_TEST=OFF -DTBBMALLOC_BUILD=OFF -DTBBMALLOC_PROXY_BUILD=OFF -DTBB_TEST=OFF -DTBB_OUTPUT_DIR_BASE=tbb
                    INSTALL_COMMAND "")

#ExternalProject_Get_Property(tbb SOURCE_DIR)
#set(TBB_INC ${SOURCE_DIR}/src)
set(TBB_INC ${CMAKE_CURRENT_BINARY_DIR}/_deps/tbb-src/include)
include_directories(${TBB_INC})
link_directories(${CMAKE_CURRENT_BINARY_DIR}/_deps/tbb-build/tbb_${CMAKE_BUILD_TYPE})

if(WIN23)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
set (TBB "tbb12_debug")
else()
set (TBB "tbb12")
endif()
else()
set (TBB "tbb")
endif()
