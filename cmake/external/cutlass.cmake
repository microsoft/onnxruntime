include(FetchContent)
FetchContent_Declare(
  cutlass
  URL ${DEP_URL_cutlass}
  URL_HASH SHA1=${DEP_SHA1_cutlass}
  SOURCE_DIR ${BUILD_DIR_NO_CONFIG}/_deps/cutlass-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/deps/cutlass-build
  DOWNLOAD_DIR ${BUILD_DIR_NO_CONFIG}/_deps/cutlass-download
)

FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
  FetchContent_Populate(cutlass)
endif()
