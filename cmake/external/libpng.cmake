set(PNG_SHARED      OFF CACHE INTERNAL "")
set(PNG_TESTS       OFF CACHE INTERNAL "")
set(PNG_EXECUTABLES OFF CACHE INTERNAL "")
set(PNG_BUILD_ZLIB  ON  CACHE INTERNAL "")

FetchContent_Declare(
  libpng
  GIT_REPOSITORY https://github.com/glennrp/libpng.git
  GIT_TAG        v1.6.39
  GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(libpng)

if (MSVC)
  # assume we're building zlib and need to manually specify the dependency.
  add_dependencies(png_static ${ZLIB_LIBRARIES})
endif()

cmake_policy(SET CMP0079 NEW)
target_link_libraries(png_static ${ZLIB_LIBRARIES})
target_include_directories(png_static PRIVATE ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

set(libpng_INCLUDE_DIRS ${libpng_SOURCE_DIR} ${libpng_BINARY_DIR})
set(libpng_LIB_NAME png_static)
