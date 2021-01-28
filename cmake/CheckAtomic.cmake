# Sometimes linking against libatomic is required for atomic ops, if
# the platform doesn't support lock-free atomics.
# Simplified version from https://github.com/llvm-mirror/llvm/blob/master/cmake/modules/CheckAtomic.cmake
# as we always require 64-bit support

function(check_working_cxx_atomics varname)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "-std=c++11 ${CMAKE_REQUIRED_FLAGS}")
  CHECK_CXX_SOURCE_COMPILES("
    #include <atomic>
    int main() {
      return std::atomic<int64_t>{};
    }
" ${varname})
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endfunction(check_working_cxx_atomics)

# This isn't necessary on MSVC
if (NOT MSVC)
  # First check if atomics work without the library.
  check_working_cxx_atomics(HAVE_CXX_ATOMICS_WITHOUT_LIB)
  # If not, check if the library exists, and atomics work with it.
  if(NOT HAVE_CXX_ATOMICS_WITHOUT_LIB)
    list(APPEND CMAKE_REQUIRED_LIBRARIES "atomic")
    check_working_cxx_atomics(HAVE_CXX_ATOMICS_WITH_LIB)
    if (NOT HAVE_CXX_ATOMICS_WITH_LIB)
      message(FATAL_ERROR "Host compiler must support std::atomic for 64-bit values!")
    endif()
  endif()
endif()

message("HAVE_CXX_ATOMICS_WITHOUT_LIB=${HAVE_CXX_ATOMICS_WITHOUT_LIB}")
message("HAVE_CXX_ATOMICS_WITH_LIB=${HAVE_CXX_ATOMICS_WITH_LIB}")
