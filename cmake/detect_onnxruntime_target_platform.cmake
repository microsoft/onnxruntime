# This file will set the onnxruntime_target_platform variable, if applicable.
# onnxruntime_target_platform identifies the platform to compile for.
block(PROPAGATE onnxruntime_target_platform)

unset(onnxruntime_target_platform)

if (MSVC)
  if (CMAKE_VS_PLATFORM_NAME)
    # Multi-platform generator
    set(onnxruntime_target_platform ${CMAKE_VS_PLATFORM_NAME})
  else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()

  if (onnxruntime_target_platform STREQUAL "ARM64" OR
      onnxruntime_target_platform STREQUAL "ARM64EC")
    # Do nothing. We'll just use the current value of onnxruntime_target_platform.
  elseif (onnxruntime_target_platform STREQUAL "ARM" OR
          CMAKE_GENERATOR MATCHES "ARM")
    set(onnxruntime_target_platform "ARM")
  elseif (onnxruntime_target_platform STREQUAL "x64" OR
          onnxruntime_target_platform STREQUAL "x86_64" OR
          onnxruntime_target_platform STREQUAL "AMD64" OR
          CMAKE_GENERATOR MATCHES "Win64")
    set(onnxruntime_target_platform "x64")
  elseif (onnxruntime_target_platform STREQUAL "Win32" OR
          onnxruntime_target_platform STREQUAL "x86" OR
          onnxruntime_target_platform STREQUAL "i386" OR
          onnxruntime_target_platform STREQUAL "i686")
    set(onnxruntime_target_platform "x86")
  else()
    message(FATAL_ERROR "Unknown target platform: ${onnxruntime_target_platform}")
  endif()
elseif(APPLE)
  if(DEFINED CMAKE_OSX_ARCHITECTURES)
    # We'll only set onnxruntime_target_platform when CMAKE_OSX_ARCHITECTURES specifies a single architecture.
    list(LENGTH CMAKE_OSX_ARCHITECTURES CMAKE_OSX_ARCHITECTURES_LEN)
    if(CMAKE_OSX_ARCHITECTURES_LEN EQUAL 1)
      set(onnxruntime_target_platform ${CMAKE_OSX_ARCHITECTURES})
    endif()
  else()
    set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
  endif()
else()
  #XXX: Sometimes the value of CMAKE_SYSTEM_PROCESSOR is set but it's wrong. For example, if you run an armv7 docker
  #image on an aarch64 machine with an aarch64 Ubuntu host OS, in the docker instance cmake may still report
  # CMAKE_SYSTEM_PROCESSOR as aarch64 by default. Given compiling this code may need more than 2GB memory, we do not
  # support compiling for ARM32 natively(only support cross-compiling), we will ignore this issue for now.
  if(NOT CMAKE_SYSTEM_PROCESSOR)
    message(WARNING "CMAKE_SYSTEM_PROCESSOR is not set. Please set it in your toolchain cmake file.")
    # Try to detect it
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang")
      execute_process(
          COMMAND "${CMAKE_C_COMPILER}" -dumpmachine
          OUTPUT_VARIABLE GCC_DUMP_MACHINE_OUT
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_VARIABLE _err
          RESULT_VARIABLE _res
      )
      if(NOT _res EQUAL 0)
        message(SEND_ERROR "Failed to run 'gcc -dumpmachine':\n ${_res}")
      endif()
      string(REPLACE "-" ";" GCC_DUMP_MACHINE_OUT_LIST "${GCC_DUMP_MACHINE_OUT}")
      list(LENGTH GCC_DUMP_MACHINE_OUT_LIST GCC_TRIPLET_LEN)
      if(GCC_TRIPLET_LEN EQUAL 4)
        list(GET GCC_DUMP_MACHINE_OUT_LIST 0 CMAKE_SYSTEM_PROCESSOR)
        message("Setting CMAKE_SYSTEM_PROCESSOR to ${CMAKE_SYSTEM_PROCESSOR}")
      endif()
    endif()
  endif()
  set(onnxruntime_target_platform ${CMAKE_SYSTEM_PROCESSOR})
endif()

if(DEFINED onnxruntime_target_platform)
  message(STATUS "onnxruntime_target_platform = ${onnxruntime_target_platform}")
else()
  message(WARNING "onnxruntime_target_platform is not set")
endif()

endblock()
