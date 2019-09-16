
set(mimalloc_root_dir ${PROJECT_SOURCE_DIR}/external/mimalloc)
set(mimalloc_output_dir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/)
set(mimalloc_wheel_dir ${mimalloc_output_dir}/onnxruntime/capi/)

add_definitions(-DUSE_MIMALLOC) # used in ONNXRuntime
include_directories(${mimalloc_root_dir}/include)

if(NOT IS_DIRECTORY ${mimalloc_wheel_dir})
  file(MAKE_DIRECTORY ${mimalloc_wheel_dir})
endif()

if (WIN32)
  # The generic MiMalloc CMakeLists.txt project lacks 
  # the needed hooks to correctly compile malloc for Windows
  # so we fall back to the specially provided VS solutions
  set(mimalloc_output mimalloc-static)

  if(NOT ${CMAKE_GENERATOR_PLATFORM} MATCHES "x64|Win32")
    message(FATAL_ERROR "MiMalloc doesn't support ARM/ARM64 targets")
  endif()
  
  set(vs_version "vs2019")
  if (${CMAKE_GENERATOR} MATCHES "Visual Studio [1-5]+ [0-9]+")
    set(vs_version "vs2017")
  endif()
  
  set(mimalloc_config "Release")
  if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(mimalloc_config "Debug")
  endif()

  set(mimalloc_target_winsdk ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION})
  if(DEFINED ENV{WindowsSDKVersion})
    set(mimalloc_target_winsdk $ENV{WindowsSDKVersion})
  endif()
  
  set(mimalloc_platform ${CMAKE_GENERATOR_PLATFORM})
  if(${CMAKE_GENERATOR_PLATFORM} MATCHES "Win32")
    set(mimalloc_platform x86)
  endif()

  # msbuild throws a fit during a postbuild step when copying files if the source uses backslashes and the destination uses forward slashes
  STRING(REGEX REPLACE "/" "\\\\" msbuild_converted_output_dir ${mimalloc_output_dir})
  add_custom_command(OUTPUT ${mimalloc_output} COMMAND msbuild ${mimalloc_root_dir}/ide/${vs_version}/mimalloc.sln
                    /p:OutDir=${msbuild_converted_output_dir} /p:Platform=${mimalloc_platform} /p:Configuration=${mimalloc_config}
                    /p:WindowsTargetPlatformVersion=${mimalloc_target_winsdk})
  add_custom_target(mimalloc_static ALL DEPENDS ${mimalloc_output})

  add_library(mimalloc IMPORTED STATIC)
  add_dependencies(mimalloc mimalloc_static)
  set_target_properties(mimalloc PROPERTIES IMPORTED_LOCATION "${mimalloc_output_dir}${mimalloc_output}.lib")

else()
  set(MI_BUILD_TESTS OFF CACHE BOOL "Build mimalloc tests" FORCE)
  add_subdirectory(${mimalloc_root_dir} EXCLUDE_FROM_ALL)
  set_target_properties(mimalloc PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  target_compile_definitions(mimalloc PUBLIC MI_USE_CXX=ON)

  # copy the dll into the directory where setup.py will look for it
  get_target_property(mimalloc_output_name mimalloc OUTPUT_NAME)
  install(TARGETS mimalloc DESTINATION ${mimalloc_wheel_dir})
endif()
