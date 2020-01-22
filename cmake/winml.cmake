# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(precompiled_header.cmake)
include(winml_sdk_helpers.cmake)
include(winml_cppwinrt.cmake)

# get the current nuget sdk kit directory
get_sdk(sdk_folder sdk_version)
set(target_folder ONNXRuntime/winml)
set(winml_api_dir ${REPO_ROOT}/winml/api)
set(winml_dll_dir ${REPO_ROOT}/winml/dll)
set(winml_lib_dir ${REPO_ROOT}/winml/lib)
set(winml_lib_api_dir ${REPO_ROOT}/winml/lib/api)
set(winml_adapter_dir ${REPO_ROOT}/winml/adapter)
set(winml_lib_api_image_dir ${REPO_ROOT}/winml/lib/api.image)
set(winml_lib_common_dir ${REPO_ROOT}/winml/lib/common)
set(winml_lib_telemetry_dir ${REPO_ROOT}/winml/lib/telemetry)

# Version parts for Windows.AI.MachineLearning.dll.
set(WINML_VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(WINML_VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(WINML_VERSION_BUILD_PART   0 CACHE STRING "Third part of numeric file/product version.")
set(WINML_VERSION_PRIVATE_PART 0 CACHE STRING "Fourth part of numeric file/product version.")
set(WINML_VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")

get_filename_component(exclusions "${winml_api_dir}/exclusions.txt" ABSOLUTE)
convert_forward_slashes_to_back(${exclusions} CPPWINRT_COMPONENT_EXCLUSION_LIST)

# For winrt idl files:
# 1) the file name must match the casing of the file on disk.
# 2) for winrt idls the casing must match the namespaces within exactly (Window.AI.MachineLearning).
# target_cppwinrt will attempt to create a winmd with the name and same casing as the supplied
# idl file. If the name of the winmd file does not match the contained namespaces, cppwinrt.exe
# will generate component template files with fully qualified names, which will not match the existing
# generated component files.
#
# For native idl files there are no casing restrictions.
get_filename_component(winrt_idl "${winml_api_dir}/Windows.AI.MachineLearning.idl" ABSOLUTE)
get_filename_component(idl_native "${winml_api_dir}/windows.ai.machineLearning.native.idl" ABSOLUTE)
get_filename_component(idl_native_internal "${winml_api_dir}/windows.ai.machineLearning.native.internal.idl" ABSOLUTE)

# generate cppwinrt sdk
add_generate_cppwinrt_sdk_headers_target(
  winml_sdk_cppwinrt                                      # the target name
  ${sdk_folder}                                           # location of sdk folder
  ${sdk_version}                                          # sdk version
  ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include  # output folder relative to CMAKE_BINARY_DIR where the generated sdk will be placed in the
  ${target_folder}                                        # folder where this target will be placed
)

# generate winml headers from idl
target_cppwinrt(winml_api
  ${winrt_idl}            # winml winrt idl to compile
  ${winml_lib_api_dir}    # location for cppwinrt generated component sources
  ${sdk_folder}           # location of sdk folder
  ${sdk_version}          # sdk version
  ${target_folder}        # the folder this target will be placed under
)

target_midl(winml_api_native
  ${idl_native}           # winml native idl to compile
  ${sdk_folder}           # location of sdk folder
  ${sdk_version}          # sdk version
  ${target_folder}        # the folder this target will be placed under
)

target_midl(winml_api_native_internal
  ${idl_native_internal}  # winml internal native idl to compile
  ${sdk_folder}           # location of sdk folder
  ${sdk_version}          # sdk version
  ${target_folder})       # the folder this target will be placed under

###########################
# Add winml_lib_telemetry
###########################
list(APPEND winml_libs winml_lib_telemetry)
# Add static library that will be archived/linked for both static/dynamic library
add_library(winml_lib_telemetry STATIC
  ${winml_lib_telemetry_dir}/inc/TelemetryEvent.h
  ${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfig.h
  ${winml_lib_common_dir}/inc/WinMLTelemetryHelper.h
  ${winml_lib_telemetry_dir}/Telemetry.cpp
  ${winml_lib_telemetry_dir}/TelemetryEvent.cpp
  ${winml_lib_telemetry_dir}/WinMLTelemetryHelper.cpp
  ${winml_lib_telemetry_dir}/pch.h
)

# Compiler options
if (onnxruntime_USE_TELEMETRY)
  set_target_properties(winml_lib_telemetry PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
endif()

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_telemetry pch.h)

# Includes
target_include_directories(winml_lib_telemetry PRIVATE ${CMAKE_SOURCE_DIR}/common/inc)
target_include_directories(winml_lib_telemetry PRIVATE ${winml_lib_telemetry_dir})
target_include_directories(winml_lib_telemetry PRIVATE ${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows)

# Properties
set_target_properties(winml_lib_telemetry
  PROPERTIES
  FOLDER
  ${target_folder})

###########################
# Add winml_adapter
###########################
list(APPEND winml_libs winml_adapter)
list(APPEND winml_adapter_files
    ${winml_adapter_dir}/CpuOrtSessionBuilder.cpp
    ${winml_adapter_dir}/CpuOrtSessionBuilder.h
    ${winml_adapter_dir}/CustomRegistryHelper.h
    ${winml_adapter_dir}/FeatureDescriptorFactory.cpp
    ${winml_adapter_dir}/FeatureDescriptorFactory.h
    ${winml_adapter_dir}/LotusEnvironment.cpp
    ${winml_adapter_dir}/LotusEnvironment.h
    ${winml_adapter_dir}/pch.h
    ${winml_adapter_dir}/WinMLAdapter.cpp
    ${winml_adapter_dir}/WinMLAdapter.h
    ${winml_adapter_dir}/ZeroCopyInputStreamWrapper.cpp
    ${winml_adapter_dir}/ZeroCopyInputStreamWrapper.h
    )

if (onnxruntime_USE_DML)
  list(APPEND winml_adapter_files
      ${winml_adapter_dir}/AbiCustomRegistryImpl.cpp
      ${winml_adapter_dir}/AbiCustomRegistryImpl.h
      ${winml_adapter_dir}/DmlOrtSessionBuilder.cpp
      ${winml_adapter_dir}/DmlOrtSessionBuilder.h
      )
endif(onnxruntime_USE_DML)

add_library(winml_adapter ${winml_adapter_files})

# Compiler definitions
onnxruntime_add_include_to_target(winml_adapter onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
target_include_directories(winml_adapter PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(winml_adapter ${onnxruntime_EXTERNAL_DEPENDENCIES})

# Specify the usage of a precompiled header
target_precompiled_header(winml_adapter pch.h)

# Includes
target_include_directories(winml_adapter PRIVATE ${winml_lib_api_dir})                                    # needed for generated headers
target_include_directories(winml_adapter PRIVATE ${winml_lib_dir})
target_include_directories(winml_adapter PRIVATE ${winml_adapter_dir})

set_target_properties(winml_adapter
  PROPERTIES
  FOLDER
  ${target_folder})

# Link libraries
if (onnxruntime_USE_DML)
  target_add_dml(winml_adapter)
endif(onnxruntime_USE_DML)

# add it to the onnxruntime shared library
set(onnxruntime_winml winml_adapter)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES winml_adapter)

###########################
# Add winml_lib_image
###########################
list(APPEND winml_libs winml_lib_image)

# Add static library that will be archived/linked for both static/dynamic library
add_library(winml_lib_image STATIC
  ${winml_lib_api_image_dir}/inc/ConverterResourceStore.h
  ${winml_lib_api_image_dir}/inc/D3DDeviceCache.h
  ${winml_lib_api_image_dir}/inc/DeviceHelpers.h
  ${winml_lib_api_image_dir}/inc/ImageConversionHelpers.h
  ${winml_lib_api_image_dir}/inc/ImageConversionTypes.h
  ${winml_lib_api_image_dir}/inc/ImageConverter.h
  ${winml_lib_api_image_dir}/inc/TensorToVideoFrameConverter.h
  ${winml_lib_api_image_dir}/inc/VideoFrameToTensorConverter.h
  ${winml_lib_api_image_dir}/CpuDetensorizer.h
  ${winml_lib_api_image_dir}/CpuTensorizer.h
  ${winml_lib_api_image_dir}/pch.h
  ${winml_lib_api_image_dir}/ConverterResourceStore.cpp
  ${winml_lib_api_image_dir}/D3DDeviceCache.cpp
  ${winml_lib_api_image_dir}/DeviceHelpers.cpp
  ${winml_lib_api_image_dir}/ImageConversionHelpers.cpp
  ${winml_lib_api_image_dir}/ImageConverter.cpp
  ${winml_lib_api_image_dir}/TensorToVideoFrameConverter.cpp
  ${winml_lib_api_image_dir}/VideoFrameToTensorConverter.cpp
)

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_image pch.h)

# Includes
target_include_directories(winml_lib_image PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/dml/DmlExecutionProvider/src/External/D3DX12)   # for d3dx12.h
target_include_directories(winml_lib_image PRIVATE ${winml_lib_api_dir})                                                              # needed for generated headers
target_include_directories(winml_lib_image PRIVATE ${winml_lib_api_image_dir})
target_include_directories(winml_lib_image PRIVATE ${ONNXRUNTIME_ROOT})
target_include_directories(winml_lib_image PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})                                                        # for status.h
target_include_directories(winml_lib_image PRIVATE ${REPO_ROOT}/cmake/external/gsl/include)
target_include_directories(winml_lib_image PRIVATE ${REPO_ROOT}/cmake/external/onnx)
target_include_directories(winml_lib_image PRIVATE ${REPO_ROOT}/cmake/external/protobuf/src)

# Properties
set_target_properties(winml_lib_image
  PROPERTIES
  FOLDER
  ${target_folder})

# Link libraries
target_link_libraries(winml_lib_image PRIVATE winml_lib_common)
if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_image)
endif(onnxruntime_USE_DML)


###########################
# Add winml_lib_api
###########################
list(APPEND winml_libs winml_lib_api)

# Add static library that will be archived/linked for both static/dynamic library
add_library(winml_lib_api STATIC
  ${winml_lib_api_dir}/impl/FeatureCompatibility.h
  ${winml_lib_api_dir}/impl/IMapFeatureValue.h
  ${winml_lib_api_dir}/impl/ISequenceFeatureValue.h
  ${winml_lib_api_dir}/impl/MapBase.h
  ${winml_lib_api_dir}/impl/SequenceBase.h
  ${winml_lib_api_dir}/impl/Tensor.h
  ${winml_lib_api_dir}/impl/TensorBase.h
  ${winml_lib_api_dir}/impl/TensorBuffer.h
  ${winml_lib_api_dir}/impl/TensorKindFrom.h
  ${winml_lib_api_dir}/impl/TensorMemoryBufferReference.h
  ${winml_lib_api_dir}/ImageFeatureDescriptor.cpp
  ${winml_lib_api_dir}/ImageFeatureDescriptor.h
  ${winml_lib_api_dir}/ImageFeatureValue.cpp
  ${winml_lib_api_dir}/ImageFeatureValue.h
  ${winml_lib_api_dir}/LearningModel.cpp
  ${winml_lib_api_dir}/LearningModel.h
  ${winml_lib_api_dir}/LearningModelBinding.cpp
  ${winml_lib_api_dir}/LearningModelBinding.h
  ${winml_lib_api_dir}/LearningModelDevice.cpp
  ${winml_lib_api_dir}/LearningModelDevice.h
  ${winml_lib_api_dir}/LearningModelEvaluationResult.cpp
  ${winml_lib_api_dir}/LearningModelEvaluationResult.h
  ${winml_lib_api_dir}/LearningModelSession.cpp
  ${winml_lib_api_dir}/LearningModelSession.h
  ${winml_lib_api_dir}/LearningModelSessionOptions.cpp
  ${winml_lib_api_dir}/LearningModelSessionOptions.h
  ${winml_lib_api_dir}/MapFeatureDescriptor.cpp
  ${winml_lib_api_dir}/MapFeatureDescriptor.h
  ${winml_lib_api_dir}/SequenceFeatureDescriptor.cpp
  ${winml_lib_api_dir}/SequenceFeatureDescriptor.h
  ${winml_lib_api_dir}/TensorFeatureDescriptor.cpp
  ${winml_lib_api_dir}/TensorFeatureDescriptor.h
  ${winml_lib_api_dir}/pch/pch.h
)

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_api pch.h)

# Includes
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_dir})
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_dir}/pch)
target_include_directories(winml_lib_api PRIVATE ${winml_adapter_dir})
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_image_dir}/inc)
target_include_directories(winml_lib_api PRIVATE ${winml_lib_telemetry_dir}/inc)

target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/date/include)
target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/gsl/include)
target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/onnx)

target_include_directories(winml_lib_api PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_include_directories(winml_lib_api PRIVATE ${ONNXRUNTIME_INCLUDE_DIR}/core/graph)
target_include_directories(winml_lib_api PRIVATE ${ONNXRUNTIME_ROOT})
target_include_directories(winml_lib_api PRIVATE ${ONNXRUNTIME_ROOT}/core/graph)
target_include_directories(winml_lib_api PRIVATE ${REPO_ROOT}/cmake/external/eigen)
target_include_directories(winml_lib_api PRIVATE ${REPO_ROOT}/cmake/external/onnx)
target_include_directories(winml_lib_api PRIVATE ${REPO_ROOT}/cmake/external/protobuf/src)
target_include_directories(winml_lib_api PRIVATE ${REPO_ROOT}/cmake/external/gsl/include)

# Properties
set_target_properties(winml_lib_api
  PROPERTIES
  FOLDER
  ${target_folder})

# Add deps
add_dependencies(winml_lib_api onnx)
add_dependencies(winml_lib_api winml_sdk_cppwinrt)
add_dependencies(winml_lib_api winml_api)
add_dependencies(winml_lib_api winml_api_native)
add_dependencies(winml_lib_api winml_api_native_internal)

# Link libraries
target_link_libraries(winml_lib_api PRIVATE winml_lib_telemetry)
if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_api)
endif(onnxruntime_USE_DML)

###########################
# Add winml_lib_common
###########################
list(APPEND winml_libs winml_lib_common)

add_library(winml_lib_common STATIC
  ${winml_lib_common_dir}/CommonDeviceHelpers.cpp
)

target_include_directories(winml_lib_common PRIVATE ${winml_lib_api_dir})
target_precompiled_header(winml_lib_common inc/pch.h)

###########################
# Add winml_dll
###########################
list(APPEND winml_libs winml_dll)

set_source_files_properties(
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated/module.g.excl.cpp
  PROPERTIES
  GENERATED
  TRUE)

# Add library
add_library(winml_dll SHARED
  ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated/module.g.excl.cpp
  ${winml_dll_dir}/windows.ai.machinelearning.def
  ${winml_dll_dir}/winml.rc
  ${winml_dll_dir}/pch.h
  ${winml_dll_dir}/module.cpp
)

# Compiler definitions
target_compile_definitions(winml_dll PRIVATE VER_MAJOR=${WINML_VERSION_MAJOR_PART})
target_compile_definitions(winml_dll PRIVATE VER_MINOR=${WINML_VERSION_MINOR_PART})
target_compile_definitions(winml_dll PRIVATE VER_BUILD=${WINML_VERSION_BUILD_PART})
target_compile_definitions(winml_dll PRIVATE VER_PRIVATE=${WINML_VERSION_PRIVATE_PART})
target_compile_definitions(winml_dll PRIVATE VER_STRING=\"${WINML_VERSION_STRING}\")

# Specify the usage of a precompiled header
target_precompiled_header(winml_dll pch.h)

# Includes
target_include_directories(winml_dll PRIVATE ${winml_dll_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_dir}/impl)
target_include_directories(winml_dll PRIVATE ${winml_adapter_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_image_dir}/inc)
target_include_directories(winml_dll PRIVATE ${winml_lib_telemetry_dir}/inc)

target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/date/include)
target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/gsl/include)
target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/external/onnx)

target_include_directories(winml_dll PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_include_directories(winml_dll PRIVATE ${ONNXRUNTIME_INCLUDE_DIR}/core/graph)
target_include_directories(winml_dll PRIVATE ${ONNXRUNTIME_ROOT})
target_include_directories(winml_dll PRIVATE ${ONNXRUNTIME_ROOT}/core/graph)
target_include_directories(winml_dll PRIVATE ${REPO_ROOT}/cmake/external/onnx)
target_include_directories(winml_dll PRIVATE ${REPO_ROOT}/cmake/external/protobuf/src)
target_include_directories(winml_dll PRIVATE ${REPO_ROOT}/cmake/external/gsl/include)
target_include_directories(winml_dll PRIVATE ${REPO_ROOT}/cmake/external/eigen)

# Properties
set_target_properties(winml_dll
  PROPERTIES
  OUTPUT_NAME windows.ai.machinelearning)
set_target_properties(winml_dll
  PROPERTIES
  FOLDER
  ${target_folder})

target_link_options(winml_dll PRIVATE
  "/DEF:${WINML_DIR}/windows.ai.machinelearning.def ${os_component_link_flags} /DELAYLOAD:d3d12.dll /DELAYLOAD:d3d11.dll /DELAYLOAD:dxgi.dll")
if (onnxruntime_USE_DML)
  target_link_options(winml_dll PRIVATE "/DELAYLOAD:directml.dll")
endif(onnxruntime_USE_DML)

# Any project that links in debug_alloc.obj needs this lib.
# unresolved external symbol __imp_SymSetOptions
# ...                        __imp_SymGetLineFromAddr64
# ...                        __imp_SymInitialize
# ...                        __imp_SymFromAddr
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  target_link_libraries(winml_dll PRIVATE dbghelp.lib)
endif()
target_link_libraries(winml_dll PRIVATE
  onnxruntime
  re2
  winml_lib_api
  winml_lib_image
  winml_lib_telemetry
  delayimp.lib)
# 1 of 3 projects that fail in link with 'failed to do memory mapped file I/O' (Only release)
# when using x86 hosted architecture. When using the LKG compiler this becomes a problem
# because it falls back to incorrectly using the public version of link.
# To avoid the scenario completely, this will tell cl/link to already start with x64 hosted,
# rather than waiting for it to fail and retry and resolve incorrectly.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set_target_properties(winml_dll PROPERTIES VS_GLOBAL_PreferredToolArchitecture "x64")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

###########################
# Flags common to all WinML libraries
###########################
# The default libraries to link with in Windows are kernel32.lib;user32.lib;gdi32.lib;winspool.lib;shell32.lib;ole32.lib;oleaut32.lib;uuid.lib;comdlg32.lib;advapi32.lib
# Remove these desktop specific libraries and use the OneCore umbrella library instead
foreach(default_lib kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdgl32.lib advapi32.lib)
  set(removed_libs "${removed_libs} /NODEFAULTLIB:${default_lib}")
endforeach()
set(CMAKE_C_STANDARD_LIBRARIES "${removed_libs} onecoreuap.lib")
set(CMAKE_CXX_STANDARD_LIBRARIES "${removed_libs} onecoreuap.lib")

set_target_properties(${winml_libs} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON)
foreach(winml_lib ${winml_libs})
  target_compile_options(${winml_lib} PRIVATE /GR- /await /wd4238 /bigobj)
  target_compile_definitions(${winml_lib} PRIVATE
    PLATFORM_WINDOWS
    _SCL_SECURE_NO_WARNINGS  # Remove warnings about unchecked iterators
    ONNX_NAMESPACE=onnx
    ONNX_ML
    LOTUS_LOG_THRESHOLD=2
    LOTUS_ENABLE_STDERR_LOGGING
  )
  target_link_libraries(${winml_lib} PRIVATE wil)
  target_include_directories(${winml_lib} BEFORE PRIVATE
    # Windows ML generated component headers
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/winml_api
    ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated
    # SDK cppwinrt headers
    ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include
  )
  target_include_directories(${winml_lib} PRIVATE
    ${winml_lib_common_dir}/inc
  )
  add_dependencies(${winml_lib}
    winml_sdk_cppwinrt
    winml_api
    winml_api_native
    winml_api_native_internal
  )
endforeach()

option(onnxruntime_BUILD_WINML_TESTS "Build WinML tests" ON)
if (onnxruntime_BUILD_WINML_TESTS)
  include(winml_unittests.cmake)
endif()
