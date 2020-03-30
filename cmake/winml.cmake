# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (NOT WINDOWS_STORE)
  message(FATAL_ERROR "WinML is only supported on WCOS")
endif()

include(precompiled_header.cmake)
include(winml_sdk_helpers.cmake)
include(winml_cppwinrt.cmake)

# get the current nuget sdk kit directory
get_sdk(sdk_folder sdk_version)
get_sdk_include_folder(${sdk_folder} ${sdk_version} sdk_include_folder)
set(dxcore_header "${sdk_include_folder}/um/dxcore.h")
set(target_folder ONNXRuntime/winml)
set(winml_adapter_dir ${REPO_ROOT}/winml/adapter)
set(winml_api_root ${REPO_ROOT}/winml/api)
set(winml_dll_dir ${REPO_ROOT}/winml/dll)
set(winml_lib_dir ${REPO_ROOT}/winml/lib)
set(winml_lib_api_dir ${REPO_ROOT}/winml/lib/api)
set(winml_lib_api_image_dir ${REPO_ROOT}/winml/lib/api.image)
set(winml_lib_api_ort_dir ${REPO_ROOT}/winml/lib/api.ort)
set(winml_lib_common_dir ${REPO_ROOT}/winml/lib/common)
set(winml_lib_telemetry_dir ${REPO_ROOT}/winml/lib/telemetry)

get_filename_component(exclusions "${winml_api_root}/exclusions.txt" ABSOLUTE)
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
get_filename_component(winrt_idl "${winml_api_root}/Windows.AI.MachineLearning.idl" ABSOLUTE)
get_filename_component(idl_native "${winml_api_root}/windows.ai.machineLearning.native.idl" ABSOLUTE)
get_filename_component(idl_native_internal "${winml_api_root}/windows.ai.machineLearning.native.internal.idl" ABSOLUTE)

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
target_compile_features(winml_lib_telemetry PRIVATE cxx_std_17)
target_compile_options(winml_lib_telemetry PRIVATE /GR- /await /wd4238)
if (onnxruntime_USE_TELEMETRY)
  set_target_properties(winml_lib_telemetry PROPERTIES COMPILE_FLAGS "/FI${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows/TraceLoggingConfigPrivate.h")
endif()

# Compiler flags
target_compile_definitions(winml_lib_telemetry PRIVATE PLATFORM_WINDOWS)
target_compile_definitions(winml_lib_telemetry PRIVATE _SCL_SECURE_NO_WARNINGS)      # remove warnings about unchecked iterators

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_telemetry pch.h)

# Includes
target_include_directories(winml_lib_telemetry PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)
target_include_directories(winml_lib_telemetry PRIVATE ${CMAKE_SOURCE_DIR}/common/inc)
target_include_directories(winml_lib_telemetry PRIVATE ${winml_lib_telemetry_dir})
target_include_directories(winml_lib_telemetry PRIVATE ${winml_lib_common_dir}/inc)
target_include_directories(winml_lib_telemetry PRIVATE ${ONNXRUNTIME_INCLUDE_DIR}/core/platform/windows)

# Properties
set_target_properties(winml_lib_telemetry
  PROPERTIES
  FOLDER
  ${target_folder})

# Link libraries
target_link_libraries(winml_lib_telemetry PRIVATE wil)

###########################
# Add winml_lib_ort
###########################

list(APPEND winml_lib_api_ort_files
  ${winml_lib_api_ort_dir}/inc/OnnxruntimeProvider.h
  ${winml_lib_api_ort_dir}/OnnxruntimeCpuSessionBuilder.h
  ${winml_lib_api_ort_dir}/OnnxruntimeCpuSessionBuilder.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeDescriptorConverter.h
  ${winml_lib_api_ort_dir}/OnnxruntimeDescriptorConverter.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeEngine.h
  ${winml_lib_api_ort_dir}/OnnxruntimeEngine.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeEngineBuilder.h
  ${winml_lib_api_ort_dir}/OnnxruntimeEngineBuilder.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeEnvironment.h
  ${winml_lib_api_ort_dir}/OnnxruntimeEnvironment.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeModel.h
  ${winml_lib_api_ort_dir}/OnnxruntimeModel.cpp
  ${winml_lib_api_ort_dir}/OnnxruntimeSessionBuilder.h
  ${winml_lib_api_ort_dir}/pch.h
    )

if (onnxruntime_USE_DML)
  list(APPEND winml_lib_api_ort_files
    ${winml_lib_api_ort_dir}/OnnxruntimeDmlSessionBuilder.h
    ${winml_lib_api_ort_dir}/OnnxruntimeDmlSessionBuilder.cpp
    )
endif(onnxruntime_USE_DML)

# Add static library that will be archived/linked for both static/dynamic library
add_library(winml_lib_ort STATIC ${winml_lib_api_ort_files})

# Compiler options
target_compile_features(winml_lib_ort PRIVATE cxx_std_17)
target_compile_options(winml_lib_ort PRIVATE /GR- /await /wd4238)

# Compiler definitions
target_compile_definitions(winml_lib_ort PRIVATE PLATFORM_WINDOWS)
target_compile_definitions(winml_lib_ort PRIVATE _SCL_SECURE_NO_WARNINGS)                         # remove warnings about unchecked iterators

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_ort pch.h)

# Includes
target_include_directories(winml_lib_ort PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                   # windows machine learning generated component headers
target_include_directories(winml_lib_ort PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)    # windows machine learning generated component headers
target_include_directories(winml_lib_ort PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers

target_include_directories(winml_lib_ort PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(winml_lib_ort PRIVATE ${REPO_ROOT}/winml)
target_include_directories(winml_lib_ort PRIVATE ${winml_lib_api_dir})                            # needed for generated headers
target_include_directories(winml_lib_ort PRIVATE ${winml_lib_api_core_dir})
target_include_directories(winml_lib_ort PRIVATE ${winml_lib_api_ort_dir})
target_include_directories(winml_lib_ort PRIVATE ${winml_lib_common_dir}/inc)
target_include_directories(winml_lib_ort PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_include_directories(winml_lib_ort PRIVATE ${ONNXRUNTIME_ROOT})

set_target_properties(winml_lib_ort
  PROPERTIES
  FOLDER
  ${target_folder})

# Add deps
add_dependencies(winml_lib_ort winml_sdk_cppwinrt)
add_dependencies(winml_lib_ort winml_api)
add_dependencies(winml_lib_ort winml_api_native)
add_dependencies(winml_lib_ort winml_api_native_internal)

# Link libraries
if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_ort)
endif()
target_link_libraries(winml_lib_ort PRIVATE wil)


###########################
# Add winml_adapter
###########################

list(APPEND winml_adapter_files
    ${winml_adapter_dir}/pch.h
    ${winml_adapter_dir}/winml_adapter_apis.h
    ${winml_adapter_dir}/winml_adapter_c_api.h
    ${winml_adapter_dir}/winml_adapter_c_api.cpp
    ${winml_adapter_dir}/winml_adapter_dml.cpp
    ${winml_adapter_dir}/winml_adapter_environment.cpp
    ${winml_adapter_dir}/winml_adapter_execution_provider.cpp
    ${winml_adapter_dir}/winml_adapter_model.cpp
    ${winml_adapter_dir}/winml_adapter_model.h
    ${winml_adapter_dir}/winml_adapter_session.cpp
    )

if (onnxruntime_USE_DML)
  list(APPEND winml_adapter_files
    ${winml_adapter_dir}/abi_custom_registry_impl.cpp
    ${winml_adapter_dir}/abi_custom_registry_impl.h
    )
endif(onnxruntime_USE_DML)

add_library(winml_adapter ${winml_adapter_files})

# wil requires C++17
set_target_properties(winml_adapter PROPERTIES CXX_STANDARD 17)
set_target_properties(winml_adapter PROPERTIES CXX_STANDARD_REQUIRED ON)

# Compiler definitions
onnxruntime_add_include_to_target(winml_adapter onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
target_include_directories(winml_adapter PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
add_dependencies(winml_adapter ${onnxruntime_EXTERNAL_DEPENDENCIES})

# Specify the usage of a precompiled header
target_precompiled_header(winml_adapter pch.h)

# Includes
target_include_directories(winml_adapter PRIVATE ${winml_adapter_dir})
target_include_directories(winml_adapter PRIVATE ${winml_lib_common_dir}/inc)

set_target_properties(winml_adapter
  PROPERTIES
  FOLDER
  ${target_folder})

# Link libraries
target_link_libraries(winml_adapter PRIVATE wil)
if (onnxruntime_USE_DML)
  target_add_dml(winml_adapter)
endif(onnxruntime_USE_DML)

# add it to the onnxruntime shared library
set(onnxruntime_winml winml_adapter)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES winml_adapter)

###########################
# Add winml_lib_image
###########################

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

# Compiler options
target_compile_features(winml_lib_image PRIVATE cxx_std_17)
target_compile_options(winml_lib_image PRIVATE /GR- /await /wd4238)

# Compiler flags
target_compile_definitions(winml_lib_image PRIVATE ONNX_NAMESPACE=onnx)
target_compile_definitions(winml_lib_image PRIVATE ONNX_ML)
target_compile_definitions(winml_lib_image PRIVATE LOTUS_LOG_THRESHOLD=2)
target_compile_definitions(winml_lib_image PRIVATE LOTUS_ENABLE_STDERR_LOGGING)
target_compile_definitions(winml_lib_image PRIVATE PLATFORM_WINDOWS)
target_compile_definitions(winml_lib_image PRIVATE _SCL_SECURE_NO_WARNINGS)                                 # remove warnings about unchecked iterators

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_image pch.h)

# Includes
target_include_directories(winml_lib_image PRIVATE ${CMAKE_CURRENT_BINARY_DIR})                                                               # windows machine learning generated component headers
target_include_directories(winml_lib_image PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                                                     # windows machine learning generated component headers
target_include_directories(winml_lib_image PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)                                      # windows machine learning generated component headers
target_include_directories(winml_lib_image PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)                                    # sdk cppwinrt headers
target_include_directories(winml_lib_image PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/dml/DmlExecutionProvider/src/External/D3DX12)   # for d3dx12.h
target_include_directories(winml_lib_image PRIVATE ${winml_lib_api_dir})                                                              # needed for generated headers
target_include_directories(winml_lib_image PRIVATE ${winml_lib_api_image_dir})
target_include_directories(winml_lib_image PRIVATE ${winml_lib_common_dir}/inc)
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

# Add deps
add_dependencies(winml_lib_image winml_sdk_cppwinrt)
add_dependencies(winml_lib_image winml_api)
add_dependencies(winml_lib_image winml_api_native)
add_dependencies(winml_lib_image winml_api_native_internal)

# Link libraries
target_link_libraries(winml_lib_image PRIVATE wil winml_lib_common)
if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_image)
endif(onnxruntime_USE_DML)


###########################
# Add winml_lib_api
###########################

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

# Compiler options
target_compile_features(winml_lib_api PRIVATE cxx_std_17)
target_compile_options(winml_lib_api PRIVATE /GR- /await /bigobj /wd4238)

# Compiler flags
target_compile_definitions(winml_lib_api PRIVATE ONNX_NAMESPACE=onnx)
target_compile_definitions(winml_lib_api PRIVATE ONNX_ML)
target_compile_definitions(winml_lib_api PRIVATE LOTUS_LOG_THRESHOLD=2)
target_compile_definitions(winml_lib_api PRIVATE LOTUS_ENABLE_STDERR_LOGGING)
target_compile_definitions(winml_lib_api PRIVATE PLATFORM_WINDOWS)
target_compile_definitions(winml_lib_api PRIVATE _SCL_SECURE_NO_WARNINGS)                         # remove warnings about unchecked iterators

# Specify the usage of a precompiled header
target_precompiled_header(winml_lib_api pch.h)

# Includes
target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                   # windows machine learning generated component headers
target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)    # windows machine learning generated component headers
target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers

target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_dir})
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_dir}/pch)
target_include_directories(winml_lib_api PRIVATE ${winml_adapter_dir})
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_image_dir}/inc)
target_include_directories(winml_lib_api PRIVATE ${winml_lib_api_ort_dir}/inc)
target_include_directories(winml_lib_api PRIVATE ${winml_lib_telemetry_dir}/inc)
target_include_directories(winml_lib_api PRIVATE ${winml_lib_common_dir}/inc)

target_include_directories(winml_lib_api PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
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
target_link_libraries(winml_lib_api PRIVATE wil winml_lib_telemetry)
if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_api)
endif(onnxruntime_USE_DML)

###########################
# Add winml_lib_common
###########################

add_library(winml_lib_common STATIC
  ${winml_lib_common_dir}/inc/common.h
  ${winml_lib_common_dir}/inc/CommonDeviceHelpers.h
  ${winml_lib_common_dir}/inc/cppwinrt_onnx.h
  ${winml_lib_common_dir}/inc/dx.h
  ${winml_lib_common_dir}/inc/errors.h
  ${winml_lib_common_dir}/inc/iengine.h
  ${winml_lib_common_dir}/inc/NamespaceAliases.h
  ${winml_lib_common_dir}/inc/onnx.h
  ${winml_lib_common_dir}/inc/PheonixSingleton.h
  ${winml_lib_common_dir}/inc/StringHelpers.h
  ${winml_lib_common_dir}/inc/WinMLTelemetryHelper.h
  ${winml_lib_common_dir}/inc/WinML_Lock.h
  ${winml_lib_common_dir}/inc/winrt_headers.h
  ${winml_lib_common_dir}/CommonDeviceHelpers.cpp
)

set_target_properties(winml_lib_common PROPERTIES CXX_STANDARD 17)
set_target_properties(winml_lib_common PROPERTIES CXX_STANDARD_REQUIRED ON)
target_compile_options(winml_lib_common PRIVATE /GR- /await /bigobj /wd4238)
target_link_libraries(winml_lib_common PRIVATE wil)
target_include_directories(winml_lib_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)
target_compile_definitions(winml_lib_common PRIVATE
  ONNX_NAMESPACE=onnx
  ONNX_ML
  LOTUS_LOG_THRESHOLD=2
  LOTUS_ENABLE_STDERR_LOGGING
  PLATFORM_WINDOWS
  _SCL_SECURE_NO_WARNINGS)
add_dependencies(winml_lib_common winml_sdk_cppwinrt)
add_dependencies(winml_lib_common winml_api)
add_dependencies(winml_lib_common winml_api_native)
add_dependencies(winml_lib_common winml_api_native_internal)

target_include_directories(winml_lib_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                   # windows machine learning generated component headers
target_include_directories(winml_lib_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)    # windows machine learning generated component headers
target_include_directories(winml_lib_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers
target_include_directories(winml_lib_common PRIVATE ${winml_lib_api_dir})
target_include_directories(winml_lib_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(winml_lib_common PRIVATE ${winml_lib_common_dir}/inc)
target_precompiled_header(winml_lib_common inc/pch.h)

if (onnxruntime_USE_DML)
  target_add_dml(winml_lib_common)
endif()

###########################
# Add winml_dll
###########################

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

# Compiler options
target_compile_features(winml_dll PRIVATE cxx_std_17)
target_compile_options(winml_dll PRIVATE /GR- /await /bigobj /wd4238)

# Compiler definitions
target_compile_definitions(winml_dll PRIVATE ONNX_NAMESPACE=onnx)
target_compile_definitions(winml_dll PRIVATE ONNX_ML)
target_compile_definitions(winml_dll PRIVATE LOTUS_LOG_THRESHOLD=2)
target_compile_definitions(winml_dll PRIVATE LOTUS_ENABLE_STDERR_LOGGING)
target_compile_definitions(winml_dll PRIVATE PLATFORM_WINDOWS)
target_compile_definitions(winml_dll PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
target_compile_definitions(winml_dll PRIVATE VER_MINOR=${VERSION_MINOR_PART})
target_compile_definitions(winml_dll PRIVATE VER_BUILD=${VERSION_BUILD_PART})
target_compile_definitions(winml_dll PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
target_compile_definitions(winml_dll PRIVATE VER_STRING=\"${VERSION_STRING}\")

# Specify the usage of a precompiled header
target_precompiled_header(winml_dll pch.h)

# Includes
target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api)                   # windows machine learning generated component headers
target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml_api/comp_generated)    # windows machine learning generated component headers
target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers

target_include_directories(winml_dll PRIVATE ${winml_dll_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_dir}/impl)
target_include_directories(winml_dll PRIVATE ${winml_lib_api_ort_dir}/inc)
target_include_directories(winml_dll PRIVATE ${winml_adapter_dir})
target_include_directories(winml_dll PRIVATE ${winml_lib_api_image_dir}/inc)
target_include_directories(winml_dll PRIVATE ${winml_lib_telemetry_dir}/inc)
target_include_directories(winml_dll PRIVATE ${winml_lib_common_dir}/inc)

target_include_directories(winml_dll PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
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

if (onnxruntime_USE_DML)
  set(delayload_dml "/DELAYLOAD:directml.dll")
endif(onnxruntime_USE_DML)

target_link_options(winml_dll PRIVATE /DEF:${WINML_DIR}/windows.ai.machinelearning.def ${os_component_link_flags} /DELAYLOAD:api-ms-win-core-libraryloader-l1-2-1.dll /DELAYLOAD:api-ms-win-core-threadpool-legacy-l1-1-0.dll /DELAYLOAD:api-ms-win-core-processtopology-obsolete-l1-1-0.dll /DELAYLOAD:api-ms-win-core-kernel32-legacy-l1-1-0.dll /DELAYLOAD:d3d12.dll /DELAYLOAD:d3d11.dll /DELAYLOAD:dxgi.dll ${delayload_dml})

if (EXISTS ${dxcore_header})
  target_link_options(winml_dll PRIVATE /DELAYLOAD:ext-ms-win-dxcore-l1-*.dll)
endif()

set_target_properties(winml_dll
  PROPERTIES
  FOLDER
  ${target_folder})

# Add deps
add_dependencies(winml_dll winml_sdk_cppwinrt)
add_dependencies(winml_dll winml_api_native)
add_dependencies(winml_dll winml_api_native_internal)

# Link libraries
target_link_libraries(winml_dll PRIVATE onnxruntime)
target_link_libraries(winml_dll PRIVATE re2)
target_link_libraries(winml_dll PRIVATE wil)
target_link_libraries(winml_dll PRIVATE winml_lib_api)
target_link_libraries(winml_dll PRIVATE winml_lib_image)
target_link_libraries(winml_dll PRIVATE winml_lib_ort)
target_link_libraries(winml_dll PRIVATE winml_lib_telemetry)
target_link_libraries(winml_dll PRIVATE delayimp.lib)

# Any project that links in debug_alloc.obj needs this lib.
# unresolved external symbol __imp_SymSetOptions
# ...                        __imp_SymGetLineFromAddr64
# ...                        __imp_SymInitialize
# ...                        __imp_SymFromAddr
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
  target_link_libraries(winml_dll PRIVATE dbghelp.lib)
endif()

# 1 of 3 projects that fail in link with 'failed to do memory mapped file I/O' (Only release)
# when using x86 hosted architecture. When using the LKG compiler this becomes a problem
# because it falls back to incorrectly using the public version of link.
# To avoid the scenario completely, this will tell cl/link to already start with x64 hosted,
# rather than waiting for it to fail and retry and resolve incorrectly.
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  set_target_properties(winml_dll PROPERTIES VS_GLOBAL_PreferredToolArchitecture "x64")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "Release")

option(onnxruntime_BUILD_WINML_TESTS "Build WinML tests" ON)
if (onnxruntime_BUILD_WINML_TESTS)
  include(winml_unittests.cmake)
endif()

# This is needed to suppress warnings that complain that no imports are found for the delayloaded library cublas64*.lib
# When cuda is enabled in the pipeline, it sets CMAKE_SHARED_LINKER_FLAGS which affects all targets including winml_dll.
# However, there are no cuda imports in winml_dll, and the linker throws the 4199 warning.
# This is needed to allow winml_dll build with cuda enabled.
target_link_options(winml_dll PRIVATE /ignore:4199)
