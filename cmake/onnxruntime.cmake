# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(UNIX)
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.lds)
  set(OUTPUT_STYLE gcc)
else()
  set(SYMBOL_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_dll.def)
  set(OUTPUT_STYLE vc)
endif()


#If you want to verify if there is any extra line in symbols.txt, run
# nm -C -g --defined libonnxruntime.so |grep -v '\sA\s' | cut -f 3 -d ' ' | sort
# after build

list(APPEND SYMBOL_FILES "${REPO_ROOT}/tools/ci_build/gen_def.py")
foreach(f ${ONNXRUNTIME_PROVIDER_NAMES})
  list(APPEND SYMBOL_FILES "${ONNXRUNTIME_ROOT}/core/providers/${f}/symbols.txt")
endforeach()

add_custom_command(OUTPUT ${SYMBOL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c
  COMMAND ${PYTHON_EXECUTABLE} "${REPO_ROOT}/tools/ci_build/gen_def.py" --version_file "${ONNXRUNTIME_ROOT}/../VERSION_NUMBER" --src_root "${ONNXRUNTIME_ROOT}" --config ${ONNXRUNTIME_PROVIDER_NAMES} --style=${OUTPUT_STYLE} --output ${SYMBOL_FILE} --output_source ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c
  DEPENDS ${SYMBOL_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(onnxruntime_generate_def ALL DEPENDS ${SYMBOL_FILE} ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c)
add_library(onnxruntime SHARED ${CMAKE_CURRENT_BINARY_DIR}/generated_source.c)
set_target_properties(onnxruntime PROPERTIES VERSION ${ORT_VERSION})
add_dependencies(onnxruntime onnxruntime_generate_def ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT})
onnxruntime_add_include_to_target(onnxruntime)

if (onnxruntime_USE_CUDA)
  target_include_directories(onnxruntime PRIVATE ${onnxruntime_CUDNN_HOME}/include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

if(UNIX)
  if (APPLE)
    set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker -dead_strip")
  else()
    set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${SYMBOL_FILE} -Xlinker --no-undefined -Xlinker --gc-sections")
  endif()
else()
  set(ONNXRUNTIME_SO_LINK_FLAG "-DEF:${SYMBOL_FILE}")
endif()

if (NOT WIN32)
  if (APPLE)
    set_target_properties(onnxruntime PROPERTIES INSTALL_RPATH "@loader_path")
  else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
  endif()
endif()

#The BEGIN_WHOLE_ARCHIVE/END_WHOLE_ARCHIVE part should contain the implementations of all the C API functions
target_link_libraries(onnxruntime PRIVATE
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_NGRAPH}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_TENSORRT}
    ${PROVIDERS_OPENVINO}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    ${onnxruntime_EXTERNAL_LIBRARIES})

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG})
set_target_properties(onnxruntime PROPERTIES LINK_DEPENDS ${SYMBOL_FILE})
if(onnxruntime_ENABLE_LTO)
  set_target_properties(onnxruntime PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  set_target_properties(onnxruntime PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()
install(TARGETS onnxruntime
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "ONNXRuntime")
