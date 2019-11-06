# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(pybind11)
FIND_PACKAGE(NumPy)

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)


# ---[ Python + Numpy
set(onnxruntime_pybind_srcs_pattern
    "${ONNXRUNTIME_ROOT}/python/*.cc"
    "${ONNXRUNTIME_ROOT}/python/*.h"
)

file(GLOB onnxruntime_pybind_srcs CONFIGURE_DEPENDS
  ${onnxruntime_pybind_srcs_pattern}
  )

#TODO(): enable cuda and test it
add_library(onnxruntime_pybind11_state MODULE ${onnxruntime_pybind_srcs})
if(HAS_CAST_FUNCTION_TYPE)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "-Wno-cast-function-type")
endif()

if(onnxruntime_PYBIND_EXPORT_OPSCHEMA)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE onnxruntime_PYBIND_EXPORT_OPSCHEMA)
endif()

if (onnxruntime_USE_MKLDNN)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE USE_MKLDNN=1)
endif()

target_include_directories(onnxruntime_pybind11_state PRIVATE ${ONNXRUNTIME_ROOT} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR})
target_include_directories(onnxruntime_pybind11_state PRIVATE ${pybind11_INCLUDE_DIRS})
if(APPLE)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/python/exported_symbols.lst")
elseif(UNIX)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/python/version_script.lds -Xlinker --gc-sections")
else()
  set(ONNXRUNTIME_SO_LINK_FLAG "-DEF:${ONNXRUNTIME_ROOT}/python/pybind.def")
endif()

set(onnxruntime_pybind11_state_libs
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_TENSORRT}
    ${PROVIDERS_NGRAPH}
    ${PROVIDERS_OPENVINO}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_DML}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnxruntime_pybind11_state_libs onnxruntime_language_interop onnxruntime_pyop)
endif()

set(onnxruntime_pybind11_state_dependencies
    ${onnxruntime_EXTERNAL_DEPENDENCIES}
    pybind11
)

add_dependencies(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_dependencies})

if (MSVC)
  # if MSVC, pybind11 looks for release version of python lib (pybind11/detail/common.h undefs _DEBUG)
  target_link_libraries(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_libs}
          ${PYTHON_LIBRARY_RELEASE} ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_EXTERNAL_LIBRARIES})
elseif (APPLE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  target_link_libraries(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_libs} ${onnxruntime_EXTERNAL_LIBRARIES}
          ${ONNXRUNTIME_SO_LINK_FLAG})
  set_target_properties(onnxruntime_pybind11_state PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  target_link_libraries(onnxruntime_pybind11_state PRIVATE ${onnxruntime_pybind11_state_libs} ${PYTHON_LIBRARY}
          ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_EXTERNAL_LIBRARIES})
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "-Xlinker -rpath=\$ORIGIN")
endif()

set_target_properties(onnxruntime_pybind11_state PROPERTIES PREFIX "")
set_target_properties(onnxruntime_pybind11_state PROPERTIES FOLDER "ONNXRuntime")
if(onnxruntime_ENABLE_LTO)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
endif()
if (MSVC)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".so")
endif()

file(GLOB onnxruntime_backend_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/backend/*.py"
)
file(GLOB onnxruntime_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/*.py"
)
file(GLOB onnxruntime_python_test_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/test/python/*.py"
)
file(GLOB onnxruntime_python_tools_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/*.py"
)
file(GLOB onnxruntime_python_datasets_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.py"
)
file(GLOB onnxruntime_python_datasets_data CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.pb"
    "${ONNXRUNTIME_ROOT}/python/datasets/*.onnx"
)

# adjust based on what target/s onnxruntime_unittests.cmake created
if (SingleUnitTestProject)
  set(test_data_target onnxruntime_test_all)
else()
  set(test_data_target onnxruntime_test_ir)
endif()

add_custom_command(
  TARGET onnxruntime_pybind11_state POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/backend
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/datasets
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/tools
  COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_ROOT}/__init__.py
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/ThirdPartyNotices.txt
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/docs/Privacy.md
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/LICENSE
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_test_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_backend_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/backend/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:onnxruntime_pybind11_state>
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_data}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_tools_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/tools/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/VERSION_NUMBER
      $<TARGET_FILE_DIR:${test_data_target}>
)

if (onnxruntime_USE_MKLDNN)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${MKLDNN_DLL_PATH}
        $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_NGRAPH)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${ngraph_LIBRARIES}/${NGRAPH_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_CODEGEN_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_CPU_BACKEND_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_IOMP5MD_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_MKLDNN_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_MKLML_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_TBB_SHARED_LIB}
		${ngraph_LIBRARIES}/${NGRAPH_TBB_SHARED_LIB_2}
        $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_OPENVINO)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${OPENVINO_CPU_EXTENSION_DIR}/${OPENVINO_CPU_EXTENSION_LIB}
        $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_TVM)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:tvm> $<TARGET_FILE:nnvm_compiler>
        $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_MKLML)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${MKLML_LIB_DIR}/${MKLML_SHARED_LIB} ${MKLML_LIB_DIR}/${IOMP5MD_SHARED_LIB}
        $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_NUPHAR)
  file(GLOB onnxruntime_python_nuphar_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/scripts/*.*"
  )
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/nuphar
    COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_nuphar_python_srcs}
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/nuphar/
  )
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  include(onnxruntime_language_interop_ops.cmake)  
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:onnxruntime_pywrapper>
      $<TARGET_FILE_DIR:${test_data_target}>/onnxruntime/capi/
  )
endif()
