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

add_custom_command(OUTPUT ${SYMBOL_FILE} 
  COMMAND ${PYTHON_EXECUTABLE} "${REPO_ROOT}/tools/ci_build/gen_def.py" --version_file "${ONNXRUNTIME_ROOT}/../VERSION_NUMBER" --src_root "${ONNXRUNTIME_ROOT}" --config ${ONNXRUNTIME_PROVIDER_NAMES} --style=${OUTPUT_STYLE} --output ${SYMBOL_FILE}
  DEPENDS ${SYMBOL_FILES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(onnxruntime_generate_def ALL DEPENDS ${SYMBOL_FILE})
add_library(onnxruntime SHARED ${onnxruntime_session_srcs})
set_target_properties(onnxruntime PROPERTIES VERSION ${VERSION_NUMBER})
add_dependencies(onnxruntime onnxruntime_generate_def ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime PRIVATE ${ONNXRUNTIME_ROOT})
onnxruntime_add_include_to_target(onnxruntime gsl)

if(UNIX)
  if (APPLE)
    set(BEGIN_WHOLE_ARCHIVE -Xlinker -all_load)
    set(END_WHOLE_ARCHIVE -Xlinker -noall_load)
  else()
    set(BEGIN_WHOLE_ARCHIVE -Xlinker --whole-archive)
    set(END_WHOLE_ARCHIVE -Xlinker --no-whole-archive)
    set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${SYMBOL_FILE} -Xlinker --no-undefined")
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
    ${BEGIN_WHOLE_ARCHIVE}
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MKLDNN}
    ${PROVIDERS_TENSORRT}
    onnxruntime_optimizer
    onnxruntime_providers    
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    ${END_WHOLE_ARCHIVE}
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    debug ${onnxruntime_EXTERNAL_LIBRARIES_DEBUG} optimized ${onnxruntime_EXTERNAL_LIBRARIES})

set_property(TARGET onnxruntime APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG})
set_target_properties(onnxruntime PROPERTIES LINK_DEPENDS ${SYMBOL_FILE})

install(TARGETS onnxruntime
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

set_target_properties(onnxruntime PROPERTIES FOLDER "ONNXRuntime")
