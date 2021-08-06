# Copyright(C) Xilinx Inc.
# Licensed under the MIT License

set(PYXIR_SHARED_LIB libpyxir.so)

if(PYTHONINTERP_FOUND)
  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "import pyxir as px; print(px.get_include_dir()); print(px.get_lib_dir());"
    RESULT_VARIABLE __result
    OUTPUT_VARIABLE __output
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(__result MATCHES 0)
    string(REGEX REPLACE ";" "\\\\;" __values ${__output})
    string(REGEX REPLACE "\r?\n" ";"    __values ${__values})
    list(GET __values 0 PYXIR_INCLUDE_DIR)
    list(GET __values 1 PYXIR_LIB_DIR)
  endif()
else()
  message(STATUS "To find Pyxir, Python interpretater is required to be found.")
endif()

add_library(pyxir SHARED IMPORTED)
message("-- Found Pyxir lib: ${PYXIR_LIB_DIR}/${PYXIR_SHARED_LIB}")
set_property(TARGET pyxir PROPERTY IMPORTED_LOCATION ${PYXIR_LIB_DIR}/${PYXIR_SHARED_LIB})

message("-- Found Pyxir include: ${PYXIR_INCLUDE_DIR}")
include_directories(${PYXIR_INCLUDE_DIR})
