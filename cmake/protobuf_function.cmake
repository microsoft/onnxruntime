# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Changelog:
#copied from https://github.com/protocolbuffers/protobuf/blob/master/cmake/protobuf-config.cmake.in
#sed -i 's/protobuf_generate/onnxruntime_protobuf_generate/g' protobuf-config.cmake.orig
#replace 'protobuf::protoc' with ${PROTOC_EXECUTABLE} and ${PROTOC_DEPS}
#remove OUTDIR

function(onnxruntime_protobuf_generate)
  include(CMakeParseArguments)
  if(EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
    set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
  else()
    set(PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)
    set(PROTOC_DEPS protobuf::protoc)
  endif()
  set(_options APPEND_PATH NO_SRC_INCLUDES)
  set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO GEN_SRC_PREFIX GEN_SRC_SUB_DIR)
  if(COMMAND target_sources)
    list(APPEND _singleargs TARGET)
  endif()
  set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS)

  cmake_parse_arguments(onnxruntime_protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT onnxruntime_protobuf_generate_PROTOS AND NOT onnxruntime_protobuf_generate_TARGET)
    message(SEND_ERROR "Error: onnxruntime_protobuf_generate called without any targets or source files")
    return()
  endif()

  if(NOT onnxruntime_protobuf_generate_OUT_VAR AND NOT onnxruntime_protobuf_generate_TARGET)
    message(SEND_ERROR "Error: onnxruntime_protobuf_generate called without a target or output variable")
    return()
  endif()

  if(NOT onnxruntime_protobuf_generate_LANGUAGE)
    set(onnxruntime_protobuf_generate_LANGUAGE cpp)
  endif()
  string(TOLOWER ${onnxruntime_protobuf_generate_LANGUAGE} onnxruntime_protobuf_generate_LANGUAGE)

  if(onnxruntime_protobuf_generate_EXPORT_MACRO AND onnxruntime_protobuf_generate_LANGUAGE STREQUAL cpp)
    set(_dll_export_decl "dllexport_decl=${onnxruntime_protobuf_generate_EXPORT_MACRO}:")
  endif()

  if(NOT onnxruntime_protobuf_generate_EXTENSIONS)
    if(onnxruntime_protobuf_generate_LANGUAGE STREQUAL cpp)
      set(onnxruntime_protobuf_generate_EXTENSIONS .pb.h .pb.cc)
    elseif(onnxruntime_protobuf_generate_LANGUAGE STREQUAL python)
      set(onnxruntime_protobuf_generate_EXTENSIONS _pb2.py)
    else()
      message(SEND_ERROR "Error: onnxruntime_protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
      return()
    endif()
  endif()

  if(onnxruntime_protobuf_generate_TARGET)
    get_target_property(_source_list ${onnxruntime_protobuf_generate_TARGET} SOURCES)
    foreach(_file ${_source_list})
      if(_file MATCHES "proto$")
        list(APPEND onnxruntime_protobuf_generate_PROTOS ${_file})
      endif()
    endforeach()
  endif()

  if(NOT onnxruntime_protobuf_generate_PROTOS)
    message(SEND_ERROR "Error: onnxruntime_protobuf_generate could not find any .proto files")
    return()
  endif()

  if (NOT onnxruntime_protobuf_generate_NO_SRC_INCLUDES)
    if(onnxruntime_protobuf_generate_APPEND_PATH)
      # Create an include path for each file specified
      foreach(_file ${onnxruntime_protobuf_generate_PROTOS})
        get_filename_component(_abs_file ${_file} ABSOLUTE)
        get_filename_component(_abs_path ${_abs_file} PATH)
        list(FIND _protobuf_include_path ${_abs_path} _contains_already)
        if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${_abs_path})
        endif()
      endforeach()
    else()
      set(_protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
  endif()

  foreach(DIR ${onnxruntime_protobuf_generate_IMPORT_DIRS})
    get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
      list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()

  set(_generated_srcs_all)

  if (onnxruntime_protobuf_generate_GEN_SRC_PREFIX)
    set(_src_prefix "${onnxruntime_protobuf_generate_GEN_SRC_PREFIX}")
  else()
    set(_src_prefix "")
  endif()

  if (onnxruntime_protobuf_generate_GEN_SRC_SUB_DIR)
    set(_src_sub_dir "${onnxruntime_protobuf_generate_GEN_SRC_SUB_DIR}")
    if (NOT EXISTS ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR}/${_src_sub_dir})
      file(MAKE_DIRECTORY ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR}/${_src_sub_dir})
    endif()
  else()
    set(_src_sub_dir "")
  endif()

  foreach(_proto ${onnxruntime_protobuf_generate_PROTOS})
    get_filename_component(_abs_file ${_proto} ABSOLUTE)
    get_filename_component(_basename ${_proto} NAME_WE)

    set(_generated_srcs)
    foreach(_ext ${onnxruntime_protobuf_generate_EXTENSIONS})
      list(APPEND _generated_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_src_sub_dir}${_src_prefix}${_basename}${_ext}")
    endforeach()
    list(APPEND _generated_srcs_all ${_generated_srcs})

    if (onnxruntime_USE_FULL_PROTOBUF)
      add_custom_command(
        OUTPUT ${_generated_srcs}
        COMMAND  ${PROTOC_EXECUTABLE}
        ARGS --${onnxruntime_protobuf_generate_LANGUAGE}_out ${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR}/${_src_sub_dir} ${_protobuf_include_path} ${_abs_file}
        DEPENDS ${_abs_file} ${PROTOC_DEPS}
        COMMENT "Running ${onnxruntime_protobuf_generate_LANGUAGE} protocol buffer (full) compiler on ${_proto}"
        VERBATIM )
    else()
      add_custom_command(
        OUTPUT ${_generated_srcs}
        COMMAND  ${PROTOC_EXECUTABLE}
        ARGS --${onnxruntime_protobuf_generate_LANGUAGE}_out lite:${_dll_export_decl}${CMAKE_CURRENT_BINARY_DIR}/${_src_sub_dir} ${_protobuf_include_path} ${_abs_file}
        DEPENDS ${_abs_file} ${PROTOC_DEPS}
        COMMENT "Running ${onnxruntime_protobuf_generate_LANGUAGE} protocol buffer compiler (lite) on ${_proto}"
        VERBATIM )
    endif()
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(onnxruntime_protobuf_generate_OUT_VAR)
    set(${onnxruntime_protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
  if(onnxruntime_protobuf_generate_TARGET)
    target_sources(${onnxruntime_protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
  endif()

endfunction()
