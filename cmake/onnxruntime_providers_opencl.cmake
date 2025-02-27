if(onnxruntime_USE_OPENCL)
  set(PROVIDERS_OPENCL onnxruntime_providers_opencl)
endif()
if (onnxruntime_USE_OPENCL)
  add_definitions(-DUSE_OPENCL=1)

  # Implement simple compiler driver to help
  #
  # a.h--------┐
  # b.h--------┤
  # kernel.cl -┴-> kernel.cl.inc -┐
  #                op.cc ---------┴-> op.o
  #
  # embed.py is responsible for transforming *.cl and related header files to
  # *.cl.inc, This enables us easily embedding the kernel src into an object
  # file. so that op and its kernel implementation src can be easily pruned
  find_package(Python3 COMPONENTS Interpreter REQUIRED)

  set(opencl_cl_path_prefix "${ONNXRUNTIME_ROOT}/core/providers/opencl/")
  file(GLOB_RECURSE opencl_cl_srcs CONFIGURE_DEPENDS
    "${opencl_cl_path_prefix}*.cl"
  )

  set(embed_tool ${PROJECT_SOURCE_DIR}/../onnxruntime/core/providers/opencl/embed.py)
  set(opencl_target_dir ${CMAKE_CURRENT_BINARY_DIR}/opencl_generated)
  set(opencl_generated_cl_includes)
  foreach(f ${opencl_cl_srcs})
    string(REPLACE ${opencl_cl_path_prefix} "" suffix ${f})
    get_filename_component(dir_of_f ${f} DIRECTORY ABSOLUTE)
    set(output "${opencl_target_dir}/${suffix}.inc")
    string(REGEX REPLACE "[/.]" "_" scan_tgt_name ${suffix})
    add_custom_target(${scan_tgt_name}
      # The Scanning Step:
      #
      # ${output}.deps-nonexisting will not be created, it purely work as
      # a trigger for updating ${output}.deps on every build. embed.py generates
      # a full list of included files by expanding all #include and calculate
      # checksum for each header file. If the new content match the existing
      # ${output}.deps, this file will not be updated.
      BYPRODUCTS ${output}.deps ${output}.deps-nonexisting
      COMMAND ${Python3_EXECUTABLE} ${embed_tool} -x cl
              -I "${ONNXRUNTIME_ROOT}/core/providers/opencl"
              -I "${dir_of_f}"
              -M ${f} -o ${output}.deps
      DEPENDS ${embed_tool} ${f}
      COMMENT "Scanning ${f} for transitive dependencies"
    )
    add_custom_command(
      # The Generating Step:
      #
      # This will be triggered by ${output}.deps, aka, any transitive
      # dependencies content change will cause the ${output}.deps being updated
      # thus, new ${output} is created and compiling and linking will be
      # triggered.
      OUTPUT ${output}
      COMMAND Python3::Interpreter ${embed_tool} -x cl
      -I "${ONNXRUNTIME_ROOT}/core/providers/opencl"
      -I "${dir_of_f}"
      ${f} -o ${output}
      DEPENDS ${embed_tool} ${f} ${output}.deps
      COMMENT "Generating ${output}"
    )
    list(APPEND opencl_generated_cl_includes "${output}")
  endforeach()

  set_source_files_properties(opencl_generated_cl_includes PROPERTIES GENERATED TRUE)
  add_custom_target(gen_opencl_embed_hdrs DEPENDS ${opencl_generated_cl_includes})

  file(GLOB_RECURSE opencl_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/opencl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/opencl/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${opencl_cc_srcs})

  onnxruntime_add_static_library(onnxruntime_providers_opencl ${opencl_cc_srcs})
  target_include_directories(onnxruntime_providers_opencl PRIVATE ${ONNXRUNTIME_ROOT} INTERFACE ${opencl_target_dir})

  set_target_properties(onnxruntime_providers_opencl PROPERTIES
    LINKER_LANGUAGE CXX
    FOLDER "ONNXRuntime"
  )
  target_compile_options(onnxruntime_providers_opencl PRIVATE "-Wno-error=unused-parameter")
  target_compile_options(onnxruntime_providers_opencl PRIVATE "-Wno-error=unused-variable")
  target_compile_options(onnxruntime_providers_opencl PRIVATE "-Wno-error=return-type")


  if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    # No supported now
    # add_subdirectory(${PROJECT_SOURCE_DIR}/external/cl3w)
    # target_link_libraries(onnxruntime_providers_opencl PUBLIC cl3w)
  else()
    find_package(OpenCL REQUIRED)
    target_compile_definitions(onnxruntime_providers_opencl PUBLIC "CL_TARGET_OPENCL_VERSION=120")
    target_link_libraries(onnxruntime_providers_opencl PUBLIC OpenCL::OpenCL)
  endif()
  onnxruntime_add_include_to_target(onnxruntime_providers_opencl onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB}
  flatbuffers Boost::mp11 ${GSL_TARGET} date::date onnxruntime_util safeint_interface Eigen3::Eigen)
  add_dependencies(onnxruntime_providers_opencl onnxruntime_providers gen_opencl_embed_hdrs)
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/opencl  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
endif()
