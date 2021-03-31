file(GLOB_RECURSE my_execution_provider_srcs
  "${ONNXRUNTIME_ROOT}/core/providers/my_ep/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/my_ep/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
)

add_library(my_execution_provider SHARED ${my_execution_provider_srcs})
add_dependencies(my_execution_provider onnxruntime_providers_shared)
target_link_libraries(my_execution_provider PRIVATE onnxruntime_providers_shared)
target_include_directories(my_execution_provider PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(my_execution_provider PRIVATE $<TARGET_PROPERTY:onnxruntime_common,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(my_execution_provider PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
if(APPLE)
  set_property(TARGET my_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/my_ep/exported_symbols.lst")
elseif(UNIX)
  set_property(TARGET my_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/my_ep/version_script.lds -Xlinker --gc-sections")
elseif(WIN32)
  set_property(TARGET my_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/my_ep/symbols.def")
else()
  message(FATAL_ERROR "my_execution_provider unknown platform, need to specify shared library exports for it")
endif()
