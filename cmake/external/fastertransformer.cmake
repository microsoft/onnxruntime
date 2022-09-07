set(fastertransformer_root_dir ${PROJECT_SOURCE_DIR}/external/FasterTransformer)

add_definitions(-DUSE_FASTERTRANSFORMER)
include_directories(${fastertransformer_root_dir}/src/fastertransformer/models/bert)

option(SM "" 75)
option(CMAKE_BUILD_TYPE "" Release)

add_subdirectory(${fastertransformer_root_dir})
remove_definitions("-DENABLE_BF16")
set_target_properties(transformer-static PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

