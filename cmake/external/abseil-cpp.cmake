set(abseil_cpp_root_dir ${PROJECT_SOURCE_DIR}/external/abseil-cpp)

set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)
add_subdirectory(${abseil_cpp_root_dir} EXCLUDE_FROM_ALL)
include_directories(${abseil_cpp_root_dir})

list(APPEND onnxruntime_EXTERNAL_LIBRARIES absl::inlined_vector absl::flat_hash_set absl::flat_hash_map absl::base absl::throw_delegate)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES absl::inlined_vector absl::flat_hash_set absl::flat_hash_map absl::base absl::throw_delegate)

