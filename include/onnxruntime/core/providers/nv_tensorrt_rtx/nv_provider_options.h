#pragma once

/**

 * @namespace onnxruntime::nv::provider_option_names
 * @details The `provider_option_names` namespace contains the following constants:
 * - `kDeviceId`: Specifies the GPU device ID to use.
 * - `kHasUserComputeStream`: Indicates whether a user-provided compute stream is used.
 * - `kUserComputeStream`: Specifies the user-provided compute stream.
 * - `kMaxWorkspaceSize`: Sets the maximum workspace size for GPU memory allocation.
 * - `kDumpSubgraphs`: Enables or disables dumping of subgraphs for debugging.
 * - `kDetailedBuildLog`: Enables or disables detailed build logs for debugging.
 * - `kProfilesMinShapes`: Specifies the minimum shapes for profiling.
 * - `kProfilesMaxShapes`: Specifies the maximum shapes for profiling.
 * - `kProfilesOptShapes`: Specifies the optimal shapes for profiling.
 * - `kCudaGraphEnable`: Enables or disables CUDA graph optimizations.
 * - `kONNXBytestream`: Specifies the ONNX model as a bytestream.
 * - `kONNXBytestreamSize`: Specifies the size of the ONNX bytestream.
 */
namespace onnxruntime {
namespace nv {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";
constexpr const char* kUserComputeStream = "user_compute_stream";
constexpr const char* kMaxWorkspaceSize = "nv_max_workspace_size";
constexpr const char* kDumpSubgraphs = "nv_dump_subgraphs";
constexpr const char* kDetailedBuildLog = "nv_detailed_build_log";
constexpr const char* kProfilesMinShapes = "nv_profile_min_shapes";
constexpr const char* kProfilesMaxShapes = "nv_profile_max_shapes";
constexpr const char* kProfilesOptShapes = "nv_profile_opt_shapes";
constexpr const char* kCudaGraphEnable = "nv_cuda_graph_enable";
constexpr const char* kONNXBytestream = "nv_onnx_bytestream";
constexpr const char* kONNXBytestreamSize = "nv_onnx_bytestream_size";

}  // namespace provider_option_names
}  // namespace nv
}  // namespace onnxruntime
