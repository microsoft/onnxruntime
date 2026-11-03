namespace {
struct ActiveNodeBindingCapture {
  const ComputeContextBase* context = nullptr;
  webgpu::NodeBindingSnapshot* snapshot = nullptr;
};
thread_local ActiveNodeBindingCapture active_node_binding_capture;
}  // namespace

void WebGpuContext::BeginNodeBindingCapture(
    const ComputeContextBase& context, std::vector<webgpu::CapturedCommandInfo>& commands,
    webgpu::NodeBindingSnapshot& snapshot) {
  ORT_ENFORCE(active_node_binding_capture.context == nullptr, "Cannot nest node binding captures.");
  BeginNodeCapture(context, commands);
  active_node_binding_capture = {&context, &snapshot};
}

void WebGpuContext::EndNodeBindingCapture(const ComputeContextBase& context) {
  active_node_binding_capture = {};
  EndNodeCapture(context);
}

Status WebGpuContext::PrepareNodeBinding(
    const webgpu::CapturedCommandInfo& prototype,
    const webgpu::NodeBindingSnapshot& snapshot, const std::vector<WGPUBuffer>& buffers,
    webgpu::CapturedCommandInfo& command) {
  ORT_RETURN_IF_NOT(snapshot.programs == 1 && snapshot.layout &&
                        snapshot.buffers.size() == buffers.size() &&
                        snapshot.segments.size() == buffers.size(),
                    "Invalid node binding prototype.");
  ORT_RETURN_IF_NOT(prototype.compute_pipeline && !prototype.pending_build &&
                        (!prototype.dynamic_uniform_offset || *prototype.dynamic_uniform_offset == 0) &&
                        prototype.indirect_buffer == nullptr,
                    "Node binding prototype is not a ready direct dispatch.");
  for (auto buffer : buffers) {
    ORT_RETURN_IF_NOT(buffer != nullptr, "Node binding replay received a null GPU buffer.");
  }
  const int capacity = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES", 0);
  ORT_RETURN_IF_NOT(capacity >= 0 && capacity <= 8192,
                    "ORT_WEBGPU_BIND_GROUP_CACHE_ENTRIES must be between 0 and 8192.");
  command = prototype;
  command.bind_group = CreateBindGroup(buffers, snapshot.segments, snapshot.layout,
                                       snapshot.program_name, 0, 0, static_cast<size_t>(capacity));
  return Status::OK();
}
