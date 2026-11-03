bool WebGpuContext::CanCaptureNodeCommands(const ComputeContextBase& context) const {
  return !context.IsGraphCaptureEnabled() &&
         ComputeContextBase::BufferManagerAccessor::GetRecording(context).graph_capture_state ==
             GraphCaptureState::Default;
}

void WebGpuContext::BeginNodeCapture(const ComputeContextBase& context,
                                     std::vector<webgpu::CapturedCommandInfo>& commands) {
  ORT_ENFORCE(CanCaptureNodeCommands(context), "Cannot nest a WebGPU node capture.");
  CaptureBegin(&commands, ComputeContextBase::BufferManagerAccessor::Get(context),
               ComputeContextBase::BufferManagerAccessor::GetRecording(context));
}

void WebGpuContext::EndNodeCapture(const ComputeContextBase& context) {
  CaptureEnd(ComputeContextBase::BufferManagerAccessor::GetRecording(context));
}

Status WebGpuContext::FlushNodeCommands(const ComputeContextBase& context) {
  return Flush(ComputeContextBase::BufferManagerAccessor::Get(context),
               ComputeContextBase::BufferManagerAccessor::GetRecording(context));
}

Status WebGpuContext::AppendNodeCommands(
    const ComputeContextBase& context,
    const std::vector<webgpu::CapturedCommandInfo>& commands) {
  ORT_RETURN_IF_NOT(CanCaptureNodeCommands(context), "Cannot replay a node inside WebGPU graph capture.");
  auto& recording = ComputeContextBase::BufferManagerAccessor::GetRecording(context);
  const auto& manager = ComputeContextBase::BufferManagerAccessor::Get(context);
  for (const auto& command : commands) {
    ORT_RETURN_IF_NOT(command.compute_pipeline && command.bind_group && !command.pending_build,
                      "WebGPU node replay command is not ready.");
    ORT_RETURN_IF_NOT(!is_profiling_ || command.pending_kernel_info.has_value(),
                      "WebGPU node replay command lacks profiling metadata.");
    recording.deferred_dispatches.push_back(command);
    recording.has_unsubmitted_work.store(true, std::memory_order_relaxed);
    if (static_cast<size_t>(recording.num_pending_dispatches) + recording.deferred_dispatches.size() >=
        max_num_pending_dispatches_) {
      ORT_RETURN_IF_ERROR(Flush(manager, recording));
    }
  }
  return Status::OK();
}
