from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_READY_PIPELINE_STATE"
HELPER = r'''
namespace {
bool ReadyPipelineStateEnabled() {
  static const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_READY_PIPELINE_STATE", 0);
  ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_READY_PIPELINE_STATE must be 0 or 1");
  return value == 1;
}

bool ReadyPipelineStateTraceEnabled() {
  static const int value = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_READY_PIPELINE_STATE_TRACE", 0);
  ORT_ENFORCE(value == 0 || value == 1, "ORT_WEBGPU_READY_PIPELINE_STATE_TRACE must be 0 or 1");
  return value == 1;
}

struct ReadyPipelineState {
  webgpu::CommandRecordingState* owner = nullptr;
  WGPUComputePassEncoder encoder = nullptr;
  WGPUComputePipeline pipeline = nullptr;
  uint64_t sets = 0;
  uint64_t skips = 0;
  void Forget() noexcept {
    encoder = nullptr;
    pipeline = nullptr;
  }
};

thread_local ReadyPipelineState* active_ready_pipeline_state = nullptr;

class ScopedReadyPipelineState {
 public:
  ScopedReadyPipelineState(webgpu::CommandRecordingState& recording, bool enabled) noexcept
      : state_{enabled ? &recording : nullptr}, previous_{active_ready_pipeline_state} {
    if (previous_) previous_->Forget();
    active_ready_pipeline_state = enabled ? &state_ : nullptr;
  }
  ~ScopedReadyPipelineState() noexcept {
    if (previous_) previous_->Forget();
    active_ready_pipeline_state = previous_;
  }
  ScopedReadyPipelineState(const ScopedReadyPipelineState&) = delete;
  ScopedReadyPipelineState& operator=(const ScopedReadyPipelineState&) = delete;
  const ReadyPipelineState& State() const noexcept { return state_; }

 private:
  ReadyPipelineState state_;
  ReadyPipelineState* previous_;
};

void ForgetReadyPipelineState(webgpu::CommandRecordingState& recording) noexcept {
  auto* state = active_ready_pipeline_state;
  if (state && state->owner == &recording) state->Forget();
}

bool ShouldSetReadyPipeline(webgpu::CommandRecordingState& recording,
                            WGPUComputePassEncoder encoder, WGPUComputePipeline pipeline) noexcept {
  auto* state = active_ready_pipeline_state;
  if (!state || state->owner != &recording || !encoder || !pipeline) return true;
  if (state->encoder == encoder && state->pipeline == pipeline) {
    ++state->skips;
    return false;
  }
  state->encoder = encoder;
  state->pipeline = pipeline;
  ++state->sets;
  return true;
}
}  // namespace

'''


def transform(source):
    source = replace_once(source, "Status WebGpuContext::EncodeDeferredDispatches(",
                          HELPER + "Status WebGpuContext::EncodeDeferredDispatches(")
    source = replace_once(source,
                          "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {",
                          "Status WebGpuContext::Run(ComputeContextBase& context, const ProgramBase& program) {\n"
                          "  if (ReadyPipelineStateEnabled()) (void)ReadyPipelineStateTraceEnabled();")
    source = replace_once(
        source, "  // Encode the recorded dispatches in order, using the same command objects for graph capture.",
        "  const bool cache_ready_pipeline = ReadyPipelineStateEnabled() && !is_profiling_ &&\n"
        "                                    recording.graph_capture_state == GraphCaptureState::Default;\n"
        "  ScopedReadyPipelineState ready_pipeline_scope(recording, cache_ready_pipeline);\n"
        "  // Encode the recorded dispatches in order, using the same command objects for graph capture.")
    source = replace_once(source, "  compute_pass_encoder.SetPipeline(*command.compute_pipeline);",
                          "  if (ShouldSetReadyPipeline(recording, compute_pass_encoder.Get(), command.compute_pipeline->Get())) {\n"
                          "    compute_pass_encoder.SetPipeline(*command.compute_pipeline);\n"
                          "  }")
    source = replace_once(source, "  ORT_ENFORCE(command.bind_group != nullptr);\n"
                          "  const auto& compute_pass_encoder = GetComputePassEncoder(recording);",
                          "  ORT_ENFORCE(command.bind_group != nullptr);\n"
                          "  if (!recording.compute_pass_encoder) ForgetReadyPipelineState(recording);\n"
                          "  const auto& compute_pass_encoder = GetComputePassEncoder(recording);")
    source = replace_once(
        source, "    EndComputePass(recording);\n  }\n}\n\nvoid WebGpuContext::CaptureBegin",
        "    EndComputePass(recording);\n"
        "    ForgetReadyPipelineState(recording);\n"
        "  }\n}\n\nvoid WebGpuContext::CaptureBegin")
    return replace_once(
        source, "  reset_deferred_state();\n  return result;\n}\n\nbool WebGpuContext::CanCaptureNodeCommands",
        '  if (cache_ready_pipeline && ReadyPipelineStateTraceEnabled()) {\n'
        '    const auto& state = ready_pipeline_scope.State();\n'
        '    LOGS_DEFAULT(INFO) << "WEBGPU_READY_PIPELINE_STATE sets=" << state.sets << " skips=" << state.skips;\n'
        '  }\n'
        "  reset_deferred_state();\n  return result;\n}\n\nbool WebGpuContext::CanCaptureNodeCommands")
