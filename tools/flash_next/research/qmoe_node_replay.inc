QMoE::~QMoE() {
  const char* path = std::getenv("ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE");
  if (path == nullptr || replay_requests_ == 0) {
    return;
  }
  static std::mutex statistics_mutex;
  std::lock_guard<std::mutex> lock(statistics_mutex);
  std::ofstream output(path, std::ios::app);
  output << "{\"kernel_identity\":" << reinterpret_cast<uintptr_t>(this)
         << ",\"requests\":" << replay_requests_
         << ",\"hits\":" << replay_hits_
         << ",\"captures\":" << replay_captures_
         << ",\"entries\":" << replay_entries_.size()
         << ",\"evictions\":" << replay_evictions_
         << ",\"shape_invalidations\":" << replay_shape_invalidations_
         << ",\"capacity_fallbacks\":" << replay_capacity_fallbacks_
         << ",\"capacity\":8}\n";
  if (!output) {
    LOGS_DEFAULT(ERROR) << "Cannot write QMoE command replay statistics to " << path;
  }
}

Status QMoE::ComputeInternal(ComputeContext& context) const {
  const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0);
  ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_QMOE_COMMAND_REPLAY must be 0 or 1");
  if (enabled == 0) {
    return ComputeWithWorkspace(context, nullptr);
  }
  std::lock_guard<std::mutex> lock(replay_mutex_);
  const Tensor* input = context.Input<Tensor>(0);
  const Tensor* routing = context.Input<Tensor>(1);
  if (!context.CanCaptureNodeCommands() || context.NodeReplayProfilingEnabled() ||
      input == nullptr || routing == nullptr ||
      input->Shape().NumDimensions() == 0 || input->Shape().GetDims().back() <= 0 ||
      input->Shape().Size() != input->Shape().GetDims().back() ||
      !input->IsDataType<MLFloat16>() || routing->Shape().Size() != 512 ||
      is_block_fp8_ || is_nvfp4_ || fc1_expert_weight_bits_ != 4 ||
      fc2_expert_weight_bits_ != 4 || block_size_ != 32 || k_ != 10 ||
      !normalize_routing_weights_ || context.Input<Tensor>(8) != nullptr ||
      context.Input<Tensor>(11) != nullptr || context.Input<Tensor>(12) != nullptr ||
      context.Input<Tensor>(13) != nullptr || context.Input<Tensor>(14) != nullptr) {
    return ComputeWithWorkspace(context, nullptr);
  }
  ++replay_requests_;
  std::vector<uint64_t> key;
  std::vector<uint64_t> workspace_key;
  key.reserve(128);
  workspace_key.reserve(96);
  key.push_back(context.NodeReplayProfilingEnabled());
  key.push_back(context.EnableMatmulFp32Accumulation());
  for (const char* flag : {"ORT_WEBGPU_QMOE_INT4_COOPERATIVE", "ORT_WEBGPU_QMOE_SUBGROUP_GATE",
                           "ORT_WEBGPU_QMOE_INT4_PREFILL", "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL",
                           "ORT_WEBGPU_MATMUL_NBITS_SKINNY", "ORT_WEBGPU_MATMUL_GEMV_N96"}) {
    key.push_back(ParseEnvironmentVariableWithDefault<int>(flag, 0) != 0);
  }
  auto append_tensor = [&key, &workspace_key](const Tensor* tensor) {
    key.push_back(tensor == nullptr ? 0 : reinterpret_cast<uintptr_t>(tensor->DataRaw()));
    workspace_key.push_back(tensor != nullptr);
    if (tensor != nullptr) {
      key.push_back(tensor->GetElementType());
      key.push_back(tensor->Shape().NumDimensions());
      workspace_key.push_back(tensor->GetElementType());
      workspace_key.push_back(tensor->Shape().NumDimensions());
      for (int64_t dimension : tensor->Shape().GetDims()) {
        key.push_back(static_cast<uint64_t>(dimension));
        workspace_key.push_back(static_cast<uint64_t>(dimension));
      }
    }
  };
  key.push_back(context.InputCount());
  for (int index = 0; index < context.InputCount(); ++index) {
    append_tensor(context.Input<Tensor>(index));
  }
  append_tensor(context.Output(0, input->Shape()));
  if (!replay_workspace_key_.empty() && replay_workspace_key_ != workspace_key) {
    ORT_RETURN_IF_ERROR(context.FlushNodeCommands());
    replay_entries_.clear();
    for (auto& tensor : replay_scratch_) {
      tensor.reset();
    }
    ++replay_shape_invalidations_;
  }
  replay_workspace_key_ = std::move(workspace_key);
  for (auto& entry : replay_entries_) {
    if (entry.key == key) {
      ORT_RETURN_IF_ERROR(context.AppendNodeCommands(entry.commands));
      entry.last_used = ++replay_clock_;
      ++replay_hits_;
      return Status::OK();
    }
  }
  if (replay_entries_.size() == 8) {
    auto oldest = std::min_element(replay_entries_.begin(), replay_entries_.end(),
                                   [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
    replay_entries_.erase(oldest);
    ++replay_evictions_;
  }
  ReplayEntry entry;
  entry.key = std::move(key);
  context.BeginNodeCapture(entry.commands);
  auto end_capture = gsl::finally([&context]() { context.EndNodeCapture(); });
  ORT_RETURN_IF_ERROR(ComputeWithWorkspace(context, &replay_scratch_));
  ORT_RETURN_IF_ERROR(context.FlushNodeCommands());
  ORT_RETURN_IF_NOT(entry.commands.size() == 5,
                    "QMoE node capture expected exactly five commands, got ", entry.commands.size());
  entry.last_used = ++replay_clock_;
  ++replay_captures_;
  replay_entries_.push_back(std::move(entry));
  return Status::OK();
}
