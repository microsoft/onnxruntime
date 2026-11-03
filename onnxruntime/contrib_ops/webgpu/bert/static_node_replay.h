#pragma once

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <mutex>

#include "core/platform/env_var_utils.h"
#include "core/providers/webgpu/compute_context.h"
#include "gsl/gsl"

namespace onnxruntime::webgpu {

class StaticNodeReplay {
 public:
  StaticNodeReplay(const char* name, std::initializer_list<int> gpu_inputs)
      : name_(name), gpu_inputs_(gpu_inputs) {}

  ~StaticNodeReplay() {
    const char* path = std::getenv("ORT_WEBGPU_STATIC_NODE_REPLAY_STATS_FILE");
    if (path == nullptr || requests_ == 0) return;
    static std::mutex statistics_mutex;
    std::lock_guard<std::mutex> lock(statistics_mutex);
    std::ofstream output(path, std::ios::app);
    output << "{\"operator\":\"" << name_ << "\",\"requests\":" << requests_
           << ",\"hits\":" << hits_ << ",\"captures\":" << captures_
           << ",\"rebinds\":" << rebinds_
           << ",\"binding_hits\":" << binding_hits_
           << ",\"binding_entries\":" << binding_entries_.size()
           << ",\"binding_evictions\":" << binding_evictions_
           << ",\"binding_capacity\":" << kCapacity
           << ",\"evictions\":" << evictions_ << ",\"entries\":" << entries_.size()
           << ",\"capacity\":" << kCapacity << "}\n";
    if (!output) {
      LOGS_DEFAULT(ERROR) << "Cannot write static node replay statistics to " << path;
    }
  }

  template <typename Compute>
  Status Run(ComputeContext& context, std::initializer_list<Tensor*> outputs, Compute&& compute,
             int64_t row_elements = 0) {
    const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_STATIC_NODE_REPLAY", 0);
    ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_STATIC_NODE_REPLAY must be 0 or 1");
    const auto* first = context.Input<Tensor>(0);
    if (enabled == 0 || !context.CanCaptureNodeCommands() || context.NodeReplayProfilingEnabled() ||
        first == nullptr || !first->IsDataType<MLFloat16>() ||
        first->Shape().NumDimensions() == 0 || first->Shape().GetDims().back() <= 0 ||
        first->Shape().Size() != (row_elements > 0 ? row_elements : first->Shape().GetDims().back())) {
      return compute();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint64_t> key;
    key.reserve(96);
    key.push_back(context.EnableMatmulFp32Accumulation());
    key.push_back(context.InputCount());
    auto append = [&key](const Tensor* tensor, bool cpu_values) {
      key.push_back(tensor != nullptr);
      if (tensor == nullptr) return;
      key.push_back(tensor->GetElementType());
      key.push_back(tensor->Shape().NumDimensions());
      for (int64_t dimension : tensor->Shape().GetDims()) key.push_back(static_cast<uint64_t>(dimension));
      key.push_back(cpu_values);
      if (cpu_values) {
        key.push_back(tensor->SizeInBytes());
        const auto* bytes = static_cast<const uint8_t*>(tensor->DataRaw());
        for (size_t i = 0; i < tensor->SizeInBytes(); ++i) key.push_back(bytes[i]);
      }
    };
    for (int index = 0; index < context.InputCount(); ++index) {
      const auto* input = context.Input<Tensor>(index);
      const bool cpu = input != nullptr && input->Location().device.Type() == OrtDevice::CPU;
      if (cpu && input->SizeInBytes() > 64) return compute();
      append(input, cpu);
    }
    key.push_back(outputs.size());
    for (const auto* output : outputs) append(output, false);
    auto bindings = [&]() {
      std::vector<WGPUBuffer> buffers;
      buffers.reserve(gpu_inputs_.size() + outputs.size() + 1);
      for (int index : gpu_inputs_) {
        const auto* tensor = context.Input<Tensor>(index);
        ORT_ENFORCE(tensor != nullptr, "Static node replay requires non-null shader inputs.");
        const auto& location = tensor->Location();
        ORT_ENFORCE(location.device.Type() == OrtDevice::GPU &&
                        location.mem_type == OrtMemType::OrtMemTypeDefault && location.name == WEBGPU_BUFFER,
                    "Static node replay requires default WebGPU-buffer shader inputs.");
        buffers.push_back(reinterpret_cast<WGPUBuffer>(const_cast<void*>(tensor->DataRaw())));
      }
      for (auto* tensor : outputs) {
        ORT_ENFORCE(tensor != nullptr, "Static node replay requires non-null shader outputs.");
        const auto& location = tensor->Location();
        ORT_ENFORCE(location.device.Type() == OrtDevice::GPU &&
                        location.mem_type == OrtMemType::OrtMemTypeDefault && location.name == WEBGPU_BUFFER,
                    "Static node replay requires default WebGPU-buffer shader outputs.");
        buffers.push_back(reinterpret_cast<WGPUBuffer>(tensor->MutableDataRaw()));
      }
      return buffers;
    };
    ++requests_;
    for (auto& entry : entries_) {
      if (entry.key == key) {
        auto buffers = bindings();
        if (entry.bindings.uniform_buffer != nullptr) buffers.push_back(entry.bindings.uniform_buffer);
        auto bound = std::find_if(binding_entries_.begin(), binding_entries_.end(), [&](const auto& candidate) {
          return candidate.prototype_id == entry.id && candidate.buffers == buffers;
        });
        if (bound != binding_entries_.end()) {
          ORT_RETURN_IF_ERROR(context.AppendNodeCommands(bound->commands));
          bound->last_used = ++clock_;
          ++binding_hits_;
        } else {
          BindingEntry binding;
          binding.prototype_id = entry.id;
          binding.buffers = std::move(buffers);
          binding.commands.resize(1);
          ORT_RETURN_IF_ERROR(context.PrepareNodeBinding(entry.commands.front(), entry.bindings,
                                                         binding.buffers, binding.commands.front()));
          ORT_RETURN_IF_ERROR(context.AppendNodeCommands(binding.commands));
          StoreBinding(std::move(binding));
          ++rebinds_;
        }
        entry.last_used = ++clock_;
        ++hits_;
        return Status::OK();
      }
    }
    if (entries_.size() == kCapacity) {
      auto oldest = std::min_element(entries_.begin(), entries_.end(),
                                     [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
      binding_entries_.erase(std::remove_if(binding_entries_.begin(), binding_entries_.end(),
                                            [&](const auto& binding) { return binding.prototype_id == oldest->id; }),
                              binding_entries_.end());
      entries_.erase(oldest);
      ++evictions_;
    }
    Entry entry;
    entry.key = std::move(key);
    context.BeginNodeBindingCapture(entry.commands, entry.bindings);
    auto end_capture = gsl::finally([&context]() { context.EndNodeBindingCapture(); });
    ORT_RETURN_IF_ERROR(compute());
    ORT_RETURN_IF_ERROR(context.FlushNodeCommands());
    ORT_RETURN_IF_NOT(entry.commands.size() == 1,
                      "Static node replay expected exactly one command for ", name_,
                      ", got ", entry.commands.size());
    auto expected = bindings();
    if (entry.bindings.uniform_buffer != nullptr) expected.push_back(entry.bindings.uniform_buffer);
    ORT_RETURN_IF_NOT(entry.bindings.programs == 1 && entry.bindings.buffers == expected,
                      "Static node replay found non-parent buffers or a different shader binding order for ", name_);
    entry.id = ++prototype_clock_;
    BindingEntry initial;
    initial.prototype_id = entry.id;
    initial.buffers = std::move(expected);
    initial.commands = entry.commands;
    StoreBinding(std::move(initial));
    entry.last_used = ++clock_;
    ++captures_;
    entries_.push_back(std::move(entry));
    return Status::OK();
  }

 private:
  static constexpr size_t kCapacity = 64;
  struct Entry {
    uint64_t id = 0;
    std::vector<uint64_t> key;
    uint64_t last_used = 0;
    std::vector<CapturedCommandInfo> commands;
    NodeBindingSnapshot bindings;
  };
  struct BindingEntry {
    uint64_t prototype_id = 0;
    uint64_t last_used = 0;
    std::vector<WGPUBuffer> buffers;
    std::vector<CapturedCommandInfo> commands;
  };
  void StoreBinding(BindingEntry binding) {
    if (binding_entries_.size() == kCapacity) {
      auto oldest = std::min_element(binding_entries_.begin(), binding_entries_.end(),
                                     [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
      binding_entries_.erase(oldest);
      ++binding_evictions_;
    }
    binding.last_used = ++clock_;
    binding_entries_.push_back(std::move(binding));
  }
  const char* name_;
  const std::vector<int> gpu_inputs_;
  std::mutex mutex_;
  std::vector<Entry> entries_;
  std::vector<BindingEntry> binding_entries_;
  uint64_t requests_ = 0;
  uint64_t hits_ = 0;
  uint64_t captures_ = 0;
  uint64_t evictions_ = 0;
  uint64_t rebinds_ = 0;
  uint64_t binding_hits_ = 0;
  uint64_t binding_evictions_ = 0;
  uint64_t prototype_clock_ = 0;
  uint64_t clock_ = 0;
};

}  // namespace onnxruntime::webgpu
