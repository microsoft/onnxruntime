from pathlib import Path
import re

from build_adapter_tensor_reuse_sdk import main as build_sdk
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_ADAPTER_INLINE_TENSORS"
OUTPUT = Path("/workspace/qwen38-int4-adapter-inline-tensors/build-v2")
HELPERS = f'''
struct AdapterInlineTensorSettings {{
  bool enabled;
  const char* statistics_path;
}};

inline const AdapterInlineTensorSettings& GetAdapterInlineTensorSettings() {{
  static const AdapterInlineTensorSettings settings = []() {{
    const int mode = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(mode == 0 || mode == 1, "{FLAG} must be 0 or 1");
#if defined(DISABLE_ABSEIL)
    ORT_ENFORCE(mode == 0, "{FLAG} requires Abseil inline-container support");
#endif
    return AdapterInlineTensorSettings{{mode == 1, std::getenv("ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE")}};
  }}();
  return settings;
}}

struct AdapterInlineTensorStatistics {{
  std::atomic<size_t> contexts{{0}};
  std::atomic<size_t> enabled_contexts{{0}};
  std::atomic<size_t> input_heap_fallbacks{{0}};
  std::atomic<size_t> output_heap_fallbacks{{0}};
  ~AdapterInlineTensorStatistics() {{
    const char* path = GetAdapterInlineTensorSettings().statistics_path;
    if (path == nullptr) return;
    std::ofstream output(path, std::ios::app);
    output << "{{\\"operator\\":\\"AdapterInlineTensors\\",\\"contexts\\":" << contexts.load()
           << ",\\"enabled_contexts\\":" << enabled_contexts.load()
           << ",\\"input_heap_fallbacks\\":" << input_heap_fallbacks.load()
           << ",\\"output_heap_fallbacks\\":" << output_heap_fallbacks.load()
           << ",\\"input_capacity\\":8,\\"output_capacity\\":5}}\\n";
    if (!output) {{
      LOGS_DEFAULT(ERROR) << "Cannot write inline tensor statistics to " << path;
    }}
  }}
}};

inline AdapterInlineTensorStatistics& GetAdapterInlineTensorStatistics() {{
  static AdapterInlineTensorStatistics statistics;
  return statistics;
}}

'''
FIELDS = '''
  bool inline_enabled_ = false;
  mutable InlinedVector<Tensor, 8> inline_input_tensors_;
  InlinedVector<Tensor, 5> inline_output_tensors_;

  size_t InputTensorCount() const {
    return inline_enabled_ ? inline_input_tensors_.size() : input_tensors_.size();
  }
  size_t OutputTensorCount() const {
    return inline_enabled_ ? inline_output_tensors_.size() : output_tensors_.size();
  }
  Tensor& InputTensorAt(int index) const {
    return inline_enabled_ ? inline_input_tensors_[index] : input_tensors_[index];
  }
  Tensor& OutputTensorAt(int index) {
    return inline_enabled_ ? inline_output_tensors_[index] : output_tensors_[index];
  }
'''


def transform(source):
    source = replace_once(source, "#pragma once",
        '#pragma once\n\n#include <atomic>\n#include <cstdlib>\n#include <fstream>\n'
        '#include "core/common/inlined_containers_fwd.h"\n'
        '#include "core/platform/env_var_utils.h"')
    source = replace_once(source, "struct OpKernelContext {", HELPERS + "struct OpKernelContext {")
    start = source.index("struct OpKernelContext {")
    end = source.index("\n};", start)
    context = source[start:end]
    context = re.sub(r"\binput_tensors_\[index\]", "InputTensorAt(index)", context)
    context = re.sub(r"\boutput_tensors_\[index\]", "OutputTensorAt(index)", context)
    context = re.sub(r"\binput_tensors_\.size\(\)", "InputTensorCount()", context)
    context = re.sub(r"\boutput_tensors_\.size\(\)", "OutputTensorCount()", context)
    context = replace_once(context,
        "    input_tensors_.resize(context_.GetInputCount());\n"
        "    output_tensors_.resize(context_.GetOutputCount());",
        '''    const auto& settings = GetAdapterInlineTensorSettings();
    inline_enabled_ = settings.enabled;
    const size_t inputs = context_.GetInputCount();
    const size_t outputs = context_.GetOutputCount();
    if (inline_enabled_) {
      inline_input_tensors_.resize(inputs);
      inline_output_tensors_.resize(outputs);
    } else {
      input_tensors_.resize(inputs);
      output_tensors_.resize(outputs);
    }
    if (settings.statistics_path != nullptr) {
      auto& statistics = GetAdapterInlineTensorStatistics();
      statistics.contexts.fetch_add(1, std::memory_order_relaxed);
      if (inline_enabled_) {
        statistics.enabled_contexts.fetch_add(1, std::memory_order_relaxed);
        statistics.input_heap_fallbacks.fetch_add(inputs > 8, std::memory_order_relaxed);
        statistics.output_heap_fallbacks.fetch_add(outputs > 5, std::memory_order_relaxed);
      }
    }''')
    return source[:start] + context + FIELDS + source[end:]


if __name__ == "__main__":
    build_sdk(OUTPUT, transform, FLAG, object_layout_changes=True)
