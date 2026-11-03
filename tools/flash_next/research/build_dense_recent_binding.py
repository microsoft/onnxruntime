from pathlib import Path

from build_dense_replay_scratch import build_variant
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_RECENT_BINDING"
OUTPUT = Path("/workspace/qwen38-int4-dense-recent-binding/build-v1")

FAST = '''
        if (recent == 1 && recent_binding_ < binding_entries_.size()) {
          auto& previous = binding_entries_[recent_binding_];
          if (previous.prototype_id == entry.id &&
              previous.buffers.size() == gpu_inputs_.size() + outputs.size() +
                                             (entry.bindings.uniform_buffer != nullptr)) {
            size_t position = 0;
            bool match = true;
            for (int index : gpu_inputs_) {
              const auto* tensor = context.Input<Tensor>(index);
              CheckBufferTensor(tensor, "inputs");
              auto buffer = reinterpret_cast<WGPUBuffer>(const_cast<void*>(tensor->DataRaw()));
              match &= previous.buffers[position++] == buffer;
            }
            for (auto* tensor : outputs) {
              CheckBufferTensor(tensor, "outputs");
              auto buffer = reinterpret_cast<WGPUBuffer>(tensor->MutableDataRaw());
              match &= previous.buffers[position++] == buffer;
            }
            if (entry.bindings.uniform_buffer != nullptr) {
              match = match && previous.buffers[position] == entry.bindings.uniform_buffer;
            }
            if (match) {
              ORT_RETURN_IF_ERROR(context.AppendNodeCommands(previous.commands));
              previous.last_used = ++clock_;
              ++binding_hits_;
              ++recent_binding_hits_;
              entry.last_used = ++clock_;
              ++hits_;
              return Status::OK();
            }
          }
        }
'''

CHECK = '''
  static void CheckBufferTensor(const Tensor* tensor, const char* kind) {
    ORT_ENFORCE(tensor != nullptr, "Static node replay requires non-null shader ", kind, ".");
    const auto& location = tensor->Location();
    ORT_ENFORCE(location.device.Type() == OrtDevice::GPU &&
                    location.mem_type == OrtMemType::OrtMemTypeDefault && location.name == WEBGPU_BUFFER,
                "Static node replay requires default WebGPU-buffer shader ", kind, ".");
  }
'''


def transform(source):
    source = source.replace("StaticNodeReplay", "DenseRecentNodeReplay")
    signature = "int64_t row_elements = 0, uint64_t variant = 0) {"
    source = replace_once(source, signature, signature + f'''
    const int recent = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_RETURN_IF_NOT(recent == 0 || recent == 1, "{FLAG} must be 0 or 1");
''')
    for kind in ("inputs", "outputs"):
        start = source.index('        ORT_ENFORCE(tensor != nullptr, "Static node replay requires non-null shader ' + kind)
        end = source.index("        buffers.push_back(", start)
        source = source[:start] + f'        CheckBufferTensor(tensor, "{kind}");\n' + source[end:]
    source = replace_once(source, "      if (entry.key == key) {\n",
                          "      if (entry.key == key) {\n" + FAST)
    source = replace_once(source, "          bound->last_used = ++clock_;",
                          "          recent_binding_ = static_cast<size_t>(bound - binding_entries_.begin());\n"
                          "          bound->last_used = ++clock_;")
    source = replace_once(source, "    binding_entries_.push_back(std::move(binding));",
                          "    binding_entries_.push_back(std::move(binding));\n"
                          "    recent_binding_ = binding_entries_.size() - 1;")
    source = replace_once(source, " private:\n", " private:\n" + CHECK)
    source = replace_once(source, "  uint64_t binding_hits_ = 0;",
                          "  uint64_t binding_hits_ = 0;\n"
                          "  uint64_t recent_binding_hits_ = 0;\n"
                          "  size_t recent_binding_ = kCapacity;")
    source = replace_once(source, '           << ",\\"binding_hits\\":" << binding_hits_',
                          '           << ",\\"binding_hits\\":" << binding_hits_\n'
                          '           << ",\\"recent_binding_hits\\":" << recent_binding_hits_')
    return source


def main():
    build_variant(OUTPUT, FLAG, transform, "DenseRecentNodeReplay", "MatMulNBitsRecentExact",
                  header_name="dense_recent_node_replay.h", kernel_type="MatMulNBitsRecent")


if __name__ == "__main__":
    main()
