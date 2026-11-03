from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
GRAPH_TRANSFORM = '''
def graph_transform(source):
    source = replace_once(source, '#include "session_options.h"',
                          '#include "session_options.h"\\n#include "models/env_utils.h"')
    source = replace_once(source, "namespace Generators {", "namespace Generators {\\n" + SELECTOR)
    return replace_once(source, "  auto session_options = OrtSessionOptions::Create();",
        "  auto session_options = OrtSessionOptions::Create();\\n"
        "  if (device_type == DeviceType::WEBGPU && WebGpuGreedyArgMaxEnabled())\\n"
        "    session_options->SetIntraOpNumThreads(1);")

'''


if __name__ == "__main__":
    source = (ROOT / "build_gpu_greedy_argmax.py").read_text()
    source = replace_once(source, "gpu-greedy-argmax/build-v2", "gpu-greedy-argmax/build-v3")
    trace = '''
static bool WebGpuGreedyArgMaxTraceEnabled() {
  static const bool enabled = []() {
    bool value = false;
    GetEnv("ORT_GENAI_WEBGPU_GREEDY_ARGMAX_TRACE", value);
    return value;
  }();
  return enabled;
}

'''
    source = replace_once(source, "struct LazyWebGpuLogits final", trace + "struct LazyWebGpuLogits final")
    source = replace_once(source, "      GreedySearch_Cpu::SelectTop();",
        '      if (WebGpuGreedyArgMaxTraceEnabled())\n'
        '        std::cerr << "WEBGPU_GREEDY_ARGMAX_CPU_FALLBACK" << std::endl;\n'
        "      GreedySearch_Cpu::SelectTop();")
    source = replace_once(source,
        '      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");',
        '      throw std::runtime_error("WebGPU greedy ArgMax was not admitted");\n'
        '    if (WebGpuGreedyArgMaxTraceEnabled())\n'
        '      std::cerr << "WEBGPU_GREEDY_ARGMAX rows=" << rows << " vocabulary=" << vocab\n'
        '                << " token_bytes=" << rows * sizeof(int64_t) << std::endl;')
    source = replace_once(source, '#include "models/graph_builder.h"',
                          '#include "models/graph_builder.h"\\n#include <iostream>')
    source = replace_once(source, "\ndef main():", "\n" + GRAPH_TRANSFORM + "\ndef main():")
    source = replace_once(source,
        '(("models/model.cpp", model_transform), ("ep/webgpu/interface.cpp", interface_transform))',
        '(("models/model.cpp", model_transform), ("ep/webgpu/interface.cpp", interface_transform),\n'
        '                                ("models/graph_executor.cpp", graph_transform))')
    with (ROOT / "build_gpu_greedy_argmax_v3.py").open("x") as stream:
        stream.write(source)
