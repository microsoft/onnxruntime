from pathlib import Path

from build_dense_replay_scratch import build_variant
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_FAST"
OUTPUT = Path("/workspace/qwen38-int4-dense-output-shape/build-v2")
KERNEL = "MatMulNBitsOutputShape"

ORIGINAL = '''  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
  auto output_shape = helper.OutputShape();'''

SHAPE = '''  TensorShape output_shape;
  const auto& a_shape = a->Shape();
  if (output_shape_fast == 1 && K_ > 0 && a_shape.NumDimensions() >= 2 &&
      a_shape.Size() == K_ && a_shape[a_shape.NumDimensions() - 1] == K_) {
    output_shape = a_shape;
    output_shape[output_shape.NumDimensions() - 1] = N_;
    output_shape_fast_hits_.fetch_add(1, std::memory_order_relaxed);
  } else {
    MatMulComputeHelper helper;
    TensorShape b_shape({N_, K_});
    ORT_RETURN_IF_ERROR(helper.Compute(a_shape, b_shape, false, true));
    output_shape = helper.OutputShape();
  }'''

DESTRUCTOR = '''
MatMulNBitsOutputShape::~MatMulNBitsOutputShape() {
  const char* path = std::getenv("ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE");
  if (path == nullptr) return;
  static std::mutex statistics_mutex;
  std::lock_guard<std::mutex> lock(statistics_mutex);
  std::ofstream output(path, std::ios::app);
  output << "{\\"K\\":" << K_ << ",\\"N\\":" << N_
         << ",\\"fast_shape_hits\\":" << output_shape_fast_hits_.load(std::memory_order_relaxed)
         << "}\\n";
  if (!output) {
    LOGS_DEFAULT(ERROR) << "Cannot write dense output shape statistics to " << path;
  }
}
'''


def transform(source):
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, DESTRUCTOR + signature + f'''
  const int output_shape_fast = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(output_shape_fast == 0 || output_shape_fast == 1, "{FLAG} must be 0 or 1");
''')
    return replace_once(source, ORIGINAL, SHAPE)


def header_transform(source):
    source = replace_once(source, "#pragma once", "#pragma once\n#include <atomic>")
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;",
        "  ~MatMulNBitsOutputShape() override;\n"
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;")
    return replace_once(source, "  int64_t K_;",
                        "  mutable std::atomic<uint64_t> output_shape_fast_hits_{0};\n  int64_t K_;")


def main():
    build_variant(OUTPUT, FLAG,
        lambda source: source.replace("StaticNodeReplay", "DenseOutputShapeNodeReplay"),
        "DenseOutputShapeNodeReplay", "MatMulNBitsOutputShapeExact",
        header_name="dense_output_shape_node_replay.h", kernel_type=KERNEL,
        source_transform=transform, header_transform=header_transform)


if __name__ == "__main__":
    main()
