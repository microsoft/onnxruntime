from pathlib import Path

from build_dense_output_shape import DESTRUCTOR, ORIGINAL
from build_dense_replay_scratch import build_variant
from build_qmoe_node_replay import replace_once


FLAG = "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATIC"
OUTPUT = Path("/workspace/qwen38-int4-dense-output-shape-static/build-v1")
KERNEL = "MatMulNBitsStaticOutputShape"

SHAPE = '''  Tensor* y = nullptr;
  const auto& a_shape = a->Shape();
  if (output_shape_static_ && K_ > 0 && a_shape.NumDimensions() >= 2 &&
      a_shape.Size() == K_ && a_shape[a_shape.NumDimensions() - 1] == K_) {
    auto output_shape = a_shape;
    output_shape[output_shape.NumDimensions() - 1] = N_;
    output_shape_fast_hits_.fetch_add(1, std::memory_order_relaxed);
    y = context.Output(0, output_shape);
  } else {
    MatMulComputeHelper helper;
    TensorShape b_shape({N_, K_});
    ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b_shape, false, true));
    auto output_shape = helper.OutputShape();
    y = context.Output(0, output_shape);
  }'''


def transform(source):
    destructor = DESTRUCTOR.replace("MatMulNBitsOutputShape", KERNEL).replace(
        "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATS_FILE", "ORT_WEBGPU_DENSE_OUTPUT_SHAPE_STATIC_STATS_FILE")
    signature = "Status MatMulNBits::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {"
    source = replace_once(source, signature, destructor + signature)
    return replace_once(source, ORIGINAL + "\n  Tensor* y = context.Output(0, output_shape);", SHAPE)


def header_transform(source):
    source = replace_once(source, "#pragma once", "#pragma once\n#include <atomic>")
    anchor = '"Only 4b/8b/2b quantization is supported for MatMulNBits op, additional bits support is planned.");'
    source = replace_once(source, anchor, anchor + f'''
    const int output_shape_static = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
    ORT_ENFORCE(output_shape_static == 0 || output_shape_static == 1, "{FLAG} must be 0 or 1");
    output_shape_static_ = output_shape_static == 1;''')
    source = replace_once(source,
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;",
        f"  ~{KERNEL}() override;\n"
        "  Status ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const override;")
    return replace_once(source, "  int64_t K_;",
        "  bool output_shape_static_{false};\n"
        "  mutable std::atomic<uint64_t> output_shape_fast_hits_{0};\n  int64_t K_;")


def comparator_transform(source):
    source = replace_once(source, "      Ort::Session session{env, argv[file], options};",
        '''      flag("0");
      Ort::Session session{env, argv[file], options};
      flag("1");
      Ort::Session candidate_session{env, argv[file], options};''')
    source = replace_once(source, "auto actual = session.Run(", "auto actual = candidate_session.Run(")
    source = replace_once(source,
        "        session.Run(Ort::RunOptions{}, names.data(), inputs.data(), inputs.size(), outputs.data(), outputs.size());",
        "        Ort::Session invalid_session{env, argv[file], options};")
    return source


def main():
    build_variant(OUTPUT, FLAG,
        lambda source: source.replace("StaticNodeReplay", "DenseStaticOutputShapeNodeReplay"),
        "DenseStaticOutputShapeNodeReplay", "MatMulNBitsStaticOutputShapeExact",
        header_name="dense_static_output_shape_node_replay.h", kernel_type=KERNEL,
        source_transform=transform, header_transform=header_transform,
        comparator_transform=comparator_transform)


if __name__ == "__main__":
    main()
