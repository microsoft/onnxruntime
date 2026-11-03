import json
import os
from pathlib import Path
import subprocess

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent
PARENT = Path("/workspace/qwen38-int4-prefill-dense-stack/build-v1")
COMPONENT = Path("/workspace/qwen38-int4-qsa-parallel-map/build-v1")
OUTPUT = Path("/workspace/qwen38-int4-qsa-concat-vector/build-v1")
FLAG = "ORT_WEBGPU_QSA_CONCAT_VECTOR"


def transform(source):
    start = source.index("Status SparseAttentionIndexerQsaConcatProgram::GenerateShaderCode")
    stop = source.index("Status SparseAttentionIndexerQsaSelectProgram::GenerateShaderCode", start)
    shader = source[start:stop].replace("SparseAttentionIndexerQsaConcatProgram", "QsaVectorConcatProgram")
    shader = shader.replace("ShaderUsage::UseElementTypeAlias", "ShaderUsage::UseValueTypeAlias")
    shader = shader.replace("present_key_element_t(", "present_key_value_t(")
    declaration = """class QsaVectorConcatProgram final : public Program<QsaVectorConcatProgram> {
 public:
  QsaVectorConcatProgram(bool has_past, bool has_current)
      : Program{"QsaConcatVector4"}, has_past_{has_past}, has_current_{has_current} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"total", ProgramUniformVariableDataType::Uint32},
      {"sequence_length", ProgramUniformVariableDataType::Uint32},
      {"past_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"total_sequence_length", ProgramUniformVariableDataType::Uint32},
      {"head_size", ProgramUniformVariableDataType::Uint32},
      {"key_row_stride", ProgramUniformVariableDataType::Uint32},
      {"key_offset", ProgramUniformVariableDataType::Uint32});
 private:
  bool has_past_;
  bool has_current_;
};

"""
    source = source[:start] + declaration + shader + source[start:]
    start = source.index("  if (present_elements > 0) {", source.index("Status SparseAttentionIndexer::ComputeQsa"))
    stop = source.index("  const int64_t rows = batch_size * sequence_length;", start)
    original = source[start:stop]
    vector = original.replace("SparseAttentionIndexerQsaConcatProgram", "QsaVectorConcatProgram")
    vector = replace_once(vector, "{past_key, ProgramTensorMetadataDependency::Type}",
                          "{past_key, ProgramTensorMetadataDependency::Type, {past_key->Shape().Size() / 4}, 4}")
    vector = replace_once(vector, "{current_key, ProgramTensorMetadataDependency::Type}",
                          "{current_key, ProgramTensorMetadataDependency::Type, {current_key->Shape().Size() / 4}, 4}")
    vector = replace_once(vector, "{present, ProgramTensorMetadataDependency::Type}",
                          "{present, ProgramTensorMetadataDependency::Type, {present_elements / 4}, 4}")
    vector = vector.replace("ToUint32(present_elements)", "ToUint32(present_elements / 4)")
    vector = replace_once(vector, "ToUint32(head_size)", "ToUint32(head_size / 4)")
    vector = replace_once(vector, "ToUint32(packed_qk ? query_shape[2] : head_size)",
                          "ToUint32((packed_qk ? query_shape[2] : head_size) / 4)")
    vector = replace_once(vector, "ToUint32(packed_qk ? num_heads * head_size : 0)",
                          "ToUint32((packed_qk ? num_heads * head_size : 0) / 4)")
    selector = f"""  const int concat_vector = ParseEnvironmentVariableWithDefault<int>("{FLAG}", 0);
  ORT_RETURN_IF_NOT(concat_vector == 0 || concat_vector == 1, "{FLAG} must be 0 or 1");
  if (concat_vector == 1 && head_size % 4 == 0 &&
      context.AdapterInfo().vendor == std::string_view{{"nvidia"}}) {{
{vector}  }} else {{
{original}  }}
"""
    return source[:start] + selector + source[stop:]


def main(output=OUTPUT, source_transform=transform, parent_path=PARENT, component_path=COMPONENT,
         no_arithmetic_changes=True):
    parent = json.loads((parent_path / "build-manifest.json").read_text())
    component = json.loads((component_path / "build-manifest.json").read_text())
    original = component_path / "sparse_attention_indexer.cc"
    assert sha(original) == component["source_sha256"][original.name]
    output.mkdir(parents=True, exist_ok=False)
    scratch = output / "scratch"
    scratch.mkdir()
    environment = dict(os.environ, TMPDIR=str(scratch))
    library = output / "libonnxruntime_providers_webgpu.so"
    link = parent["link_command"].copy()
    link[link.index("-o") + 1] = str(library)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    assert sha(library) == parent["provider_sha256"], "Combined parent relink mismatch"
    library.unlink()
    source = output / original.name
    source.write_text(source_transform(original.read_text()))
    obj = output / (source.name + ".o")
    command = component["compile_commands"][0].copy()
    command[1:1] = ["-I" + str(component_path)]
    command[command.index("-c") + 1] = str(source)
    command[command.index("-o") + 1] = str(obj)
    subprocess.run(command, cwd=parent["working_directory"], env=environment, check=True)
    assert link.count(str(component_path / obj.name)) == 1
    link[link.index(str(component_path / obj.name))] = str(obj)
    subprocess.run(link, cwd=parent["working_directory"], env=environment, check=True)
    report = dict(status="built_not_qualified", provider_sha256=sha(library),
        baseline_provider_sha256=parent["provider_sha256"], baseline_byte_identical=True,
        compile_commands=[command], link_command=link, working_directory=parent["working_directory"],
        source_sha256={source.name: sha(source)}, no_existing_kernel_object_layout_changes=True,
        no_arithmetic_changes=no_arithmetic_changes, quality_qualified=False, headline_eligible=False)
    (output / "build-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "provider_sha256", "baseline_byte_identical")}))


if __name__ == "__main__":
    main()
