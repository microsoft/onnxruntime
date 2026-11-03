import json
from pathlib import Path

from build_qmoe_node_replay import sha


ROOT = Path(__file__).parent


def main():
    parent = ROOT / "build_gpu_timestamp_ns_v1.py"
    output = ROOT / "build_ready_pipeline_state_v1.py"
    report = ROOT / "ready-pipeline-state-v1-generation.json"
    if output.exists() or report.exists():
        raise FileExistsError("Fresh scoped pipeline-state generation required")
    source = parent.read_text().replace("gpu_timestamp_ns_transform", "ready_pipeline_state_transform").replace(
        "qwen38-int4-gpu-timestamp-ns/build-v1", "qwen38-int4-ready-pipeline-state/build-v1").replace(
            "original3a_compute_with_additive_gpu_profile_nanoseconds",
            "original3a_scoped_redundant_pipeline_state_skip")
    source = source.replace('"precision_qualified": False, ', "")
    source = source.replace('"compute_code_and_all_other_objects_unchanged": True',
                            '"shader_math_and_all_other_objects_unchanged": True')
    compile(source, str(output), "exec")
    with output.open("x") as stream:
        stream.write(source)
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "no_header_or_object_layout_change": True,
                   "source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), parent, output, ROOT / "ready_pipeline_state_transform.py")}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
