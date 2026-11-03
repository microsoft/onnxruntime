from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("adapter-tensor-vector-reuse-v7", "adapter-inline-tensors-v2")
            .replace("qwen38-int4-adapter-tensor-reuse/build-v7",
                     "qwen38-int4-adapter-inline-tensors/build-v2")
            .replace("build_adapter_tensor_reuse", "build_adapter_inline_tensors")
            .replace("run_adapter_tensor_reuse", "run_adapter_inline")
            .replace("adapter_tensor_vector", "adapter_inline_tensor")
            .replace("ORT_WEBGPU_ADAPTER_TENSOR_VECTOR_STATS_FILE", "ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE")
            .replace("reuse", "inline"))


STATISTICS_CHECK = '''        assert rows and all(row["operator"] == "AdapterInlineTensors" and
            row["contexts"] > 0 and row["enabled_contexts"] == row["contexts"] * mode and
            row["input_capacity"] == 8 and row["output_capacity"] == 5 and
            0 <= row["input_heap_fallbacks"] <= row["enabled_contexts"] and
            0 <= row["output_heap_fallbacks"] <= row["enabled_contexts"] for row in rows), rows
'''


def native_source(source):
    source = base(source)
    source = replace_once(source,
        '    assert build["complete_sdk_copy_for_relative_include_wiring"]',
        '''    assert build["complete_sdk_copy_for_relative_include_wiring"]
    assert not build["no_object_layout_changes"]''')
    start = source.index("        if mode:")
    end = source.index('        evidence.append(prefix.with_suffix(".qualification.json"))', start)
    source = source[:start] + '''        rows = [json.loads(line) for line in stats.read_text().splitlines()]
''' + STATISTICS_CHECK + '''        evidence.append(stats)
''' + source[end:]
    return replace_once(source, "        all_152_original_objects_byte_identical=True,",
        '''        all_152_original_objects_byte_identical=True,
        complete_context_layout_rebuild=True, input_inline_capacity=8, output_inline_capacity=5,''')


def model_source(source):
    source = base(source)
    start = source.index("    command[-1] = (")
    end = source.index('    if phase == "coverage":', start)
    source = source[:start] + '''    command[-1] = (
        "unset ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE; "
        f"export {FLAG}={int(mode == 'inline')}; " + command[-1])
    if phase == "coverage" and mode == "inline":
        command[-1] = (
            f"export ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE=/evidence/webgpu-perf-oct20/"
            f"{prefix.name}-adapter-stats.jsonl; " +
            command[-1].replace("unset ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE; ", "", 1))
''' + source[end:]
    source = replace_once(source, '    assert native["all_152_original_objects_byte_identical"]',
        '''    assert native["all_152_original_objects_byte_identical"]
    assert native["complete_context_layout_rebuild"]''')
    source = replace_once(source, '    assert build["complete_sdk_copy_for_relative_include_wiring"]',
        '''    assert build["complete_sdk_copy_for_relative_include_wiring"]
    assert not build["no_object_layout_changes"]''')
    start = source.index("    vector_path =")
    end = source.index('    log = prefix.with_suffix(".log").read_text()', start)
    source = source[:start] + '''    vector_path = ROOT / (prefix.name + "-adapter-stats.jsonl")
    vector_coverage = None
    if args.phase == "coverage" and args.mode == "inline":
        rows = [json.loads(line) for line in vector_path.read_text().splitlines()]
        mode = 1
''' + STATISTICS_CHECK + '''        vector_coverage = dict(
            contexts=sum(row["contexts"] for row in rows),
            enabled_contexts=sum(row["enabled_contexts"] for row in rows),
            input_heap_fallbacks=sum(row["input_heap_fallbacks"] for row in rows),
            output_heap_fallbacks=sum(row["output_heap_fallbacks"] for row in rows),
            input_capacity=8, output_capacity=5)
        assert vector_coverage["contexts"] >= result["dense_replay_hits"]
        statistics.append(vector_path)
    else:
        assert not vector_path.exists()
''' + source[end:]
    return source


def main():
    for old, new, transform in (
        ("run_adapter_tensor_reuse_native.py", "run_adapter_inline_native.py", native_source),
        ("run_adapter_tensor_reuse_model.py", "run_adapter_inline_model.py", model_source),
        ("run_adapter_tensor_reuse_screen.py", "run_adapter_inline_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
