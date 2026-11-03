from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    source = (source.replace("adapter-inline-tensors-v2", "dense-flag-cache-v1")
              .replace("qwen38-int4-adapter-inline-tensors/build-v2", "qwen38-int4-dense-flag-cache/build-v1")
              .replace("build_adapter_inline_tensors", "build_dense_flag_cache")
              .replace("run_adapter_inline", "run_dense_flag_cache")
              .replace("adapter_inline_tensor", "dense_flag_cache")
              .replace("ORT_WEBGPU_ADAPTER_INLINE_STATS_FILE", "ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE")
              .replace("complete_context_layout_rebuild", "private_kernel_layout_only")
              .replace('assert not build["no_object_layout_changes"]', 'assert build["private_kernel_layout_only"]')
              .replace('    assert build["complete_sdk_copy_for_relative_include_wiring"]\n', "")
              .replace("inline", "cache"))
    return source


def native_source(source):
    source = base(source)
    source = replace_once(source, "\ndef main():", '''
def verify_dense_rows(rows):
    assert len(rows) == 18
    assert all(row["operator"] == "MatMulNBitsExact" and row["requests"] == 128 and
               row["entries"] == row["captures"] == 1 and row["hits"] == 127 and
               row["rebinds"] + row["binding_hits"] == row["hits"] and
               row["binding_hits"] > 0 for row in rows), rows


def main():''')
    source = replace_once(source, "fp32_accumulation=False)",
                          "fp32_accumulation=False, stats_verifier=verify_dense_rows)")
    start = source.index('        assert rows and all(row["operator"]')
    end = source.index("        evidence.append(stats)", start)
    source = source[:start] + '''        assert rows and all(row["operator"] == "DenseFrozenFlags" and
            row["enabled"] == mode and row["requests"] >= row["parses"] >= 0 and
            (row["parses"] <= 8 if mode else row["requests"] == row["parses"])
            for row in rows), rows
        if mode:
            assert sum(row["parses"] for row in rows) < sum(row["requests"] for row in rows)
''' + source[end:]
    source = replace_once(source,
        '    fixture = "/evidence/webgpu-perf-oct20/gdn-prep-fixtures/gdn_prep_b1_l1.onnx"',
        '''    fixture_path = sorted(Path("/home/sayanshaw/repos/qwen38-int4-generic-matmul-replay/fixtures-v1").glob("*.onnx"))[0]
    fixture = str(fixture_path).replace("/home/sayanshaw/repos", "/workspace", 1)
    comparator = str(DIRECTORY / "native_compare").replace("/home/sayanshaw/repos", "/workspace", 1)''')
    return source.replace("input_cache_capacity=8, output_cache_capacity=5,",
                          "cached_selectors_per_kernel=8,")


def model_source(source):
    source = base(source)
    start = source.index('        assert rows and all(row["operator"]')
    end = source.index("        statistics.append(vector_path)", start)
    source = source[:start] + '''        assert rows and all(row["operator"] == "DenseFrozenFlags" and
            row["enabled"] == 1 and row["requests"] >= row["parses"] and
            0 <= row["parses"] <= 8 for row in rows), rows
        vector_coverage = dict(kernels=len(rows),
            requests=sum(row["requests"] for row in rows), parses=sum(row["parses"] for row in rows))
        assert vector_coverage["requests"] >= result["dense_replay_hits"]
        assert vector_coverage["parses"] < vector_coverage["requests"]
''' + source[end:]
    return source


def main():
    for old, new, transform in (
        ("run_adapter_inline_native.py", "run_dense_flag_cache_native.py", native_source),
        ("run_adapter_inline_model.py", "run_dense_flag_cache_model.py", model_source),
        ("run_adapter_inline_screen.py", "run_dense_flag_cache_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
