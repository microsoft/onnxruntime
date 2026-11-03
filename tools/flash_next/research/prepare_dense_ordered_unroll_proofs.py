from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("dense-flag-cache-v1", "dense-ordered-unroll2-v1")
            .replace("qwen38-int4-dense-flag-cache/build-v1", "qwen38-int4-dense-ordered-unroll2/build-v1")
            .replace("dense_flag_cache", "dense_ordered_unroll")
            .replace("ORT_WEBGPU_DENSE_FLAG_CACHE_STATS_FILE", "ORT_WEBGPU_DENSE_ORDERED_UNROLL2_STATS_FILE")
            .replace("cache_decode_gain", "unroll_decode_gain")
            .replace('"cache"', '"unroll"').replace("'cache'", "'unroll'"))


def native_source(source):
    source = base(source)
    start = source.index('        assert rows and all(row["operator"]')
    end = source.index("        evidence.append(stats)", start)
    source = source[:start] + '''        assert rows and all(row["operator"] == "DenseOrderedUnrollShader" and
            row["mode"] == mode and row["tile_size_k"] > 0 for row in rows), rows
''' + source[end:]
    source = source.replace("cached_selectors_per_kernel=8", "ordered_k_tiles_per_loop=2")
    source = replace_once(source, 'f"if env {FLAG}=2 {comparator}',
                          'f"if env {FLAG}=2 ORT_WEBGPU_DENSE_BLOCK32=1 '
                          'ORT_WEBGPU_DENSE_BITCAST_UNPACK=1 ORT_TEST_DENSE_FP32_ACCUMULATION=0 {comparator}')
    return source


def model_source(source):
    source = base(source)
    start = source.index('        assert rows and all(row["operator"]')
    end = source.index("        statistics.append(vector_path)", start)
    source = source[:start] + '''        assert rows and all(row["operator"] == "DenseOrderedUnrollShader" and
            row["mode"] == 1 and row["tile_size_k"] > 0 for row in rows), rows
        vector_coverage = dict(generated_shaders=len(rows),
            tile_sizes=sorted({row["tile_size_k"] for row in rows}),
            actual_shader_generation_only=True, not_a_standard_tps_claim=True)
''' + source[end:]
    return source


def main():
    for old, new, transform in (
        ("run_dense_flag_cache_native.py", "run_dense_ordered_unroll_native.py", native_source),
        ("run_dense_flag_cache_model.py", "run_dense_ordered_unroll_model.py", model_source),
        ("run_dense_flag_cache_screen.py", "run_dense_ordered_unroll_screen.py", base)):
        target = ROOT / new
        with target.open("x") as stream:
            stream.write(transform((ROOT / old).read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
