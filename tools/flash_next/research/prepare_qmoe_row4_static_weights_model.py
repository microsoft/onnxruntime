from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def base(source):
    return (source.replace("hc-rms-tree-unroll-v1", "qmoe-row4-static-weights-v1")
            .replace("qwen38-int4-hc-rms-tree-unroll/build-v1", "qwen38-int4-qmoe-row4-static-weights/build-v1")
            .replace("hc_rms_tree_unroll", "qmoe_row4_static_weights")
            .replace("ORT_WEBGPU_HC_RMS_TREE_UNROLL", "ORT_WEBGPU_QMOE_ROW4_STATIC_WEIGHTS")
            .replace("HCRmsTreeUnrollShader", "QmoeRow4StaticWeightsShader")
            .replace('row["levels"] == 8', 'row["words"] == 4')
            .replace("ordered_reduction_levels=8", "static_decoded_weight_vectors=8")
            .replace("unroll_decode_gain_both_modes", "prefill_ttft_gain_both_modes")
            .replace("no_supported_decode_gain", "no_supported_prefill_ttft_gain"))


def model_source(source):
    source = base(source)
    source = replace_once(source,
        '"qmoe_row4_static_weights_process_modes_and_invalid_selector_qualified"',
        '"row4_static_weights_process_modes_and_invalid_selector_qualified"')
    return replace_once(source, 'native["finite_bitwise_comparisons"] == 1214287872',
                        'native["finite_bitwise_comparisons"] == 601620480')


def screen_source(source):
    source = base(source)
    source = replace_once(source, "from qualify_row4_cpu_locality import compare_samples",
                          "from compare_qmoe_prefill_samples import compare_prefill_samples as compare_samples")
    source = replace_once(source, "    sources = {str(Path(__file__)): sha(Path(__file__))}",
        '''    sources = {str(Path(__file__)): sha(Path(__file__))}
    for helper in ("compare_qmoe_prefill_samples.py", "qualify_row4_cpu_locality.py",
                   "run_row4_cpu_locality.py"):
        sources[str(ROOT / helper)] = sha(ROOT / helper)''')
    return replace_once(source, 'comparison_kind="bracketed_separate_load_original_provider_net_screen"',
                        'comparison_kind="bracketed_separate_load_original_provider_prefill_screen"')


def main():
    for kind, transform in (("model", model_source), ("screen", screen_source)):
        target = ROOT / f"run_qmoe_row4_static_weights_{kind}.py"
        with target.open("x") as stream:
            stream.write(transform((ROOT / f"run_hc_rms_tree_unroll_{kind}.py").read_text()))
        print(target.name)


if __name__ == "__main__":
    main()
