from prepare_adapter_inline_proofs import ROOT, base, model_source, native_source


def capacity17(source):
    return (source.replace("adapter-inline-tensors-v2", "adapter-inline-tensors-cap17-v3")
            .replace("qwen38-int4-adapter-inline-tensors/build-v2",
                     "qwen38-int4-adapter-inline-tensors/build-v3")
            .replace("build_adapter_inline_tensors", "build_adapter_inline_capacity17")
            .replace("run_adapter_inline", "run_adapter_inline_capacity17")
            .replace('row["input_capacity"] == 8', 'row["input_capacity"] == 17')
            .replace("input_inline_capacity=8", "input_inline_capacity=17")
            .replace("input_capacity=8", "input_capacity=17")
            .replace('        assert vector_coverage["contexts"] >= result["dense_replay_hits"]',
                     '''        assert vector_coverage["contexts"] >= result["dense_replay_hits"]
        assert vector_coverage["input_heap_fallbacks"] == 0
        assert vector_coverage["output_heap_fallbacks"] == 0'''))


def main():
    for old, new, transform in (
        ("run_adapter_tensor_reuse_native.py", "run_adapter_inline_capacity17_native.py", native_source),
        ("run_adapter_tensor_reuse_model.py", "run_adapter_inline_capacity17_model.py", model_source),
        ("run_adapter_tensor_reuse_screen.py", "run_adapter_inline_capacity17_screen.py", base)):
        with (ROOT / new).open("x") as stream:
            stream.write(capacity17(transform((ROOT / old).read_text())))
        print(new)


if __name__ == "__main__":
    main()
