from pathlib import Path

from build_dense_packed4_bindings import build_bindings, transform as bank_transform
from build_qmoe_node_replay import replace_once


def transform(source):
    source = bank_transform(source)
    anchor = '        flag("0");'
    source = replace_once(source, anchor, '''
        auto& candidate_bank = gpu_inputs[(round + 1) % gpu_inputs.size()];
        auto candidate_copy = env.CopyTensors(inputs, candidate_bank, nullptr);
        if (!candidate_copy.IsOK()) throw std::runtime_error(candidate_copy.GetErrorMessage());
''' + anchor)
    source = replace_once(source,
        "auto actual = session.Run(Ort::RunOptions{}, names.data(), bank.data(), bank.size(),",
        "auto actual = session.Run(Ort::RunOptions{}, names.data(), candidate_bank.data(), candidate_bank.size(),")
    return source


def main():
    parent = Path("/workspace/qwen38-int4-dense-recent-binding/build-v1")
    build_bindings(parent, parent.parent / "binding-proof-v1", transform)


if __name__ == "__main__":
    main()
