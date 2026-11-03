import json
from pathlib import Path

from build_qmoe_node_replay import replace_once, sha


ROOT = Path(__file__).parent


def build_source(source):
    begin = source.index("def make_program(source):")
    end = source.index("def main():", begin)
    source = source[:begin] + source[end:]
    source = replace_once(source, "from build_hc_owned_vector import make_program as vector_program",
                          "from hc_gate_staging_transform import transform")
    source = source.replace("OWNED_PAIR34", "GATE_STAGING").replace("OwnedPair34", "GateStaging")
    source = source.replace("owned-pair34", "gate-staging")
    source = replace_once(source, '"single_owner_pair_stores": True,', '"gates_computed_once_per_workgroup": True,')
    source = replace_once(source, '"padded_block_half_elements": 34,', '"padded_block_half_elements": 36,')
    source = replace_once(source, '"padded_block_u32_words": 17,', '"padded_block_u32_words": 18,')
    source = replace_once(source, '"shared_storage_bytes": 26368,', '"shared_storage_bytes": 27668,')
    return replace_once(source, "inputs = [Path(__file__),",
        'inputs = [Path(__file__), Path(__file__).with_name("hc_gate_staging_transform.py"),\n'
        '              Path(__file__).with_name("prepare_hc_gate_staging.py"),\n'
        '              Path(__file__).with_name("build_hc_owned_pair34.py"),')


def controller_source(source):
    source = source.replace("OWNED_PAIR34", "GATE_STAGING").replace("OwnedPair34", "GateStaging")
    source = source.replace("owned-pair34", "gate-staging").replace("owned_pair34", "gate_staging")
    return replace_once(source, "evidence = [Path(__file__), GUARD,",
        'evidence = [Path(__file__), ROOT / "prepare_hc_gate_staging.py",\n'
        '                ROOT / "run_hc_owned_pair34_v1.py", GUARD,')


def main():
    inputs = [ROOT / "build_hc_owned_pair34.py", ROOT / "run_hc_owned_pair34_v1.py"]
    outputs = [ROOT / "build_hc_gate_staging_v1.py", ROOT / "run_hc_gate_staging_v1.py"]
    report = ROOT / "hc-gate-staging-v1-generation.json"
    if report.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh gate-staging generation prefix required")
    for original, output, transform in zip(inputs, outputs, (build_source, controller_source), strict=True):
        output.write_text(transform(original.read_text()))
    with report.open("x") as stream:
        json.dump({"status": "generated_not_qualified",
                   "source_sha256": {str(path): sha(path) for path in (Path(__file__), *inputs, *outputs)}},
                  stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
