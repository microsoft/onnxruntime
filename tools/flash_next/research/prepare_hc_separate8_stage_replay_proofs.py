from pathlib import Path

from build_hc_separate8_stage_replay import FLAG
from build_qmoe_node_replay import replace_once
from prepare_hc_separate8_proofs import benchmark_command as parent_command, controller_source as parent_controller
from prepare_webgpu_int4_metrics import BENCHMARK as METRICS_BENCHMARK


ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark_webgpu_int4_hc_separate8_stage_replay.py"


def adapt(source):
    return source.replace("hc_separate8", "hc_separate8_stage_replay").replace(
        "hc-separate8", "hc-separate8-stage-replay").replace("ORT_WEBGPU_HC_SEPARATE8", FLAG)


def benchmark_command(command):
    command = parent_command(command)
    command[-1] = command[-1].replace("benchmark_webgpu_int4_hc_separate8.py", BENCHMARK.name)
    return command


def controller_source(kind):
    source = adapt(parent_controller(kind))
    if kind == "native":
        source = replace_once(source, 'and build["no_object_layout_changes"]',
            'and not build["no_object_layout_changes"] and build["private_leaf_kernel_layout_changed"]\n'
            '            and build["no_runtime_object_layout_changes"]')
        source = replace_once(source, "if (len(rows) != 12", "if (len(rows) != (24 if selected else 12)")
        source = replace_once(source,
            'or sum(row["operator"] == "HCPostNormProjectionF16" for row in rows) != 6',
            'or (not selected and sum(row["operator"] == "HCPostNormProjectionF16" for row in rows) != 6)')
        source = replace_once(source,
            'or sum(row["operator"] == "HCSharedPostNormProjectionF16" for row in rows) != 6',
            'or (not selected and sum(row["operator"] == "HCSharedPostNormProjectionF16" for row in rows) != 6)')
        source = replace_once(source, "        replay.append(",
            '        if selected and any(sum(row["operator"] == name for row in rows) != 6 for name in (\n'
            '                "HCSeparate8Norm", "HCSharedSeparate8Norm",\n'
            '                "HCSeparate8Projection", "HCSharedSeparate8Projection")):\n'
            '            raise RuntimeError("Required normalization/projection replay instances not observed")\n'
            "        replay.append(")
    if kind == "model":
        source = replace_once(source, 'row["instances"] == 12 and row["binding_hits"] > 0',
                              'row["instances"] == (24 if row["phase"] == "candidate" else 12)\n'
                              '               and row["binding_hits"] > 0')
    return source


if __name__ == "__main__":
    benchmark = replace_once(METRICS_BENCHMARK.read_text(),
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n',
        '                "ORT_WEBGPU_QMOE_GROUPED_PREFILL",\n' + f'                "{FLAG}",\n')
    outputs = [(BENCHMARK, benchmark)]
    outputs.extend((ROOT / f"run_hc_separate8_stage_replay_{kind}.py", controller_source(kind))
                   for kind in ("native", "invalid", "model", "cpu_local8_standard", "cpu_local8_screen"))
    outputs.append((ROOT / "run_hc_separate8_stage_replay_progression.py",
                    adapt((ROOT / "run_hc_separate8_progression.py").read_text())))
    for path, source in outputs:
        with path.open("x") as stream:
            stream.write(source)
        print(path.name)
