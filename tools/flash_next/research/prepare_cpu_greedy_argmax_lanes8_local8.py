from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent
HELPER = '''
def verify_workers(raw, mode):
    assert mode == "control"
    verify_cpu_policy(raw, "local8")
    masks = raw["thread_cpu_affinity"]
    return dict(total_observed_threads=len(masks),
                individually_pinned_threads=sum(len(mask) == 1 for mask in masks.values()))

'''


def standard(source):
    source = source.replace("cpu-greedy-argmax-lanes8-v1", "cpu-greedy-argmax-lanes8-local8-v1")
    source = replace_once(source, "from run_genai_allocator_workers import OLD_RUNTIME, verify_workers",
        "from run_genai_allocator_workers import OLD_RUNTIME\n"
        "from run_row4_default_cpu_policy import verify_cpu_policy\n"
        "from run_row4_cpu_locality import placement_command")
    source = replace_once(source, "def prefix_for(", HELPER + "\ndef prefix_for(")
    source = replace_once(source, "    return command\n",
        '    command[-1] += " --inherit-worker-cpu-affinity"\n'
        '    return placement_command(command, "local8")\n')
    source = replace_once(source, 'cpu_path = ROOT / f"{VARIANT}-cpu.qualification.json"',
                          'cpu_path = ROOT / "cpu-greedy-argmax-lanes8-v1-cpu.qualification.json"')
    source = replace_once(source, 'invalid_path = ROOT / f"{VARIANT}-invalid-gpu7-attempt1.qualification.json"',
                          'invalid_path = ROOT / "cpu-greedy-argmax-lanes8-v1-invalid-gpu7-attempt1.qualification.json"')
    source = replace_once(source, "    prerequisites = []",
        '''    locality_path = ROOT / "row4-default-cpu-policy-v1-all-contexts-20261031.qualification.json"
    locality = json.loads(locality_path.read_text())
    assert locality["status"] == "repeated_default_policy_cpu_locality_supported_all_contexts"
    for name, digest in locality["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    prerequisites = []''')
    source = replace_once(source, "runtime, cpu_path, invalid_path, template_path",
                          "runtime, cpu_path, invalid_path, locality_path, template_path")
    source = replace_once(source, 'ROOT / "run_genai_allocator_workers.py", ROOT / "benchmark_webgpu_int4.py"',
        'ROOT / "run_genai_allocator_workers.py", ROOT / "benchmark_webgpu_int4.py",\n'
        '               ROOT / "prepare_cpu_greedy_argmax_lanes8_local8.py",\n'
        '               ROOT / "run_row4_default_cpu_policy.py", ROOT / "run_row4_cpu_locality.py",\n'
        '               ROOT / "cpu_local_windows.py"')
    source = source.replace("manual_affinity_override=False", "manual_affinity_override=True")
    return source.replace("model_and_arithmetic_unchanged=True,",
                          'model_and_arithmetic_unchanged=True, cpu_placement="local8",\n'
                          "        matched_cpu_and_worker_policy_in_both_modes=True,")


def screen(source):
    source = source.replace("run_cpu_greedy_argmax_lanes8_standard",
                            "run_cpu_greedy_argmax_lanes8_local8_standard")
    source = replace_once(source, "from run_genai_allocator_workers import verify_workers",
                          "from run_cpu_greedy_argmax_lanes8_local8_standard import verify_workers")
    source = source.replace('and not proof["manual_affinity_override"]',
                            'and proof["manual_affinity_override"]')
    return source.replace("manual_affinity_override=False", "manual_affinity_override=True")


if __name__ == "__main__":
    for kind, transform in (("standard", standard), ("screen", screen)):
        path = ROOT / f"run_cpu_greedy_argmax_lanes8_local8_{kind}.py"
        with path.open("x") as stream:
            stream.write(transform((ROOT / f"run_cpu_greedy_argmax_lanes8_{kind}.py").read_text()))
        print(path.name)
