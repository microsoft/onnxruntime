import inspect
from pathlib import Path

from build_qmoe_node_replay import replace_once
from run_dense_columns16_synthetic import run_guard


ROOT = Path(__file__).parent


def prepare():
    source = (ROOT / "run_hc_owned_vector_native.py").read_text()
    source = source.replace("build_hc_owned_vector", "build_hc_parallel16")
    source = source.replace("hc-owned-vector", "hc-parallel16")
    source = source.replace("-gpu7", "-gpu4")
    source = replace_once(source,
        "hc-post-norm-projection-v2-native-gpu4-20261029.qualification.json",
        "hc-post-norm-projection-v2-native-gpu7-20261029.qualification.json")
    source = source.replace("hc-parallel16-v1-native-", "hc-parallel16-v1-native-attempt3-")
    source = replace_once(source, 'DATA = BUILD.parent / "original-references-v1"',
                          'DATA = BUILD.parent / "original-references-attempt3"')
    source = replace_once(source,
        "from run_dense_columns16_synthetic import run_guard, UUID, workspace",
        "from run_dense_columns16_synthetic import workspace\n\n"
        'UUID = "GPU-68170cb3-c9ea-a588-7558-ae82635d9173"\n\n' +
        inspect.getsource(run_guard).replace('"--gpu-index", "7"', '"--gpu-index", "4"'))
    source = replace_once(source, "evidence = [Path(__file__), build_path, comparator_path]",
        'evidence = [Path(__file__), ROOT / "prepare_hc_parallel16_native.py",\n'
        '                ROOT / "build_hc_parallel16.py", ROOT / "test_hc_parallel16.py",\n'
        '                ROOT / "run_dense_columns16_synthetic.py",\n'
        '                ROOT / "shared_activation_guard.py", ROOT / "nvml_monitor.py",\n'
        '                build_path, comparator_path]')
    assert "hc-owned-vector" not in source
    assert source.count('"--gpu-index", "4"') == 1
    output = ROOT / "run_hc_parallel16_native.py"
    with output.open("x") as stream:
        stream.write(source)
    return output


if __name__ == "__main__":
    print(prepare())
