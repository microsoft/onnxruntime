import ast
from pathlib import Path

from build_qmoe_node_replay import replace_once


ROOT = Path(__file__).parent


def adapt(source):
    for before, after in (
        ("qwen38-int4-hc-narrow-rms-broadcast/build-v3", "qwen38-int4-hc-wide-rms-broadcast/build-v1"),
        ("hc-narrow-rms-broadcast-v3", "hc-wide-rms-broadcast-v1"),
        ("hc-narrow-rms-broadcast-cpu-local8-v3", "hc-wide-rms-broadcast-cpu-local8-v1"),
        ("hc_narrow_rms_broadcast", "hc_wide_rms_broadcast"),
        ("build_hc_wide_rms_broadcast_v3 import", "build_hc_wide_rms_broadcast import"),
        ("HCPostNormProjectionNarrowRmsBroadcast", "HCPostNormProjectionWideRmsBroadcast"),
    ):
        source = source.replace(before, after)
    return source


def native_source():
    source = adapt((ROOT / "run_hc_narrow_rms_broadcast_native.py").read_text())
    source = replace_once(source, 'PROOF = BUILD.parent / "native-proof-v1"',
                          'PROOF = Path("/home/sayanshaw/repos/qwen38-int4-hc-narrow-rms-broadcast/native-proof-v1")')
    return replace_once(source, 'DATA = PROOF / "original-references"',
                        'DATA = BUILD.parent / "original-references-v1"')


def model_source():
    source = adapt((ROOT / "run_hc_narrow_rms_broadcast_model.py").read_text())
    return source.replace('ROOT / "prepare_hc_wide_rms_broadcast_model.py"',
                          'ROOT / "prepare_hc_wide_rms_broadcast_proofs.py"')


def standard_source():
    return adapt((ROOT / "run_hc_narrow_rms_broadcast_cpu_local8_standard.py").read_text())


def screen_source():
    return adapt((ROOT / "run_hc_narrow_rms_broadcast_cpu_local8_screen.py").read_text())


def invalid_source():
    return adapt((ROOT / "run_hc_narrow_rms_broadcast_invalid.py").read_text())


if __name__ == "__main__":
    for name, source in (
        ("run_hc_wide_rms_broadcast_native.py", native_source()),
        ("run_hc_wide_rms_broadcast_model.py", model_source()),
        ("run_hc_wide_rms_broadcast_cpu_local8_standard.py", standard_source()),
        ("run_hc_wide_rms_broadcast_cpu_local8_screen.py", screen_source()),
        ("run_hc_wide_rms_broadcast_invalid.py", invalid_source()),
    ):
        ast.parse(source)
        with (ROOT / name).open("x") as stream:
            stream.write(source)
        print(name)
