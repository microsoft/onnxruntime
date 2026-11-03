import json
from pathlib import Path

from attribute_cpu_allocations_v2 import sha


ROOT = Path(__file__).parent


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected frozen source anchor: {old}")
    return text.replace(old, new, 1)


def main():
    helper = ROOT / "probe_pinned_libc_ifunc_v1.py"
    controller = ROOT / "attribute_cpu_libc_ifunc_v1.py"
    helper_source = helper.read_text().replace("cpu-libc-ifunc-v1", "cpu-libc-ifunc-v2")
    helper_source = helper_source.replace("symbol-root-v4", "symbol-root-v5")
    helper_source = once(
        helper_source, 'raise RuntimeError("Pinned Python has no build ID")',
        'print("NOTICE pinned Python ELF has no build ID; recording explicit absence")')
    helper_source = once(
        helper_source, '"python_build_id": build_id[1],',
        '"python_build_id": build_id[1] if build_id else None,\n'
        '    "python_build_id_status": "present" if build_id else "absent_in_pinned_image",')
    source = controller.read_text().replace("cpu-libc-ifunc-v1", "cpu-libc-ifunc-v2")
    source = source.replace("cpu-libc-ifunc-attribution-v1", "cpu-libc-ifunc-attribution-v2")
    source = source.replace("probe_pinned_libc_ifunc_v1.py", "probe_pinned_libc_ifunc_v2.py")
    source = source.replace("symbol-root-v4", "symbol-root-v5")
    source = source.replace("callchains-symbols-v4", "callchains-symbols-v5")
    source = once(
        source, 'driver_report_path, libc, new_root / "opt/conda/bin/python3.11"]',
        'driver_report_path, libc, new_root / "opt/conda/bin/python3.11",\n'
        '                ROOT / "cpu-libc-ifunc-v2-generation.json"]')
    outputs = {
        ROOT / "probe_pinned_libc_ifunc_v2.py": helper_source,
        ROOT / "attribute_cpu_libc_ifunc_v2.py": source,
    }
    manifest = ROOT / "cpu-libc-ifunc-v2-generation.json"
    if manifest.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Fresh generated offline-probe variants required")
    for path, text in outputs.items():
        compile(text, str(path), "exec")
    for path, text in outputs.items():
        with path.open("x") as stream:
            stream.write(text)
    with manifest.open("x") as stream:
        json.dump({
            "reason": "Conda Python ELF demonstrably has no build ID; preserve SHA provenance and explicitly report absent build ID.",
            "failed_attempt_preserved": "cpu-libc-ifunc-attribution-v1-run.log",
            "parent_source_sha256": {str(path): sha(path) for path in
                                     (Path(__file__), helper, controller)},
            "generated_sha256": {str(path): sha(path) for path in outputs},
        }, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
