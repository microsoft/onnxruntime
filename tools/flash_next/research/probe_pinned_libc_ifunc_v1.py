import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path("/evidence/webgpu-perf-oct20")
spec = json.loads((ROOT / "cpu-libc-ifunc-v1-spec.json").read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
        ("symbol", ctypes.c_char_p), ("symbol_address", ctypes.c_void_p),
    ]


libc = ctypes.CDLL("libc.so.6")
dladdr = ctypes.CDLL(None).dladdr
dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
dladdr.restype = ctypes.c_int
selected = []
for name in spec["ifunc_names"]:
    pointer = ctypes.cast(getattr(libc, name), ctypes.c_void_p).value
    info = DlInfo()
    if not pointer or dladdr(pointer, ctypes.byref(info)) != 1:
        raise RuntimeError(f"No loaded-DSO identity for {name}")
    filename = info.filename.decode()
    if filename.endswith("/libc.so.6"):
        if sha(Path(filename)) != spec["libc_sha256"]:
            raise RuntimeError("CPU helper loaded a different libc")
        selected.append({"name": name, "elf_address": pointer - info.base,
                         "dso": filename, "inside_verified_libc": True})
    else:
        selected.append({"name": name, "dso": filename,
                         "inside_verified_libc": False})

python = Path("/opt/conda/bin/python3.11")
destination = ROOT / "cpu-decode-symbol-root-v4/opt/conda/bin/python3.11"
if not destination.resolve().is_relative_to(
        (ROOT / "cpu-decode-symbol-root-v4").resolve()):
    raise RuntimeError("Python symbol destination escapes the fresh root")
notes = subprocess.run(["readelf", "-n", str(python)], check=True,
                       capture_output=True, text=True).stdout
build_id = re.search(r"Build ID: ([0-9a-f]+)", notes)
if build_id is None:
    raise RuntimeError("Pinned Python has no build ID")
digest = sha(python)
destination.parent.mkdir(parents=True, exist_ok=True)
with python.open("rb") as source, destination.open("xb") as target:
    shutil.copyfileobj(source, target, 1024 * 1024)
if sha(destination) != digest:
    raise RuntimeError("Copied Python differs from pinned image")
result = {
    "libc_sha256": spec["libc_sha256"],
    "selected_ifuncs": selected,
    "cpu_affinity": sorted(os.sched_getaffinity(0)),
    "glibc_tunables": os.environ.get("GLIBC_TUNABLES"),
    "python_sha256": digest,
    "python_build_id": build_id[1],
    "python_bytes": python.stat().st_size,
}
with (ROOT / "cpu-libc-ifunc-v1-probe.json").open("x") as stream:
    json.dump(result, stream, indent=2)
    stream.write("\n")
print("PASS verified libc IFUNC resolution and pinned Python symbol copy")
