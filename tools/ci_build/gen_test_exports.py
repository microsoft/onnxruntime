# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Export only symbols consumed by a test module from its Windows host executable."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Symbols:
    undefined: set[str] = field(default_factory=set)
    defined: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)


SYMBOL = re.compile(
    r"^\s*[0-9A-F]+\s+([0-9A-F]+)\s+(UNDEF|SECT[0-9A-F]+|ABS)\s+"
    r"(.+?)\s+(External|WeakExternal)\s+\|\s+(\S+)",
    re.IGNORECASE,
)


def read_symbols(output: Iterable[str]) -> Symbols:
    symbols = Symbols()
    for line in output:
        match = SYMBOL.match(line)
        if not match:
            continue
        value, section, symbol_type, storage, name = match.groups()
        if storage == "WeakExternal" or section == "ABS":
            continue
        if section == "UNDEF" and int(value, 16) == 0:
            symbols.undefined.add(name)
        else:
            symbols.defined.add(name)
            if "()" in symbol_type:
                symbols.functions.add(name)
    if not symbols.defined and not symbols.undefined:
        raise ValueError(
            "No COFF external symbols found; /GL objects or an unexpected linker output are not supported."
        )
    return symbols


def select_exports(host: Symbols, module: Symbols) -> dict[str, bool]:
    exports = {}
    for reference in sorted(module.undefined - module.defined):
        name = reference.removeprefix("__imp_")
        if name not in host.defined or name in module.defined:
            continue
        is_data = name not in host.functions
        if is_data and reference == name:
            raise ValueError(f"Test module references host data {name} without __declspec(dllimport).")
        exports[name] = is_data
    if not exports:
        raise ValueError("No host symbols are required by the test module; refusing to create an empty import library.")
    if len(exports) > 65535:
        raise ValueError(f"{len(exports)} required exports exceed the Windows limit of 65535.")
    return exports


def write_exports(path: Path, exports: dict[str, bool]) -> None:
    lines = ["EXPORTS"]
    lines.extend(f'  "{name}"' + (" DATA" if exports[name] else "") for name in sorted(exports))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_force_includes(path: Path, exports: dict[str, bool]) -> None:
    # MSVC must extract exported symbols from static archives even when the host never references them.
    lines = [f'#pragma comment(linker, "/include:{name}")' for name in sorted(exports)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_symbols(linker: str, response_file: Path) -> Symbols:
    # CUDA object symbol dumps can be large; do not retain the dump and split lines in memory.
    with tempfile.TemporaryFile(mode="w+t") as output:
        result = subprocess.run(
            [linker, "/dump", "/nologo", "/symbols", f"@{response_file}"],
            check=False,
            stdout=output,
        )
        output.seek(0)
        if result.returncode:
            shutil.copyfileobj(output, sys.stderr)
            result.check_returncode()
        return read_symbols(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linker", required=True)
    parser.add_argument("--host", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force-include", required=True, type=Path)
    args = parser.parse_args()
    exports = select_exports(dump_symbols(args.linker, args.host), dump_symbols(args.linker, args.module))
    write_exports(args.output, exports)
    write_force_includes(args.force_include, exports)
    print(f"Exporting {len(exports)} host symbols required by the CUDA internal-test module.")


if __name__ == "__main__":
    main()
