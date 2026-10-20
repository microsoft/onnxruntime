import argparse
import re
import shutil
from pathlib import Path

_PRIVATE_INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*["<](src/[^">]+)[">]')

_PRIVATE_INCLUDES = frozenset(
    {
        '#include "src/dawn/common/Compiler.h"',
        '#include "src/utils/assert.h"',
        '#include "src/utils/log.h"',
    }
)

_STANDALONE_SUPPORT = """\
#include <cstdlib>
#include <iostream>
#if defined(__has_attribute)
#if __has_attribute(no_sanitize)
#define DAWN_NO_SANITIZE(value) __attribute__((no_sanitize(value)))
#else
#define DAWN_NO_SANITIZE(value)
#endif
#else
#define DAWN_NO_SANITIZE(value)
#endif
#define DAWN_CHECK(condition) do { if (!(condition)) { std::abort(); } } while (0)
namespace dawn {
inline std::ostream& ErrorLog() { return std::cerr; }
}
"""


def standalone_proc(source: str) -> str:
    lines = source.splitlines(keepends=True)
    found = {
        f'#include "{match.group(1)}"' for line in lines if (match := _PRIVATE_INCLUDE_PATTERN.match(line)) is not None
    }
    if found != _PRIVATE_INCLUDES:
        raise ValueError(
            f"Unexpected Dawn proc includes; missing: {sorted(_PRIVATE_INCLUDES - found)}; "
            f"unexpected: {sorted(found - _PRIVATE_INCLUDES)}"
        )
    result = []
    for line in lines:
        if _PRIVATE_INCLUDE_PATTERN.match(line):
            if found:
                result.append(_STANDALONE_SUPPORT)
                found.clear()
        else:
            result.append(line)
    return "".join(result)


def export_package(dawn_source: Path, dawn_build: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("The package output directory must be empty")
    for include_root in (dawn_source / "include", dawn_build / "gen/include"):
        if not include_root.is_dir():
            raise ValueError(f"Dawn include directory not found: {include_root}")
        for header in include_root.rglob("*.h"):
            destination = output / "include" / header.relative_to(include_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(header, destination)
    source_root = dawn_build / "gen/src/dawn"
    proc = standalone_proc((source_root / "dawn_proc.cpp").read_text(encoding="utf-8"))
    (output / "src").mkdir(parents=True, exist_ok=True)
    (output / "src/dawn_proc.cpp").write_text(proc, encoding="utf-8")
    shutil.copyfile(source_root / "dawn_thread_dispatch_proc.cpp", output / "src/dawn_thread_dispatch_proc.cpp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a standalone Dawn API package for external-host CI")
    parser.add_argument("--dawn-source", type=Path, required=True)
    parser.add_argument("--dawn-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    export_package(arguments.dawn_source, arguments.dawn_build, arguments.output)


if __name__ == "__main__":
    main()
