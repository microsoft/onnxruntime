#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Freeze SVE intrinsics translation units into KleidiAI-style machine code.

Compiles an SVE intrinsics reference TU with gcc, extracts the requested
extern "C" functions from the object file, and emits a portable .S file in
which every instruction is a raw word (KAI_ASM_INST -> GAS ".inst" /
armasm64 "DCD" via aarch64/kai_asm_macros.h). The output assembles with
toolchains that have no SVE support (e.g. MSVC armasm64) and the bytes are
identical on every platform.

Safety checks (the generator aborts if any fails):
  * no relocations against any frozen function's .text.* section
    (compile with -fno-stack-protector; .eh_frame relocs are ignored --
    unwind info is not carried into the frozen file);
  * no adrp/adr (page-relative addressing of data), no bl/blr (calls),
    and no PC-relative literal-pool loads -- every input must arrive via
    the argument registers/stack so the code is position-independent;
  * no platform-reserved registers (RESERVED_REGS): x18 is the AArch64
    platform register -- Windows ARM64 reserves it for the TEB, Darwin
    reserves it, and Linux shadow-call-stack builds use it. Frozen bytes
    ship on every platform, so code that allocates x18 corrupts the TEB on
    Windows (observed: 0xC0000005 writing to 0x250 from inside ntdll).
    A save/restore wrapper is NOT a fix -- the OS may rewrite x18 at any
    context switch. -ffixed-x18 is applied by default (see DEFAULT_CFLAGS);
  * the section contains only 4-byte instruction words (no embedded data).

Example (QGEMM svmmla kernels):
  python3 gen_sve_asm.py \
      --src sve/qgemm_mmla_sve_impl.cpp \
      --out aarch64/qgemm_mmla_sve_asm.S \
      --march armv8.2-a+sve+i8mm \
      --symbols MlasGemmS8S8KernelSmmlaSveImpl,MlasGemmU8X8KernelUmmlaSveImpl \
      --module qgemm_mmla_sve
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]{8})\s+(.*)$")
FORBIDDEN_MNEMONIC_RE = re.compile(r"^(adrp|adr|bl|blr)\s")
LITERAL_LOAD_RE = re.compile(r"^ldr\s+[^,]+,\s+0x[0-9a-f]+\s*$")

# Registers no frozen function may touch, and the compiler flag that keeps the
# compiler off them. Extend both together if another platform reserves more.
RESERVED_REGS = ("x18",)
RESERVED_REG_RE = re.compile(r"\b[wx]18\b")
DEFAULT_CFLAGS = ("-ffixed-x18",)


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def check_relocations(obj, symbols):
    out = run(["readelf", "-r", obj])
    section = None
    for line in out.splitlines():
        m = re.match(r"Relocation section '(\S+)'", line)
        if m:
            section = m.group(1)
            continue
        if section and any(f".text.{s}" in section for s in symbols):
            if re.match(r"^[0-9a-f]{6,}", line.strip()):
                sys.exit(f"reloc in frozen section {section}: {line.strip()}")


def extract_functions(obj, symbols, allow_missing=False):
    # Parsing is driven by sections (-ffunction-sections puts each frozen
    # function in its own .text.<name>), NOT by symbol header lines: gcc
    # emits local labels inside a function (e.g. the .SVLPSPL/.SVLPEND SVE
    # stack-probe loop), which objdump renders as symbol headers that must
    # not terminate extraction.
    out = run(["objdump", "-d", obj])
    functions = {}
    current = None
    for line in out.splitlines():
        m = re.match(r"^Disassembly of section \.text\.(\S+):$", line)
        if m:
            current = m.group(1) if m.group(1) in symbols else None
            if current:
                functions[current] = []
            continue
        if current is None:
            continue
        if not line.strip():
            continue
        if re.match(r"^[0-9a-f]+ <[^>]+>:$", line):
            continue
        m = INSN_RE.match(line)
        if not m:
            sys.exit(f"non-instruction line in {current}: {line!r}")
        _, word, disasm = m.groups()
        disasm = disasm.split("//")[0].rstrip()
        if FORBIDDEN_MNEMONIC_RE.match(disasm):
            sys.exit(f"forbidden instruction in {current}: {disasm}")
        if LITERAL_LOAD_RE.match(disasm):
            sys.exit(f"literal-pool load in {current}: {disasm}")
        if RESERVED_REG_RE.search(disasm):
            sys.exit(
                f"platform-reserved register in {current}: {disasm}\n"
                f"  frozen bytes ship on every platform and {'/'.join(RESERVED_REGS)} is "
                f"reserved (Windows TEB, Darwin, Linux shadow-call-stack).\n"
                f"  compile with {' '.join(DEFAULT_CFLAGS)} (applied by default; do not "
                f"override it away)."
            )
        functions[current].append((word, disasm))
    missing = [s for s in symbols if s not in functions]
    if missing and not allow_missing:
        sys.exit(f"symbols not found in object: {missing}")
    return functions


HEADER = """\
/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    {module}_asm.S

Abstract:

    Portable machine-code variant of the SVE {module} kernels, in the style
    of Arm's KleidiAI library: every instruction is emitted as a raw
    instruction word (GAS ".inst" / armasm64 "DCD" via KAI_ASM_INST), so the
    file assembles with toolchains that have no SVE support, and the bytes
    are identical on every platform.

    GENERATED FILE - DO NOT EDIT BY HAND.

    Generated from the SVE intrinsics translation unit(s) ({src}), which
    remain the reference implementation and regeneration source (script:
    sve/gen_sve_asm.py). The functions here export the same extern "C"
    symbols; the build links exactly one of the two implementations. The
    code is fully position-independent (the generator verifies there are no
    relocations, calls, or literal pools).

--*/

#include "kai_asm_macros.h"

    KAI_ASM_CODE({module})
"""


def emit(functions, symbols, module, src, out_path):
    lines = [HEADER.format(module=module, src=src)]
    for sym in symbols:
        lines.append("")
        lines.append("    KAI_ASM_ALIGN")
        lines.append(f"    KAI_ASM_GLOBAL({sym})")
        lines.append(f"    KAI_ASM_FUNCTION_TYPE({sym})")
        lines.append(f"KAI_ASM_FUNCTION_LABEL({sym})")
        lines.append("    KAI_ASM_BTI_C")
        for word, disasm in functions[sym]:
            lines.append(f"    KAI_ASM_INST(0x{word})  // {disasm}")
        lines.append(f"    KAI_ASM_FUNCTION_END({sym})")
    lines.append("")
    lines.append("    KAI_ASM_END")
    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src", required=True, action="append", help="intrinsics TU; repeatable, objects are scanned in order"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--march", required=True)
    # Default matches how ORT actually builds these sources (RelWithDebInfo is
    # -O2). Freezing at -O3 measured 7-8% SLOWER at M == 1 on X925 -- the
    # single-row path has too little work to absorb -O3's extra unrolling --
    # while being flat at M > 1, so -O2 is both the consistent and the faster
    # choice. Re-measure before changing it.
    ap.add_argument("--opt", default="O2", help="optimization level (O2/O3)")
    ap.add_argument("--include", action="append", default=[], help="extra -I directory; repeatable")
    ap.add_argument("--define", action="append", default=[])
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--cxx", default="g++")
    ap.add_argument(
        "--cflag",
        action="append",
        default=[],
        help="extra compiler flag; repeatable. Added on top of DEFAULT_CFLAGS",
    )
    ap.add_argument(
        "--no-default-cflags",
        action="store_true",
        help=f"drop DEFAULT_CFLAGS ({' '.join(DEFAULT_CFLAGS)}). For testing the "
        "reserved-register check only -- a real regeneration without them will abort.",
    )
    args = ap.parse_args()

    symbols = args.symbols.split(",")

    functions = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, src in enumerate(args.src):
            obj = os.path.join(tmp, f"frozen{i}.o")
            cmd = [
                args.cxx,
                "-std=c++17",
                f"-{args.opt}",
                f"-march={args.march}",
                "-fno-stack-protector",
                "-ffunction-sections",
                f"-I{os.path.dirname(os.path.abspath(src))}",
            ]
            cmd += [f"-I{d}" for d in args.include]
            cmd += [f"-D{d}" for d in args.define]
            if not args.no_default_cflags:
                cmd += list(DEFAULT_CFLAGS)
            cmd += args.cflag
            cmd += ["-c", "-o", obj, src]
            run(cmd)
            wanted = [s for s in symbols if s not in functions]
            check_relocations(obj, wanted)
            functions.update(extract_functions(obj, wanted, allow_missing=True))
    missing = [s for s in symbols if s not in functions]
    if missing:
        sys.exit(f"symbols not found in any object: {missing}")
    emit(functions, symbols, args.module, ", ".join(args.src), args.out)

    total = sum(len(v) for v in functions.values())
    print(f"wrote {args.out}: {len(symbols)} functions, {total} instructions")


if __name__ == "__main__":
    main()
