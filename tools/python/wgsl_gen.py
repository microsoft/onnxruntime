#!/usr/bin/env python3
"""CLI entry point for the WGSL template generator.

Invoked from CMake during the WebGPU EP build to translate
``*.wgsl.template`` files into C++ headers.

Usage::

    python wgsl_gen.py \\
        -i <source-dir> [-i <source-dir> ...] \\
        --output <out-dir> \\
        --generator {static-cpp|static-cpp-literal} \\
        [-I <include-prefix>] \\
        [--ext .wgsl.template] \\
        [--preserve-code-ref] \\
        [--clean] \\
        [--verbose]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wgsl_gen.py",
        description="Generate C++ headers from WGSL template files.",
    )

    parser.add_argument(
        "-i",
        "--input",
        action="append",
        default=[],
        metavar="DIR",
        help="Source directory to scan for template files. May be repeated.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        metavar="DIR",
        help="Output directory for generated files.",
    )
    parser.add_argument(
        "-g",
        "--generator",
        choices=["static-cpp", "static-cpp-literal"],
        default="static-cpp-literal",
        help="Code generator to use (default: static-cpp-literal).",
    )
    parser.add_argument(
        "-I",
        "--include-prefix",
        default="",
        metavar="PREFIX",
        help='Include path prefix used when emitting #include directives in '
             'index_impl.h (e.g. "wgsl_template_gen/").',
    )
    parser.add_argument(
        "-e",
        "--ext",
        default=".wgsl.template",
        metavar="EXT",
        help="Template file extension to scan for (default: .wgsl.template).",
    )
    parser.add_argument(
        "--preserve-code-ref",
        action="store_true",
        help="Emit source line references as comments before each generated chunk.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Recursively delete the output directory before generating.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose progress information.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="wgsl_gen.py (Python port) 0.1.0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # argparse can't express "this option is required only when no
    # subcommand is given"; validate -i/--input and --output here.
    errors: List[str] = []
    if not args.input:
        errors.append("at least one -i/--input directory is required")
    if not args.output:
        errors.append("--output is required")
    if errors:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 2

    if args.verbose:
        print("wgsl_gen.py invoked with:")
        print(f"  inputs: {args.input}")
        print(f"  output: {args.output}")
        print(f"  generator: {args.generator}")
        print(f"  include-prefix: {args.include_prefix!r}")
        print(f"  ext: {args.ext}")
        print(f"  preserve-code-ref: {args.preserve_code_ref}")
        print(f"  clean: {args.clean}")

    from wgsl_template import build

    build(
        source_dirs=[Path(p) for p in args.input],
        out_dir=Path(args.output),
        template_ext=args.ext,
        generator=args.generator,
        include_path_prefix=args.include_prefix,
        preserve_code_reference=args.preserve_code_ref,
        clean=args.clean,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    # Make the sibling wgsl_template package importable when run as a script.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
