# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""WGSL template engine.

The public entry point is :func:`build`, which orchestrates the
load -> parse -> generate -> emit pipeline.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from .code_generator import resolve_code_generator
from .errors import (
    WgslTemplateBuildError,
    WgslTemplateError,
    WgslTemplateGenerateError,
    WgslTemplateLoadError,
    WgslTemplateParseError,
)
from .generator import generate_directory
from .loader import load_from_directories
from .parser import parse
from .types import SourceDir


def build(
    *,
    source_dirs: Sequence[str | os.PathLike[str] | SourceDir],
    out_dir: str | os.PathLike[str],
    template_ext: str = ".wgsl.template",
    generator: str = "static-cpp-literal",
    include_path_prefix: str = "",
    preserve_code_reference: bool = False,
    clean: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Run the full build pipeline: load → parse → generate → emit.

    Returns the list of relative output paths that were inspected (one
    per emitted file). Files whose contents are unchanged are not
    rewritten, which keeps CMake's mtime-based incremental rebuild
    tracking honest.
    """

    if not source_dirs:
        raise WgslTemplateBuildError(
            "source_dirs must be provided and cannot be empty",
            "invalid-options",
        )

    out_path = Path(out_dir).resolve()

    # --clean wipes the output directory before building.
    if clean and out_path.exists():
        if verbose:
            print(f"Cleaning output directory: {out_path}")
        if out_path.is_file():
            raise WgslTemplateBuildError(
                f'Output path "{out_path}" is an existing file; expected a directory',
                "invalid-options",
            )
        shutil.rmtree(out_path)

    if verbose:
        print("Building WGSL templates...")
        print(f"  Sources: {list(source_dirs)}")
        print(f"  Output: {out_path}")
        print(f"  Generator: {generator}")
        print(f"  Template extension: {template_ext}")
        if include_path_prefix:
            print(f"  Include prefix: {include_path_prefix}")
        if preserve_code_reference:
            print("  Preserve code references: enabled")

    # PASS0 - load.
    pass0 = load_from_directories(source_dirs, ext=template_ext)
    # PASS1 - parse.
    pass1 = parse(pass0)
    # PASS2 - generate.
    code_generator = resolve_code_generator(generator)
    pass2 = generate_directory(
        pass1,
        code_generator,
        preserve_code_reference=preserve_code_reference,
    )
    # Build (per-template + index*.h + string_table.h).
    files = code_generator.build(
        pass2,
        template_ext=template_ext,
        include_path_prefix=include_path_prefix,
    )

    # Write files. Idempotent: only overwrite when content differs.
    out_path.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for rel_path, content in files.templates.items():
        assert isinstance(content, str)
        full_path = (out_path / rel_path).resolve()
        # Security: refuse to write outside the output directory.
        try:
            full_path.relative_to(out_path)
        except ValueError as e:
            raise WgslTemplateBuildError(
                f"Security violation: attempted to write file outside output directory: {rel_path}",
                "path-security-violation",
                file_path=str(rel_path),
            ) from e

        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Idempotent write: only rewrite when content differs. This
        # preserves mtime so CMake's incremental builds stay clean.
        # Force LF line endings unconditionally for stable output
        # across Windows / Linux.
        normalized = content.replace("\r\n", "\n")
        new_bytes = normalized.encode("utf-8")
        existing_bytes: bytes | None = None
        if full_path.exists():
            try:
                existing_bytes = full_path.read_bytes()
            except OSError:
                existing_bytes = None
        if existing_bytes != new_bytes:
            full_path.write_bytes(new_bytes)
        written.append(rel_path)

    if verbose:
        print(f"Build completed successfully! ({len(written)} file(s))")

    return written


__all__ = [
    "SourceDir",
    "WgslTemplateBuildError",
    "WgslTemplateError",
    "WgslTemplateGenerateError",
    "WgslTemplateLoadError",
    "WgslTemplateParseError",
    "build",
]
