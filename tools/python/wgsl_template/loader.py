# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""PASS0: load template files from one or more source directories."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Union

from .errors import WgslTemplateLoadError
from .types import SourceDir, TemplatePass0, TemplateRepository

DEFAULT_EXT = ".wgsl.template"

_LINE_SPLIT_RE = re.compile(r"\r?\n")


SourceDirInput = Union[str, "os.PathLike[str]", SourceDir]


def _normalize_source_dir(item: SourceDirInput) -> SourceDir:
    if isinstance(item, SourceDir):
        return item
    return SourceDir(path=os.fspath(item), alias=None)


def _resolve_within(base: Path, target: Path) -> bool:
    """True iff ``target`` is the same as ``base`` or lives underneath it.

    Used as a defense against symlinks or crafted paths that would
    escape the source directory.
    """

    try:
        target.relative_to(base)
        return True
    except ValueError:
        return False


def _load_directory(
    source_dir: Path,
    base_dir: Path,
    ext: str,
    alias: str | None,
    templates: dict,
) -> None:
    """Recursively scan ``source_dir`` for files ending in ``ext``."""

    try:
        entries = sorted(os.listdir(source_dir))
    except OSError as e:
        raise WgslTemplateLoadError(
            f"Error scanning directory {source_dir}: {e}",
            "scan-directory",
        ) from e

    for entry in entries:
        full_path = source_dir / entry

        # Reject anything resolving outside base_dir.
        try:
            resolved = full_path.resolve()
        except OSError:
            # Broken symlink etc.
            continue
        if not _resolve_within(base_dir, resolved):
            continue

        # Skip symlinks and special files; only follow real directories
        # and read real files.
        if full_path.is_symlink():
            continue
        if full_path.is_dir():
            _load_directory(full_path, base_dir, ext, alias, templates)
        elif full_path.is_file() and entry.endswith(ext):
            _load_file(full_path, base_dir, alias, templates)


def _load_file(
    file_path: Path,
    base_dir: Path,
    alias: str | None,
    templates: dict,
) -> None:
    resolved_file = file_path.resolve()
    if not _resolve_within(base_dir, resolved_file):
        raise WgslTemplateLoadError(
            f"Security violation: attempted to read file outside base directory: {file_path}",
            "read-file",
            file_path=str(file_path),
        )

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as e:
        raise WgslTemplateLoadError(
            f"Error loading template file {file_path}: {e}",
            "read-file",
            file_path=str(file_path),
        ) from e

    lines = _LINE_SPLIT_RE.split(content)

    relative = file_path.relative_to(base_dir)
    # Always use POSIX-style paths in template names.
    template_name = relative.as_posix()
    if alias:
        template_name = f"{alias}/{template_name}"

    if template_name in templates:
        raise WgslTemplateLoadError(
            f"Template name conflict: {template_name} already exists",
            "template-conflict",
            file_path=str(file_path),
        )

    templates[template_name] = TemplatePass0(
        file_path=str(resolved_file),
        raw=lines,
    )


def load_from_directories(
    directories: Sequence[SourceDirInput],
    *,
    ext: str = DEFAULT_EXT,
) -> TemplateRepository:
    """Load all template files matching ``ext`` from each directory.

    Returns a :class:`TemplateRepository` whose ``templates`` map is
    sorted by template name.
    """

    if not directories:
        raise WgslTemplateLoadError(
            "At least one source directory is required",
            "scan-directory",
        )

    raw_templates: dict = {}
    resolved_base_paths: list[Path] = []

    for raw_item in directories:
        spec = _normalize_source_dir(raw_item)
        dir_path = Path(spec.path)

        # Resolve relative paths against the cwd.
        if not dir_path.is_absolute():
            dir_path = Path.cwd() / dir_path

        if not dir_path.exists():
            raise WgslTemplateLoadError(
                f"Cannot access directory {dir_path}: not found",
                "scan-directory",
            )
        if not dir_path.is_dir():
            raise WgslTemplateLoadError(
                f"Path {dir_path} is not a directory",
                "scan-directory",
            )

        resolved_base = dir_path.resolve()
        _load_directory(
            source_dir=resolved_base,
            base_dir=resolved_base,
            ext=ext,
            alias=spec.alias,
            templates=raw_templates,
        )
        resolved_base_paths.append(resolved_base)

    # Sort by template name for stable, host-independent ordering.
    sorted_templates = {name: raw_templates[name] for name in sorted(raw_templates)}

    # Use the first input directory as the repository's base path.
    base_path = str(resolved_base_paths[0]) if resolved_base_paths else ""

    return TemplateRepository(
        base_path=base_path,
        templates=sorted_templates,
    )


def load_from_directory(
    directory: SourceDirInput,
    *,
    ext: str = DEFAULT_EXT,
) -> TemplateRepository:
    """Convenience wrapper for the single-directory case."""
    return load_from_directories([directory], ext=ext)
