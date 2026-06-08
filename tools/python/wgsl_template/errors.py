# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exception types for the WGSL template engine."""

from __future__ import annotations


class WgslTemplateError(Exception):
    """Base class for all WGSL template errors."""

    def __init__(
        self,
        message: str,
        kind: str = "",
        *,
        file_path: str | None = None,
        line_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.file_path = file_path
        self.line_number = line_number


class WgslTemplateLoadError(WgslTemplateError):
    """Raised by the loader (PASS0)."""


class WgslTemplateParseError(WgslTemplateError):
    """Raised by the parser (PASS1)."""


class WgslTemplateGenerateError(WgslTemplateError):
    """Raised by the generator (PASS2)."""


class WgslTemplateBuildError(WgslTemplateError):
    """Raised by the top-level build orchestrator or code generators."""
