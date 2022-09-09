# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import TextIO, List


class SourceWriter:
    _writer: TextIO
    _indent_str: str
    _indent_depth: int
    _needs_indent: bool
    _namespaces: List[str]

    def __init__(self, base_writer: TextIO, indent_str: str = "  "):
        self._writer = base_writer
        self._indent_str = indent_str
        self._indent_depth = 0
        self._needs_indent = False
        self._namespaces = []

    def __enter__(self):
        self._writer.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._writer.__exit__(exc_type, exc_val, exc_tb)

    def write(self, str: str):
        if not str or len(str) <= 0:
            return

        for c in str:
            if self._needs_indent:
                self._needs_indent = False
                if self._indent_depth > 0:
                    self._writer.write(self._indent_str * self._indent_depth)

            if c == "\n":
                self._needs_indent = True

            self._writer.write(c)

    def writeline(self, str: str = None):
        if str:
            self.write(str)
        self.write("\n")

    def push_indent(self):
        self._indent_depth += 1

    def pop_indent(self):
        self._indent_depth -= 1

    def push_namespace(self, namespace: str):
        self._namespaces.append(namespace)
        self.writeline(f"namespace {namespace} {{")

    def pop_namespace(self):
        self.writeline(f"}} // namespace {self._namespaces.pop()}")

    def pop_namespaces(self):
        while len(self._namespaces) > 0:
            self.pop_namespace()
