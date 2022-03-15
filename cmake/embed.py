#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This file generate a C++ header file which holds the blob of another file.

There is a long story about embeding another file in C++ source code. See
http://open-std.org/JTC1/SC22/WG21/docs/papers/2020/p1040r6.html for more
information. But non of the mentioned tools or methods are cross-platform or
standardized.

The xxd command on Unix is a simple tool that can process a file into a header
file. This script implement `xxd -i` funcationality with additional features,
for example, #include directive expansion for C-like language. This tool is
added to help embeding OpenCL kernal source into C++ source via include"""

import io
import os
import sys
import pip
import subprocess
import hashlib


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    elif hasattr(pip, '_internal') and hasattr(pip._internal, 'main'):
        pip._internal.main(['install', package])
    else:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'package'])


for i in range(2):
    try:
        from pcpp import Preprocessor, OutputDirective, Action
        break
    except ImportError:
        print("pip install pcpp")
        install('pcpp')

template = """
#ifndef CONTENT_NAME
#define CONTENT_NAME kernel_src
#endif
#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

unsigned char CONCAT(__embed_uchar_, CONTENT_NAME)[{array_length}] = {{
{content}
}};
char* CONTENT_NAME = (char*)CONCAT(__embed_uchar_, CONTENT_NAME);
[[maybe_unused]] unsigned int CONCAT(CONTENT_NAME, _len) = {length};

#undef CONTENT_NAME
"""


def batch_iter(iter, batch_size=12):
    n = 0
    batch = []
    for i in iter:
        batch.append(i)
        n += 1
        if n == batch_size:
            yield batch
            n = 0
            batch = []
    if batch:
        yield batch


def compute_path(path):
    """compute path relative to onnxruntime project root. Assume that this file
    is in <project_root>/cmake/"""
    # this file <project_root>/cmake/embed.py
    project_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.relpath(path, project_root)


class OpenCLPreprocessor(Preprocessor):
    """"A preprocessor that expand #include directive."""
    def __init__(self, includes):
        super().__init__()
        for i in includes:
            self.add_path(i)

        self.line_directive = None
        # this will compress consecutive whitespaces into one, it will result
        # much better opencl debug experence, and smaller executable
        self.compress = 2

    def on_error(self, file, line, msg):
        super().on_error(file, line, msg)
        if (self.return_code != 0):
            sys.exit(self.return_code)

    def on_comment(self, tok):
        return ""

    def on_include_not_found(self, is_malformed, is_system_include, curdir,
                             includepath):
        if is_malformed:
            self.on_error(self.lastdirective.source, self.lastdirective.lineno,
                          "Malformed #include statement: %s" % includepath)
        else:
            self.on_error(self.lastdirective.source, self.lastdirective.lineno,
                          "Include file '%s' not found" % includepath)
        print(self.return_code)

    def on_directive_handle(self, directive, toks, ifpassthru, precedingtoks):
        """only expand include"""
        if directive.value == "include" or directive.value == "pragma":
            self.lastdirective = directive
            return True

        raise OutputDirective(Action.IgnoreAndPassThrough)

    def get_dependencies(self):
        dedup = set()
        inc = [it.included_abspath for it in self.include_times]
        inc = [i for i in inc if not (i in dedup or dedup.add(i))]
        return inc


def get_sha1sum(path):
    sha1 = hashlib.sha1()
    if os.path.exists(path):
        with open(path, "rb") as f:
            sha1.update(f.read())
    return sha1.hexdigest()


def write_output(path, content: str, *, write_if_changed_only=False):
    if path is None:
        print(content)
        return

    content = content.encode("utf8")
    should_write = True
    if write_if_changed_only and os.path.exists(path):
        with open(path, "rb") as f:
            old_content = f.read(len(content) + 1)
        if old_content == content:
            should_write = False
    if should_write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="this file works like `xxd -i`")
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--no_null_terminated", action="store_true")
    parser.add_argument("file", type=os.path.realpath, help="input file")
    parser.add_argument("-x",
                        default=None,
                        choices=["cl"],
                        help="specify how to treat the input file")
    parser.add_argument("-I", action="append", default=None)
    parser.add_argument("-M", action="store_true")
    parser.add_argument("--output", "-o", default=None, type=os.path.realpath)

    args = parser.parse_args()
    # print("pre-processing", args.file)

    file_content = open(args.file, "rb").read()

    # handling opencl preprocessing
    if args.x == "cl":
        clpp = OpenCLPreprocessor(args.I)
        clpp.parse(file_content.decode("utf8"), os.path.realpath(args.file))
        out = io.StringIO()
        clpp.write(out)
        if args.M:
            deps = clpp.get_dependencies()
            sha1_and_deps = map(lambda path: get_sha1sum(path) + " " + path, deps)
            write_output(args.output, "\n".join(sha1_and_deps), write_if_changed_only=True)
            exit(0)

        file_content = out.getvalue()
        file_content_cl = file_content
        file_content = file_content.encode("utf8")

    # encode file_content as an C source textual representation of an array
    file_content = [c for c in file_content]
    file_content_len = len(file_content)
    if not args.no_null_terminated:
        file_content.append(0)
    file_content = [f"0x{c:02x}" for c in file_content]
    lines = [
        ", ".join(line_content) for line_content in batch_iter(file_content)
    ]

    # the array body
    content = io.StringIO()
    if not args.no_header:
        filename_relative_to_ort = compute_path(args.file)
        content.write(f"  // generated from {filename_relative_to_ort}\n\n")
        if args.x == "cl":
            for line in file_content_cl.split("\n"):
                content.write(f"  //{line}\n")
    content.write("  ")
    content.write(",\n  ".join(lines))

    # format the template string with the array body and other meta data
    generated = template.strip().format(filename=filename_relative_to_ort,
                                        content=content.getvalue(),
                                        array_length=file_content_len +
                                        (0 if args.no_null_terminated else 1),
                                        length=file_content_len)
    write_output(args.output, generated)
