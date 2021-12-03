#!/usr/bin/env python

"""This file implement `xxd -i` funcationality with additional features."""

import io
import os
import shutil
import pcpp


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

# this file <REL_DIR>/cmake/embed.py
REL_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def compute_path(path):
    return os.path.relpath(path, REL_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="this file works like `xxd -i`")
    parser.add_argument("--no_header", action="store_true")
    parser.add_argument("--no_null_terminated", action="store_true")
    parser.add_argument("file", type=os.path.realpath, help="input file")
    parser.add_argument("--output", "-o", default=None, type=os.path.realpath)

    args = parser.parse_args()
    with open(args.file, "rb") as f:
        filename = compute_path(args.file)
        raw_content = [c for c in f.read()]
        raw_content_len = len(raw_content)
        if not args.no_null_terminated:
            raw_content.append(0)
        raw_content = [f"0x{c:02x}" for c in raw_content]
        lines = [", ".join(line_content) for line_content in batch_iter(raw_content)]
        content = io.StringIO()
        if not args.no_header:
            header = f"// generated from {filename}\n\n"
            content.write("  ")
            content.write(header)

        content.write("  ")
        content.write(",\n  ".join(lines))

        generated = template.strip().format(
            filename=filename,
            content=content.getvalue(),
            array_length=raw_content_len + (0 if args.no_null_terminated else 1),
            length=raw_content_len
        )
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, "w") as fout:
                fout.write(generated)
                fout.write("\n")
        else:
            print(generated)
