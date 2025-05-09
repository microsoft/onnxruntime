"""
Github publishes the markdown documentation with jekyll enabled.
This extension does not publish any folder starting with `_`.
These folders need to be renamed.
"""

import os
import re


def rename_folder(root):
    """
    Renames all folder starting with `_`.
    Returns the list of renamed folders.
    """
    found = []
    for r, dirs, _files in os.walk(root):
        for name in dirs:
            if name.startswith("_"):
                found.append((r, name))
    renamed = []
    for r, name in found:
        into = name.lstrip("_")
        renamed.append((r, name, into))
        full_src = os.path.join(r, name)
        full_into = os.path.join(r, into)
        if os.path.exists(full_into):
            raise RuntimeError("%r already exists, previous documentation should be removed.")
        print(f"rename {full_src!r}")
        os.rename(full_src, full_into)

    return renamed


def replace_files(root, renamed):
    subs = {r[1]: r[2] for r in renamed}
    reg = re.compile('(\\"[a-zA-Z0-9\\.\\/\\?\\:@\\-_=#]+\\.([a-zA-Z]){2,6}([a-zA-Z0-9\\.\\&\\/\\?\\:@\\-_=#])*\\")')

    for r, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[-1] != ".html":
                continue
            full = os.path.join(r, name)
            with open(full, encoding="utf-8") as f:
                content = f.read()
            find = reg.findall(content)
            repl = []
            for f in find:
                if f[0].startswith("http"):
                    continue
                for k, v in subs.items():
                    if k == v:
                        raise ValueError(f"{k!r} == {v!r}")
                    if (f'"{k}') in f[0]:
                        repl.append((f[0], f[0].replace(f'"{k}', f'"{v}')))
                    if (f"/{k}") in f[0]:
                        repl.append((f[0], f[0].replace(f"/{k}", f"/{v}")))
            if len(repl) == 0:
                continue
            print(f"update {full!r}")
            for k, v in repl:
                content = content.replace(k, v)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        root = sys.argv[-1]
    else:
        root = "../../build/docs/html"
    print(f"look into {root!r}")
    ren = rename_folder(root)
    if len(ren) == 0:
        ren = [
            ("", "_static", "static"),
            ("", "_images", "images"),
            ("", "_downloads", "downloads"),
            ("", "_sources", "sources"),
            ("", "_modules", "modules"),
        ]
    replace_files(root, ren)
    print("done.")
