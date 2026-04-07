#!/usr/bin/env python3
"""Diagnostic script to check jinja2/markupsafe availability in the Python environment."""

import sys
import os


def main():
    print("=== [DIAG] Python info ===")
    print("executable:", sys.executable)
    print("version:", sys.version)
    print()

    print("=== [DIAG] sys.path ===")
    for p in sys.path:
        print(" ", p)
    print()

    print("=== [DIAG] Try importing jinja2 ===")
    try:
        import jinja2
        print("jinja2 loaded from:", jinja2.__file__)
        print("jinja2 version:", jinja2.__version__)
        print("has BaseLoader:", hasattr(jinja2, "BaseLoader"))
    except Exception as e:
        print("Cannot import jinja2:", e)
    print()

    print("=== [DIAG] Try importing markupsafe ===")
    try:
        import markupsafe
        print("markupsafe loaded from:", markupsafe.__file__)
        print("has Markup:", hasattr(markupsafe, "Markup"))
    except Exception as e:
        print("Cannot import markupsafe:", e)
    print()

    # If a build directory is passed as argument, inspect the Dawn-fetched copies
    if len(sys.argv) > 1:
        build_dir = sys.argv[1]
        inspect_dawn_bundled(build_dir)


def inspect_dawn_bundled(build_dir):
    """Walk the build directory to find Dawn's fetched jinja2/markupsafe and inspect them."""
    print("=== [DIAG] Searching for Dawn-fetched jinja2 in", build_dir, "===")

    jinja2_dir = None
    markupsafe_dir = None

    for root, dirs, files in os.walk(build_dir):
        if root.endswith("dawn-src/third_party/jinja2"):
            jinja2_dir = root
        elif root.endswith("dawn-src/third_party/markupsafe"):
            markupsafe_dir = root
        if jinja2_dir and markupsafe_dir:
            break

    # Inspect jinja2
    if jinja2_dir:
        print("Found jinja2 dir:", jinja2_dir)
        print("Contents:")
        for f in sorted(os.listdir(jinja2_dir))[:20]:
            full = os.path.join(jinja2_dir, f)
            size = os.path.getsize(full) if os.path.isfile(full) else -1
            print(f"  {f}  ({size} bytes)" if size >= 0 else f"  {f}/")

        init_py = os.path.join(jinja2_dir, "__init__.py")
        if os.path.isfile(init_py):
            with open(init_py) as f:
                content = f.read()
            print("__init__.py size:", len(content), "bytes")
            print("BaseLoader in __init__.py:", "BaseLoader" in content)
            # Show first 20 lines
            print("--- __init__.py first 20 lines ---")
            for line in content.splitlines()[:20]:
                print(" ", line)
        else:
            print("WARNING: __init__.py NOT FOUND!")
    else:
        print("Dawn jinja2 directory NOT found")
    print()

    # Inspect markupsafe
    if markupsafe_dir:
        print("Found markupsafe dir:", markupsafe_dir)
        print("Contents:")
        for f in sorted(os.listdir(markupsafe_dir))[:20]:
            full = os.path.join(markupsafe_dir, f)
            size = os.path.getsize(full) if os.path.isfile(full) else -1
            print(f"  {f}  ({size} bytes)" if size >= 0 else f"  {f}/")

        init_py = os.path.join(markupsafe_dir, "__init__.py")
        if os.path.isfile(init_py):
            with open(init_py) as f:
                content = f.read()
            print("__init__.py size:", len(content), "bytes")
            print("--- __init__.py first 20 lines ---")
            for line in content.splitlines()[:20]:
                print(" ", line)
        else:
            print("WARNING: __init__.py NOT FOUND!")
    else:
        print("Dawn markupsafe directory NOT found")
    print()

    # Try importing the bundled copies
    if jinja2_dir:
        parent = os.path.dirname(jinja2_dir)
        print("=== [DIAG] Try importing Dawn-bundled jinja2 from", parent, "===")
        # Clear any previously imported jinja2/markupsafe
        for mod_name in list(sys.modules):
            if mod_name == "jinja2" or mod_name.startswith("jinja2."):
                del sys.modules[mod_name]
            if mod_name == "markupsafe" or mod_name.startswith("markupsafe."):
                del sys.modules[mod_name]

        sys.path.insert(0, parent)
        try:
            import jinja2
            print("jinja2 loaded from:", jinja2.__file__)
            print("jinja2 version:", jinja2.__version__)
            print("has BaseLoader:", hasattr(jinja2, "BaseLoader"))
            if not hasattr(jinja2, "BaseLoader"):
                print("Available attributes:", [a for a in dir(jinja2) if not a.startswith("_")])
        except Exception as e:
            import traceback
            print("IMPORT FAILED:", e)
            traceback.print_exc()


if __name__ == "__main__":
    main()
