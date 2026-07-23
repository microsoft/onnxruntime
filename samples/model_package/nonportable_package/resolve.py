"""Resolve the non-portable (installed-layout) model package.

This package's model files live OUTSIDE the package directory, in a sibling
`external_assets/` folder. The manifest uses "layout": "installed", which allows
absolute paths, but a repository cannot commit machine-specific absolute paths
(and using ".." relative segments, while supported, is fragile and ugly).

So each variant ships an `ort_info.template.json` containing the placeholder
token `__ASSETS_DIR__`. Run this script once after checkout to write the real
`ort_info.json` files with the absolute path to `external_assets/` on this machine.

    python resolve.py                 # assets at ../external_assets (default)
    python resolve.py --assets DIR    # assets at a custom absolute/relative dir

After running, load the package normally (e.g. with the C++ sample). Re-run this
script if you move the checkout to a new location.
"""
import argparse
import glob
import os

TOKEN = "__ASSETS_DIR__"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=os.path.join(HERE, os.pardir, "external_assets"),
                    help="Path to the external_assets directory (default: ../external_assets).")
    args = ap.parse_args()

    assets_dir = os.path.abspath(args.assets)
    if not os.path.isdir(assets_dir):
        raise SystemExit(f"external assets directory not found: {assets_dir}")
    # forward slashes keep the JSON clean and are accepted by ORT on all platforms
    assets_posix = assets_dir.replace(os.sep, "/")

    templates = glob.glob(os.path.join(HERE, "**", "ort_info.template.json"), recursive=True)
    if not templates:
        raise SystemExit("no ort_info.template.json files found next to this script.")

    for tpl in templates:
        text = open(tpl, "r", encoding="utf-8").read().replace(TOKEN, assets_posix)
        out = os.path.join(os.path.dirname(tpl), "ort_info.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print("wrote", os.path.relpath(out, HERE))

    print(f"\nResolved {len(templates)} variant(s) against assets dir:\n  {assets_posix}")
    print("The package is now ready to load.")


if __name__ == "__main__":
    main()
