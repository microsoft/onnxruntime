# CGManifest Files
This directory contains CGManifest (cgmanifest.json) files.
See here for details: https://docs.opensource.microsoft.com/tools/cg/cgmanifest.html

`cgmanifests/cgmanifest.json` contains entries that don't belong in more specific categories (e.g., git submodules).

## Git Submodules
`cgmanifests/submodules/cgmanifest.json` contains entries for git submodules.
It can be generated like this:

```
$ cd <repo root>

$ python cgmanifests/submodules/generate_submodule_cgmanifest.py > cgmanifests/submodules/cgmanifest.json
```

Please update this file when git submodules change.
