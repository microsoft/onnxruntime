This directory contains Component Governance cgmanifest.json files.
See here for more details: https://docs.opensource.microsoft.com/tools/cg/cgmanifest.html

Any cgmanifest file entries that don't belong in more specific categories
(e.g., git submodules) can go into `cgmanifests/cgmanifest.json`.

### Git Submodules
The cgmanifest.json file containing git submodule entries can be generated like this:

```
$ cd <repo root>

$ python cgmanifests/submodules/generate_submodule_cgmanifest.py > cgmanifests/submodules/cgmanifest.json
```
