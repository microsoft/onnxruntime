# CGManifest Files
This directory contains CGManifest (cgmanifest.json) files.
See here for details: https://docs.opensource.microsoft.com/tools/cg/cgmanifest.html

`cgmanifests/cgmanifest.json` contains entries that don't belong in more specific categories (e.g., git submodules).

## Git Submodules
`cgmanifests/submodules/cgmanifest.json` contains entries for git submodules.
It can be generated like this:

1. Change to the repository root directory.
2. Ensure the submodules are checked out. For example, with:
    ```
    $ git submodule update --init --recursive
    ```
3. Run the generator script:
    ```
    $ python cgmanifests/submodules/generate_submodule_cgmanifest.py > cgmanifests/submodules/cgmanifest.json
    ```

Please update this cgmanifest.json file when any git submodules change.
