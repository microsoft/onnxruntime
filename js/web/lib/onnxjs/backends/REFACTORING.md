# Code Refactoring for WebGL backend

## Items

- Simplify workflow for kernel implementation

## Simplify workflow for kernel implementation

1.  Merge `RunData` and `ProgramInfo`.

    - remove `RunData` and `createRunData`.
    - discard `inputTextureDatas` and `outputTextureData` from `RunData`.
    - move `uniformData` and `draw` to `ProgramInfo`.

2.  No operator class any more.

    - operator resolves to kernel implementation function.
    - kernel implementation function calls inference handler to run webgl program.
    - kernel implementation offers a function to construct program info.
