# Code Refactoring for WebGL backend

## Items

- Simplify workflow for kernel implementation

- New artifact cache

- New Texture management

## Simplify workflow for kernel implementation

1.  Merge `RunData` and `ProgramInfo`.

    - remove `RunData` and `createRunData`.
    - discard `inputTextureDatas` and `outputTextureData` from `RunData`.
    - move `uniformData` and `draw` to `ProgramInfo`.

2.  No operator class any more.

    - operator resolves to kernel implementation function.
    - kernel implementation function calls inference handler to run webgl program.
    - kernel implementation offers a function to construct program info.

3.  Kernel implementation no longer need to create texture layout and texture data.

## New artifact cache

1.  Key is generated from ProgramInfo and input

## New Texture management

1.  New texture cache

    - (inference) In-use cache: (TensorID) => textures[textureType]--> texture info

    - (inference) Recycled cache (w,h,c) => texture
