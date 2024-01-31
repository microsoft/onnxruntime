Selected files from https://github.com/apple/coremltools/blob/7.1/ that are used to create the ML Model at runtime.
Directory structure matches the original.
We manually build the C++ sources using the ORT CoreML provider makefile.

There are a few edits required to allow building on Windows and Linux to test the CoreML EP implementation on
non-Apple platforms. This are denoted with ORT_EDIT comments. These are the only changes to the original sources.
