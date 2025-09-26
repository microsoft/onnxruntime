Internal test EP that is used to test and validate interactions between the ORT framework, optimizers and an EP.

Current usages:
 - validating support in a minimal build for an EP that compiles nodes into kernels at runtime
 - validating EP support for a mix of static and compiled kernels
 - validating support for layout transform to NHWC for static and compiled kernels
 - validating allocator sharing between EPs