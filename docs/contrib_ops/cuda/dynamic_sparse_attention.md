# CUDA DynamicSparseAttention grouped decode

`ORT_DSA_INT8_GROUPED_MAX_ROWS` controls the maximum query rows eligible for
the existing grouped INT8 decode kernel. The default is 2, preserving existing
dispatch. Set it to 4 before starting the process to allow three- and four-row
speculative verification; the supported range is 2 through 4.

All other grouped-kernel eligibility conditions still apply. Ineligible cache
formats, shapes, and devices retain their existing dispatch. This does not
change the operator schema, cache layout, or sparse-index selection semantics.

The INT8 grouped-decode and vector-staging reference tests cover one through
four query rows. To exercise the opt-in path, run the INT8 grouped-decode tests
in a fresh process with `ORT_DSA_INT8_GROUPED_MAX_ROWS=4`.
