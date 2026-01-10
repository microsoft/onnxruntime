Generic transpose optimizer that has an abstraction layer so it is independent of the ORT implementation.

optimizer_api.h: Defines the API for the abstraction layer
ort_optimizer_api_impl.*: ORT implementation of the abstraction layer
transpose_optimizer.*: Generic implementation of logic to be able to move a Transpose node past another node.
ort_transpose_optimizer.*: ORT specific extensions to transpose optimization.
