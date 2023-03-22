class DeviceArray:
    def __init__(self, ndarray) -> None: ...
    def UpdateHostNumpyArray(self) -> None: ...  # noqa: N802

class blas_op:  # noqa: N801
    T: int
    N: int

def is_composable_kernel_available(*args, **kwargs): ...
