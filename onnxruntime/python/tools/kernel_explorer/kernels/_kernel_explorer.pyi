class DeviceArray:
    def __init__(self, ndarray) -> None: ...
    def UpdateHostNumpyArray(self) -> None: ...  # noqa: N802
    def UpdateDeviceArray(self) -> None: ...  # noqa: N802

class blas_op:  # noqa: N801
    T: int
    N: int

class qkv_format:  # noqa: N801
    Q_K_V_BNSH: int
    Q_K_V_BSNH: int
    QKV_BSN3H: int
    Q_KV_BSNH_BSN2H: int

def is_composable_kernel_available(*args, **kwargs): ...
def is_hipblaslt_available(*args, **kwargs): ...

def enable_collect_tuning_results(*args, **kwargs): ...
def get_collected_tuning_results(*args, **kwargs): ...
