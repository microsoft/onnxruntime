import ctypes
import time

from nvml_monitor import get_nvml


def sample_clocks(uuid):
    monitor = get_nvml()
    handle = monitor._handle(uuid)

    def read(name, indices=(), wide=False):
        value_type = ctypes.c_ulonglong if wide else ctypes.c_uint
        value = value_type()
        function = getattr(monitor.library, name)
        function.argtypes = [ctypes.c_void_p, *[ctypes.c_uint for _ in indices], ctypes.POINTER(value_type)]
        function.restype = ctypes.c_int
        monitor._check(function(handle, *indices, ctypes.byref(value)), name)
        return value.value

    return dict(monotonic_ns=time.monotonic_ns(),
        sm_clock_mhz=read("nvmlDeviceGetClockInfo", (1,)),
        memory_clock_mhz=read("nvmlDeviceGetClockInfo", (2,)),
        temperature_c=read("nvmlDeviceGetTemperature", (0,)),
        power_w=read("nvmlDeviceGetPowerUsage") / 1000,
        throttle_reason_mask=read("nvmlDeviceGetCurrentClocksThrottleReasons", wide=True))
