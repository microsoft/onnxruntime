import atexit
import ctypes
import functools

NVML_ERROR_INSUFFICIENT_SIZE = 7


class MemoryInfo(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class ProcessInfoV2(ctypes.Structure):
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong),
                ("gpuInstanceId", ctypes.c_uint), ("computeInstanceId", ctypes.c_uint)]


class NvmlClient:
    def __init__(self):
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        signatures = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlErrorString": [ctypes.c_int],
            "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)],
            "nvmlDeviceGetComputeRunningProcesses_v2": [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ProcessInfoV2)],
        }
        for name, arguments in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_char_p if name == "nvmlErrorString" else ctypes.c_int
        self._check(self.library.nvmlInit_v2(), "nvmlInit_v2")
        self.closed = False

    def _check(self, code, operation):
        if code:
            message = self.library.nvmlErrorString(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"{operation} failed: NVML {code}: {message}")

    def _handle(self, uuid):
        if self.closed:
            raise RuntimeError("NVML client is closed")
        handle = ctypes.c_void_p()
        self._check(self.library.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)),
                    "nvmlDeviceGetHandleByUUID")
        return handle

    def free_memory_bytes(self, uuid):
        memory = MemoryInfo()
        self._check(self.library.nvmlDeviceGetMemoryInfo(self._handle(uuid), ctypes.byref(memory)),
                    "nvmlDeviceGetMemoryInfo")
        return memory.free

    def compute_pids(self, uuid):
        handle = self._handle(uuid)
        count = ctypes.c_uint()
        query = self.library.nvmlDeviceGetComputeRunningProcesses_v2
        code = query(handle, ctypes.byref(count), None)
        if code != NVML_ERROR_INSUFFICIENT_SIZE:
            self._check(code, "nvmlDeviceGetComputeRunningProcesses_v2")
        if count.value == 0:
            if code == NVML_ERROR_INSUFFICIENT_SIZE:
                raise RuntimeError("NVML reported insufficient process capacity without a required size")
            return set()
        for _ in range(3):
            processes = (ProcessInfoV2 * count.value)()
            code = query(handle, ctypes.byref(count), processes)
            if code == NVML_ERROR_INSUFFICIENT_SIZE:
                continue
            self._check(code, "nvmlDeviceGetComputeRunningProcesses_v2")
            return {processes[index].pid for index in range(count.value)}
        raise RuntimeError("NVML process list kept growing across three allocation attempts")

    def close(self):
        if not self.closed:
            self._check(self.library.nvmlShutdown(), "nvmlShutdown")
            self.closed = True


@functools.cache
def get_nvml():
    client = NvmlClient()
    atexit.register(client.close)
    return client
