import ctypes
import unittest
from unittest import mock

import nvml_monitor as nvml
import safe_model_run


class NvmlTests(unittest.TestCase):
    def setUp(self):
        self.library = mock.Mock()
        for name in ("nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetHandleByUUID"):
            getattr(self.library, name).return_value = 0
        self.library.nvmlErrorString.return_value = b"test NVML error"
        self.patch = mock.patch.object(nvml.ctypes, "CDLL", return_value=self.library)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = nvml.NvmlClient()
        self.addCleanup(self.client.close)

    def responses(self, values):
        def query(handle, count, processes):
            code, pids = next(values)
            ctypes.cast(count, ctypes.POINTER(ctypes.c_uint)).contents.value = len(pids)
            if processes is not None and code == 0:
                for index, pid in enumerate(pids):
                    processes[index].pid = pid
            return code
        self.library.nvmlDeviceGetComputeRunningProcesses_v2.side_effect = query

    def test_processes_and_changed_capacity(self):
        self.responses(iter([(7, [10]), (7, [10, 20]), (0, [10, 20])]))
        self.assertEqual(self.client.compute_pids("GPU-test"), {10, 20})

    def test_no_processes(self):
        self.responses(iter([(0, [])]))
        self.assertEqual(self.client.compute_pids("GPU-test"), set())

    def test_query_failure_is_not_an_empty_process_list(self):
        self.responses(iter([(3, [])]))
        with self.assertRaisesRegex(RuntimeError, "NVML 3"):
            self.client.compute_pids("GPU-test")

    def test_continuous_capacity_growth_fails(self):
        self.responses(iter([(7, [1]), (7, [1, 2]), (7, [1, 2, 3]), (7, [1, 2, 3, 4])]))
        with self.assertRaisesRegex(RuntimeError, "kept growing"):
            self.client.compute_pids("GPU-test")

    def test_inconsistent_capacity_fails(self):
        self.responses(iter([(7, [])]))
        with self.assertRaisesRegex(RuntimeError, "without a required size"):
            self.client.compute_pids("GPU-test")

    def test_memory_bytes_are_not_rounded_up(self):
        def memory(handle, pointer):
            ctypes.cast(pointer, ctypes.POINTER(nvml.MemoryInfo)).contents.free = 130000 * 1048576 - 1
            return 0
        self.library.nvmlDeviceGetMemoryInfo.side_effect = memory
        self.assertLess(self.client.free_memory_bytes("GPU-test") // 1048576, 130000)

    def test_closed_client_fails_and_shutdown_is_idempotent(self):
        self.client.close()
        self.client.close()
        self.library.nvmlShutdown.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.client.compute_pids("GPU-test")

    def test_guard_uses_process_api(self):
        with mock.patch.object(safe_model_run, "get_nvml", return_value=self.client):
            self.responses(iter([(7, [123]), (0, [123])]))
            self.assertEqual(safe_model_run.gpu_compute_pids("GPU-test"), {123})

    def test_process_lifetime_factory_initializes_once(self):
        nvml.get_nvml.cache_clear()
        self.addCleanup(nvml.get_nvml.cache_clear)
        with mock.patch.object(nvml, "NvmlClient", return_value=self.client) as constructor, \
                mock.patch.object(nvml.atexit, "register") as register:
            self.assertIs(nvml.get_nvml(), nvml.get_nvml())
            constructor.assert_called_once()
            register.assert_called_once_with(self.client.close)


if __name__ == "__main__":
    unittest.main()
