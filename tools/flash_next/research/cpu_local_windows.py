import os
from pathlib import Path


def inherit_worker_affinity():
    expected = set(os.sched_getaffinity(0))
    if not expected:
        raise RuntimeError("Cannot inherit an empty main-thread CPU affinity")
    previous = {}
    for task in sorted(Path("/proc/self/task").iterdir()):
        tid = int(task.name)
        previous[task.name] = sorted(os.sched_getaffinity(tid))
        os.sched_setaffinity(tid, expected)
        if set(os.sched_getaffinity(tid)) != expected:
            raise RuntimeError(f"Worker {tid} did not inherit the main-thread CPU affinity")
    return dict(target_cpus=sorted(expected), previous_thread_masks=previous,
                scope="existing workers after model load, before warmup; new workers inherit main-thread mask")


class CpuLocalWindows:
    def __init__(self):
        self.control = set(os.sched_getaffinity(0))
        self.candidate = self.control & set(range(48, 96))
        if not self.candidate or self.candidate == self.control:
            raise RuntimeError("GPU7 CPU-local comparison needs both local and nonlocal allowed CPUs")

    def switch(self, mode):
        if mode not in (0, 1):
            raise ValueError(f"Unsupported CPU-local mode: {mode}")
        expected = self.candidate if mode == 1 else self.control
        os.sched_setaffinity(0, expected)
        if set(os.sched_getaffinity(0)) != expected:
            raise RuntimeError("Main-thread CPU affinity did not match the requested comparison mode")

    def restore(self):
        os.sched_setaffinity(0, self.control)
        if set(os.sched_getaffinity(0)) != self.control:
            raise RuntimeError("Main-thread CPU affinity was not restored")

    def metadata(self):
        return dict(control_cpus=sorted(self.control), candidate_cpus=sorted(self.candidate),
                    scope="benchmark main thread only; existing worker threads unchanged",
                    gpu_index=7, gpu_uuid="GPU-0e70e969-bbfc-80c5-d582-88be134632ea",
                    candidate_numa_node=1, gpu_topology_verified_by_host_runner=True)
