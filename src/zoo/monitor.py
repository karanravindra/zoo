"""GPU monitoring for training logs: utilisation, memory, power and temperature, sampled in the background."""

from __future__ import annotations

import threading

import torch

GPU_FIELDS = ["gpu_util", "gpu_mem_gb", "gpu_alloc_gb", "gpu_reserved_gb", "gpu_power_w", "gpu_temp_c"]


class GpuMonitor:
    """Samples one CUDA device through NVML every `interval` seconds on a daemon thread.

    `read()` returns the readings since the previous `read()` as a dict keyed by `GPU_FIELDS`, and starts a new
    window:
      gpu_util         mean utilisation, percent
      gpu_mem_gb       most device memory in use (all processes), GiB
      gpu_alloc_gb     most memory torch had allocated to tensors in this process, GiB
      gpu_reserved_gb  memory torch's allocator holds right now, GiB
      gpu_power_w      mean power draw, watts
      gpu_temp_c       highest temperature, Celsius
    Off CUDA, or if NVML is unavailable, the NVML fields are left out (blank cells in metrics.csv); the two torch
    fields are still reported on CUDA. `summary()` gives run-wide figures; `close()` stops the thread."""

    def __init__(self, device: torch.device | str, interval: float = 0.5) -> None:
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self._nvml = self._handle = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reset()
        self._run_util_sum, self._run_n, self._run_mem, self._run_alloc = 0.0, 0, 0.0, 0.0
        if not self.cuda:
            return
        self._index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        try:
            import pynvml

            pynvml.nvmlInit()
            uuid = getattr(torch.cuda.get_device_properties(self._index), "uuid", None)
            # by UUID, so a CUDA_VISIBLE_DEVICES remapping cannot point this at another card
            self._handle = (
                pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}") if uuid else pynvml.nvmlDeviceGetHandleByIndex(self._index)
            )
            self._nvml = pynvml
        except Exception as e:  # no NVML: keep going with the torch numbers only
            print(f"GpuMonitor: NVML unavailable ({e}); logging torch memory only")
            return
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    def _reset(self) -> None:
        self._util_sum, self._power_sum, self._n, self._mem, self._temp = 0.0, 0.0, 0, 0.0, 0.0

    def _sample(self) -> None:
        n = self._nvml
        try:
            util = n.nvmlDeviceGetUtilizationRates(self._handle).gpu
            mem = n.nvmlDeviceGetMemoryInfo(self._handle).used / 2**30
            power = n.nvmlDeviceGetPowerUsage(self._handle) / 1000
            temp = n.nvmlDeviceGetTemperature(self._handle, n.NVML_TEMPERATURE_GPU)
        except Exception:
            return
        with self._lock:
            self._util_sum += util
            self._power_sum += power
            self._n += 1
            self._mem, self._temp = max(self._mem, mem), max(self._temp, temp)
            self._run_util_sum += util
            self._run_n += 1
            self._run_mem = max(self._run_mem, mem)

    def _loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self._sample()

    def read(self) -> dict[str, float]:
        if not self.cuda:
            return {}
        out: dict[str, float] = {}
        if self._nvml is not None:
            if self._n == 0:
                self._sample()  # a window shorter than the sampling interval
            with self._lock:
                if self._n:
                    out = {
                        "gpu_util": self._util_sum / self._n,
                        "gpu_mem_gb": self._mem,
                        "gpu_power_w": self._power_sum / self._n,
                        "gpu_temp_c": self._temp,
                    }
                self._reset()
        out["gpu_alloc_gb"] = torch.cuda.max_memory_allocated(self._index) / 2**30
        out["gpu_reserved_gb"] = torch.cuda.memory_reserved(self._index) / 2**30
        self._run_alloc = max(self._run_alloc, out["gpu_alloc_gb"])
        torch.cuda.reset_peak_memory_stats(self._index)
        return out

    def summary(self) -> dict[str, float]:
        """Whole-run figures for results.json: peak device memory, peak torch allocation, mean utilisation."""
        if not self.cuda:
            return {}
        out = {"peak_gpu_alloc_gb": self._run_alloc}
        if self._run_n:
            out |= {"peak_gpu_mem_gb": self._run_mem, "mean_gpu_util": self._run_util_sum / self._run_n}
        return out

    def close(self) -> None:
        self._stop.set()
