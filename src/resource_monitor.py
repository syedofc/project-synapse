# ==============================================================================
# File 06d: 06_src/06d_resource_monitor.py
# Purpose: Simulates a device resource monitor.
# ==============================================================================
"""
The ResourceMonitor provides the "context" vector to the TaskWeaver.

It now supports:
1. real telemetry via `psutil` when available,
2. explicit manual overrides for adaptation analysis, and
3. deterministic zero fallbacks when telemetry is unavailable.
"""
import os

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - availability depends on environment
    psutil = None

class ResourceMonitor:
    def __init__(self, context_dim, mode="telemetry", fallback_mode="zeros"):
        """
        Initializes the resource monitor.

        Args:
            context_dim (int): The dimensionality of the context vector to produce.
        """
        self.context_dim = context_dim
        self.mode = mode
        self.fallback_mode = fallback_mode
        self._override_context = None
        self.last_source = "uninitialized"
        print(
            f"ResourceMonitor initialized. mode={mode}, fallback_mode={fallback_mode}, "
            f"context_dim={context_dim}."
        )

    def set_context_override(self, context_vector):
        override = torch.as_tensor(context_vector, dtype=torch.float32)
        if override.numel() != self.context_dim:
            raise ValueError(
                f"Override context size {override.numel()} does not match expected "
                f"context_dim={self.context_dim}."
            )
        self._override_context = override.clone()
        self.last_source = "override"

    def clear_context_override(self):
        self._override_context = None

    def _resize_features(self, features):
        if self.context_dim == 0:
            return torch.zeros(0, dtype=torch.float32)

        if not features:
            features = [0.0]

        resized = []
        while len(resized) < self.context_dim:
            resized.extend(features)
        return torch.tensor(resized[: self.context_dim], dtype=torch.float32)

    def _safe_clip(self, value, lo=0.0, hi=1.0):
        return max(lo, min(hi, float(value)))

    def _collect_telemetry_features(self):
        if psutil is None:
            raise RuntimeError("psutil is not installed")

        virtual_memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())

        cpu_percent = self._safe_clip(psutil.cpu_percent(interval=None) / 100.0)
        memory_percent = self._safe_clip(virtual_memory.percent / 100.0)
        swap_percent = self._safe_clip(psutil.swap_memory().percent / 100.0)
        rss_fraction = self._safe_clip(process.memory_info().rss / max(virtual_memory.total, 1))

        if hasattr(os, "getloadavg"):
            load_one = os.getloadavg()[0]
            cpu_count = max(os.cpu_count() or 1, 1)
            load_norm = self._safe_clip(load_one / cpu_count)
        else:
            load_norm = 0.0

        battery_percent = 0.5
        plugged_in = 0.5
        if hasattr(psutil, "sensors_battery"):
            battery = psutil.sensors_battery()
            if battery is not None:
                if battery.percent is not None:
                    battery_percent = self._safe_clip(battery.percent / 100.0)
                plugged_in = 1.0 if battery.power_plugged else 0.0

        temperature_norm = 0.0
        if hasattr(psutil, "sensors_temperatures"):
            try:
                sensors = psutil.sensors_temperatures()
                values = []
                for entries in sensors.values():
                    for entry in entries:
                        if getattr(entry, "current", None) is not None:
                            values.append(float(entry.current))
                if values:
                    temperature_norm = self._safe_clip(sum(values) / len(values) / 100.0)
            except Exception:
                temperature_norm = 0.0

        return [
            cpu_percent,
            memory_percent,
            swap_percent,
            rss_fraction,
            load_norm,
            battery_percent,
            plugged_in,
            temperature_norm,
        ]

    def _fallback_context(self):
        if self.context_dim == 0:
            self.last_source = "zero-dim"
            return torch.zeros(0, dtype=torch.float32)

        if self.fallback_mode == "random":
            self.last_source = "random-fallback"
            return torch.rand(self.context_dim, dtype=torch.float32)

        self.last_source = "zero-fallback"
        return torch.zeros(self.context_dim, dtype=torch.float32)

    def get_context(self):
        """
        Returns the current context vector.

        Returns:
            torch.Tensor: A normalized tensor representing device context.
        """
        if self._override_context is not None:
            self.last_source = "override"
            return self._override_context.clone()

        if self.context_dim == 0:
            self.last_source = "zero-dim"
            return torch.zeros(0, dtype=torch.float32)

        if self.mode == "telemetry":
            try:
                features = self._collect_telemetry_features()
                self.last_source = "telemetry"
                return self._resize_features(features)
            except Exception:
                return self._fallback_context()

        if self.mode == "random":
            self.last_source = "random"
            return torch.rand(self.context_dim, dtype=torch.float32)

        if self.mode == "zeros":
            self.last_source = "zeros"
            return torch.zeros(self.context_dim, dtype=torch.float32)

        return self._fallback_context()
