"""
Thread-safe process management for Fano components.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from .config import FANO_ROOT


class ProcessManager:
    """
    Thread-safe manager for component processes (pool, explorer, documenter).

    Uses a lock to protect state access from concurrent requests and
    background threads (e.g., server restart).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._processes = {
            "pool": None,
            "explorer": None,
            "documenter": None,
        }

    def get(self, component: str) -> Optional[subprocess.Popen]:
        """Get a process by component name."""
        with self._lock:
            return self._processes.get(component)

    def set(self, component: str, proc: Optional[subprocess.Popen]):
        """Set a process for a component."""
        with self._lock:
            self._processes[component] = proc

    def is_running(self, component: str) -> bool:
        """Check if a component is running."""
        with self._lock:
            proc = self._processes.get(component)
            return proc is not None and proc.poll() is None

    def get_pid(self, component: str) -> Optional[int]:
        """Get the PID of a running component."""
        with self._lock:
            proc = self._processes.get(component)
            if proc is not None and proc.poll() is None:
                return proc.pid
            return None

    def start_pool(self) -> subprocess.Popen:
        """Start the pool service."""
        pool_script = FANO_ROOT / "pool" / "run_pool.py"
        proc = subprocess.Popen(
            [sys.executable, str(pool_script)],
            cwd=str(FANO_ROOT),
        )
        self.set("pool", proc)
        return proc

    def start_explorer(self, mode: str = "start") -> subprocess.Popen:
        """Start the explorer."""
        explorer_script = FANO_ROOT / "explorer" / "fano_explorer.py"
        proc = subprocess.Popen(
            [sys.executable, str(explorer_script), mode],
            cwd=str(FANO_ROOT / "explorer"),
        )
        self.set("explorer", proc)
        return proc

    def start_documenter(self) -> subprocess.Popen:
        """Start the documenter."""
        documenter_script = FANO_ROOT / "documenter" / "fano_documenter.py"
        proc = subprocess.Popen(
            [sys.executable, str(documenter_script), "start"],
            cwd=str(FANO_ROOT / "documenter"),
        )
        self.set("documenter", proc)
        return proc

    def stop(self, component: str) -> bool:
        """
        Stop a component. Returns True if stopped successfully.
        """
        with self._lock:
            proc = self._processes.get(component)
            if proc is None or proc.poll() is not None:
                return False

            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

            self._processes[component] = None
            return True

    def cleanup_all(self):
        """Stop all managed processes."""
        with self._lock:
            for name, proc in self._processes.items():
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass

            # Clear all references
            for name in self._processes:
                self._processes[name] = None

    def register_external(self, component: str, proc: subprocess.Popen):
        """Register an externally started process."""
        self.set(component, proc)
