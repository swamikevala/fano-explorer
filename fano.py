#!/usr/bin/env python3
"""
Fano Platform Launcher

Usage:
    python fano.py              # Start control panel (and pool if configured)
    python fano.py --no-pool    # Start control panel without auto-starting pool
    python fano.py pool         # Start only the pool service
    python fano.py explorer     # Start only the explorer
    python fano.py documenter   # Start only the documenter

The control panel provides a web UI at http://localhost:8080 where you can
manage the explorer and documenter, view logs, and monitor status.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import yaml

FANO_ROOT = Path(__file__).resolve().parent
LOGS_DIR = FANO_ROOT / "logs"

# Load .env file
env_path = FANO_ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = FANO_ROOT / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


def print_banner():
    """Print the Fano banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ███████╗ █████╗ ███╗   ██╗ ██████╗                     ║
    ║   ██╔════╝██╔══██╗████╗  ██║██╔═══██╗                    ║
    ║   █████╗  ███████║██╔██╗ ██║██║   ██║                    ║
    ║   ██╔══╝  ██╔══██║██║╚██╗██║██║   ██║                    ║
    ║   ██║     ██║  ██║██║ ╚████║╚██████╔╝                    ║
    ║   ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝                     ║
    ║                                                           ║
    ║   Mathematical Discovery Platform                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def start_pool() -> subprocess.Popen:
    """Start the pool service."""
    print("  Starting pool service...")

    # Ensure logs directory exists
    LOGS_DIR.mkdir(exist_ok=True)
    log_file = open(LOGS_DIR / "pool.log", "w", encoding="utf-8", buffering=1)

    pool_script = FANO_ROOT / "pool" / "run_pool.py"
    proc = subprocess.Popen(
        [sys.executable, str(pool_script)],
        cwd=str(FANO_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    time.sleep(3)  # Give it time to start
    if proc.poll() is not None:
        print("  [ERROR] Pool failed to start! Check logs/pool.log")
        return None
    print(f"  Pool started (PID: {proc.pid})")
    print(f"  Pool logs: logs/pool.log")
    return proc


def start_control_panel(config: dict, pool_proc: subprocess.Popen = None) -> None:
    """Start the control panel web server."""
    from control.server import start_server

    host = config.get("control", {}).get("host", "127.0.0.1")
    port = config.get("control", {}).get("port", 8080)
    debug = config.get("control", {}).get("debug", True)

    print(f"\n  Control Panel: http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")

    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        url = f"http://{host}:{port}"
        try:
            # Try webbrowser first
            webbrowser.open(url)
        except Exception:
            # Fallback for Windows
            if sys.platform == "win32":
                os.startfile(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    start_server(host=host, port=port, debug=debug, pool_process=pool_proc)


def run_pool_only():
    """Run only the pool service."""
    print("\n  Starting Pool Service...")
    pool_script = FANO_ROOT / "pool" / "run_pool.py"
    try:
        subprocess.run(
            [sys.executable, str(pool_script)],
            cwd=str(FANO_ROOT),
        )
    except KeyboardInterrupt:
        print("\n  Stopped.")


def run_explorer_only(mode: str = "start"):
    """Run only the explorer."""
    print(f"\n  Starting Explorer ({mode})...")
    explorer_script = FANO_ROOT / "explorer" / "fano_explorer.py"
    try:
        subprocess.run([sys.executable, str(explorer_script), mode], cwd=str(FANO_ROOT / "explorer"))
    except KeyboardInterrupt:
        print("\n  Stopped.")


def run_documenter_only():
    """Run only the documenter."""
    print("\n  Starting Documenter...")
    documenter_script = FANO_ROOT / "documenter" / "fano_documenter.py"
    try:
        subprocess.run([sys.executable, str(documenter_script), "start"], cwd=str(FANO_ROOT / "documenter"))
    except KeyboardInterrupt:
        print("\n  Stopped.")


def main():
    print_banner()

    config = load_config()

    # Parse arguments
    args = sys.argv[1:]

    # Handle specific component commands
    if "pool" in args:
        run_pool_only()
        return

    if "explorer" in args:
        mode = "start"
        if "backlog" in args:
            mode = "backlog"
        elif "review" in args:
            mode = "review"
        run_explorer_only(mode)
        return

    if "documenter" in args:
        run_documenter_only()
        return

    # Default: Start control panel (and pool if configured)
    pool_proc = None
    auto_start_pool = config.get("llm", {}).get("pool", {}).get("auto_start", True)

    if auto_start_pool and "--no-pool" not in args:
        pool_proc = start_pool()

    try:
        start_control_panel(config, pool_proc=pool_proc)
    except KeyboardInterrupt:
        print("\n  Shutting down...")
    finally:
        # Clean up all processes started from control panel
        from control.server import cleanup_all_processes
        cleanup_all_processes()
        print("  Done.")


if __name__ == "__main__":
    main()
