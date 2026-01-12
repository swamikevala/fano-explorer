"""
Configuration management for the control panel.
"""

from pathlib import Path

import yaml

# Core paths
FANO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = FANO_ROOT / "logs"
EXPLORER_DATA_DIR = FANO_ROOT / "explorer" / "data"
DOC_PATH = FANO_ROOT / "documenter" / "document" / "main.md"


def load_config() -> dict:
    """Load the unified configuration from config.yaml."""
    config_path = FANO_ROOT / "config.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {}


def save_config(config: dict) -> None:
    """Save configuration to config.yaml."""
    config_path = FANO_ROOT / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
