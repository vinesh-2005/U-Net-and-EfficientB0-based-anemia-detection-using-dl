"""Configuration loading and validation."""
from pathlib import Path
from typing import Any
import yaml

def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return value

def save_yaml(path: str | Path, value: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)

def require_confirmed_label_rule(labels: dict[str, Any]) -> float:
    target = labels.get("target", {})
    threshold = target.get("threshold_g_dl")
    definition = target.get("label_definition")
    if threshold is None or not definition or str(target.get("label_rule_id", "")).startswith("UNVERIFIED"):
        raise ValueError("The EYES-DEFY-ANEMIA Hb threshold and study label definition must be confirmed in configs/labels.yaml.")
    return float(threshold)
