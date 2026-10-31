"""Inventory raw images, annotations, and clinical files without assuming semantics."""
from pathlib import Path
from typing import Any

def inventory(root: str | Path, extensions: list[str]) -> dict[str, list[str]]:
    root = Path(root)
    ext = {x.lower() for x in extensions}
    return {"images": sorted(str(p.relative_to(root)) for p in (root / "images").rglob("*") if p.suffix.lower() in ext),
            "annotations": sorted(str(p.relative_to(root)) for p in (root / "annotations").rglob("*") if p.suffix.lower() in ext),
            "clinical": sorted(str(p.relative_to(root)) for p in (root / "clinical").rglob("*" ) if p.is_file())}

def write_inventory_report(items: dict[str, list[str]], output: str | Path) -> None:
    import json
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(items, indent=2), encoding="utf-8")
