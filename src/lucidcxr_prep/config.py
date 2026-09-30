"""Loads configs/paths.yaml and resolves every path relative to the repo root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    montgomery_raw: Path
    shenzhen_raw: Path
    output_root: Path
    manifests_dir: Path
    seed: int
    resolutions: tuple[int, ...]


def load_paths(config_path: Path | None = None) -> Paths:
    config_path = config_path or (REPO_ROOT / "configs" / "paths.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    def resolve(rel: str) -> Path:
        return (REPO_ROOT / rel).resolve()

    return Paths(
        montgomery_raw=resolve(raw["raw_data"]["montgomery"]),
        shenzhen_raw=resolve(raw["raw_data"]["shenzhen"]),
        output_root=resolve(raw["output_root"]),
        manifests_dir=resolve(raw["manifests_dir"]),
        seed=int(raw["seed"]),
        resolutions=tuple(int(r) for r in raw["resolutions"]),
    )
