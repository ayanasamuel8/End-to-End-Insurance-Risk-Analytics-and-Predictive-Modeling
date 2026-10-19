"""Shared helpers for the DVC pipeline stages."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PARAMS = "params.yaml"


def load_params(path: str = DEFAULT_PARAMS) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args(description: str, argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--params", default=DEFAULT_PARAMS, help="Path to the params file.")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, payload: dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
