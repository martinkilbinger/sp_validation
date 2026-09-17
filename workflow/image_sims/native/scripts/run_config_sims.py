#!/usr/bin/env python3
"""Run-config handling for the native image-sims workflow.

`$name` in any value is replaced by the top-level scalar key `name`
(`my_base_dir`, `sim`, ...), repeatedly, so one key can build on another.
Nothing else expands anything, so a leftover `$name` would become a literal
directory -- `expand_vars` raises instead.

Used by the Snakefile (on snakemake's own merged config) and by
status_sims.py (which loads the files itself, via `load`).
"""

import re
from pathlib import Path

MAX_DEPTH = 5


class ConfigError(ValueError):
    pass


def expand_vars(config: dict) -> dict:
    """Expand $name in every value of `config`, in place."""
    scalars = {k: v for k, v in config.items()
               if isinstance(v, (str, int, float)) and not isinstance(v, bool)}

    def sub(value, depth=0):
        if isinstance(value, str):
            out = re.sub(r"\$(\w+)",
                         lambda m: str(scalars.get(m.group(1), m.group(0))), value)
            return out if out == value or depth >= MAX_DEPTH else sub(out, depth + 1)
        if isinstance(value, dict):
            return {k: sub(v, depth) for k, v in value.items()}
        if isinstance(value, list):
            return [sub(v, depth) for v in value]
        return value

    for key in list(config):
        config[key] = sub(config[key])
    left = [k for k, v in config.items() if isinstance(v, str) and "$" in v]
    if left:
        raise ConfigError(
            f"unexpanded $variable(s) in {left} -- a $name must name a "
            f"top-level scalar key of the run config.")
    return config


def load(run_config, config_yaml=None) -> dict:
    """config.yaml, then `run_config` on top, then $name expansion."""
    import yaml

    if config_yaml is None:
        config_yaml = Path(__file__).resolve().parent.parent / "config.yaml"
    config = {}
    for path in (config_yaml, run_config):
        if path and Path(path).exists():
            with open(path) as f:
                config.update(yaml.safe_load(f) or {})
    return expand_vars(config)


def run_dirs(run_config, config_yaml=None):
    """Return `(run_dir, products_dir)` as Paths, products_dir defaulting to
    run_dir exactly as the Snakefile does."""
    outputs = load(run_config, config_yaml).get("outputs") or {}
    if "run_dir" not in outputs:
        raise ConfigError(f"{run_config}: outputs.run_dir is not set")
    run_dir = Path(outputs["run_dir"])
    return run_dir, Path(outputs.get("products_dir") or run_dir)
