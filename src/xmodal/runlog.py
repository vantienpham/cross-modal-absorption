"""Run directories with the three JSON sidecars the sync script pulls back.

``config.json`` / ``env.json`` / ``metrics.json`` beside arbitrarily large
artifacts, so ``slurm/sync.sh pull`` can fetch the summary of every run and the
tensors of none. ``env.json`` records the assigned GPU and driver because this
cluster is not uniform across partitions, and two "identical" runs that disagree
usually disagree about the node they landed on.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, torch.dtype):
            return str(obj)
    except ImportError:
        pass
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


def _git_stamp() -> dict[str, str]:
    """Read the .git_commit stamp rsync leaves; the remote tree has no .git/."""
    stamp = Path(".git_commit")
    if not stamp.is_file():
        return {"commit": "unknown", "note": "no .git_commit stamp (use slurm/sync.sh push)"}
    out: dict[str, str] = {}
    for line in stamp.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _gpu_info() -> dict[str, Any]:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        gpus = [l.strip() for l in res.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    info: dict[str, Any] = {"nvidia_smi": gpus}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
    except ImportError:
        pass
    return info


class RunDir:
    """A single run's output directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, Any] = {}

    def write_config(self, config: Any) -> None:
        self._dump("config.json", _jsonable(config))

    def write_env(self, extra: dict[str, Any] | None = None) -> None:
        env = {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "slurm": {
                k: v for k, v in os.environ.items()
                if k.startswith("SLURM_") and k in {
                    "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_PARTITION",
                    "SLURM_JOB_NODELIST", "SLURM_NTASKS", "SLURM_CPUS_PER_TASK",
                }
            },
            "git": _git_stamp(),
            "gpu": _gpu_info(),
        }
        if extra:
            env.update(_jsonable(extra))
        self._dump("env.json", env)

    def log(self, **kw: Any) -> None:
        self._metrics.update(_jsonable(kw))

    def write_metrics(self) -> None:
        self._metrics["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._dump("metrics.json", self._metrics)

    def artifact(self, name: str) -> Path:
        return self.path / name

    def _dump(self, name: str, obj: Any) -> None:
        (self.path / name).write_text(json.dumps(obj, indent=2, sort_keys=True))
