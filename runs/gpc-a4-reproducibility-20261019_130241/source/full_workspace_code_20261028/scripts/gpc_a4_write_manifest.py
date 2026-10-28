#!/usr/local/bin/python3.10
import hashlib
import json
import sys
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    out, run_time, log, status, protocol = map(Path, sys.argv[1:])
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            files[str(path)] = {"size": path.stat().st_size, "sha256": digest(path)}
    payload = {
        "run_time": run_time.name,
        "linked_previous_runs": ["20261015_155822", "20261016_141306"],
        "main_log": str(log),
        "status_json": str(status),
        "frozen_expanded_protocol": str(protocol),
        "frozen_expanded_protocol_sha256": digest(protocol),
        "official_query_gallery_used_during_training": False,
        "historical_gpc_v1_untouched": True,
        "prcc_touched": False,
        "ndp_included": False,
        "science_config": {
            "loss": "CE + Triplet + Bio + 0.1*GPC",
            "pair_budget": "per-batch minimum count across cross/same/id",
            "amp_training_initial_scale": 1024,
        },
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
