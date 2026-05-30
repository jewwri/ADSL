import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE = re.compile(
    r"(HalfCheetah|Hopper|Walker2d)v_4_"
    r"(action_perturbation|observation_corruption|reward_poisoning)_"
    r"(random_sparse|bursty)_"
    r"(clean|attack_none|attack_defended|detector_only)_seed(\d+)"
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise AssertionError(f"Missing CSV: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _run_key(row: dict) -> tuple[str, str, str, str, int]:
    match = RUN_RE.search(row.get("run_name", ""))
    if not match:
        raise AssertionError(f"Cannot parse run_name={row.get('run_name')!r}")
    env, poison_type, schedule, condition, seed = match.groups()
    env_id = {"HalfCheetah": "HalfCheetah-v4", "Hopper": "Hopper-v4", "Walker2d": "Walker2d-v4"}[env]
    return env_id, poison_type, schedule, condition, int(seed)


def _expected_keys(scope: dict, conditions: list[str]) -> set[tuple[str, str, str, str, int]]:
    return {
        (env_id, poison_type, schedule, condition, int(seed))
        for env_id in scope["environments"]
        for poison_type in scope["poison_types"]
        for schedule in scope["schedules"]
        for condition in conditions
        for seed in scope["seeds"]
    }


def _validate_rows(name: str, rows: list[dict], spec: dict, scope: dict) -> None:
    expected_rows = int(spec["expected_rows"])
    if len(rows) != expected_rows:
        raise AssertionError(f"{name}: expected {expected_rows} rows, found {len(rows)}")

    expected_step = str(spec["expected_global_step"])
    bad_steps = sorted({row.get("global_step") for row in rows if row.get("global_step") != expected_step})
    if bad_steps:
        raise AssertionError(f"{name}: unexpected global_step values {bad_steps}")

    observed = {_run_key(row) for row in rows}
    expected = _expected_keys(scope, spec["conditions"])
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        extra = sorted(observed - expected)[:10]
        raise AssertionError(f"{name}: matrix mismatch missing={missing} extra={extra}")


def validate(root: Path, manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    scope = manifest["scope"]
    messages = []

    for name, spec in manifest["experiments"].items():
        summary_paths = spec.get("summaries") or [spec["summary"]]
        rows = []
        for rel_path in summary_paths:
            rows.extend(_read_csv(root / rel_path))
        _validate_rows(name, rows, spec, scope)
        messages.append(f"OK {name}: {len(rows)} rows")

        for key in ("root", "report", "comparison_report", "visual_data"):
            if key in spec and not (root / spec[key]).exists():
                raise AssertionError(f"{name}: missing {key} path {spec[key]}")

    for rel_path in manifest.get("derived_outputs", []):
        if not (root / rel_path).exists():
            raise AssertionError(f"Missing derived output: {rel_path}")
    messages.append("OK derived outputs")

    ignored_archive = root / manifest["ignored_archive"]
    if not ignored_archive.exists():
        raise AssertionError(f"Missing ignored archive: {ignored_archive}")
    messages.append("OK ignored archive present")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", default="experiments/canonical_experiments.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    for message in validate(root, manifest_path):
        print(message)


if __name__ == "__main__":
    main()
