import _bootstrap  # noqa: F401

import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

from adsl.pipelines import run_experiment
from run_dissertation_campaign import build_config


ENVS = ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4"]
SCHEDULES = ["random_sparse", "bursty"]
POISON_TYPES = ["reward_poisoning", "action_perturbation", "observation_corruption"]
CONDITIONS = ["clean", "attack_none", "attack_defended"]
SEEDS = [0, 1, 2, 3, 4]


def _condition_from_config(raw: dict) -> str:
    if raw.get("controller", {}).get("enabled"):
        return "attack_defended"
    if raw.get("corruption", {}).get("enabled"):
        return "attack_none"
    return "clean"


def _key_slug(key: tuple[str, str, str, str, int]) -> str:
    return "__".join(str(part).replace("/", "_").replace("-", "_") for part in key)


def _complete_metrics(metrics_path: Path, total_steps: int) -> bool:
    if not metrics_path.exists():
        return False
    try:
        with metrics_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        max_step = max((int(float(row.get("global_step") or 0)) for row in rows), default=0)
    except Exception:
        return False
    return max_step >= total_steps


def _observed_cells(base_root: Path, total_steps: int, include_active: bool = True) -> tuple[set, set]:
    completed = set()
    active = set()
    for cfg_path in base_root.glob("*/dissertation_*/config.json"):
        try:
            raw = json.loads(cfg_path.read_text())
            key = (
                raw["env"]["id"],
                raw["corruption"]["schedule"],
                raw["corruption"]["type"],
                _condition_from_config(raw),
                int(raw["seed"]),
            )
        except Exception:
            continue
        if _complete_metrics(cfg_path.parent / "metrics.csv", total_steps):
            completed.add(key)
        elif include_active:
            active.add(key)
    return completed, active


def _all_cells(args) -> list[tuple[str, str, str, str, int]]:
    return [
        (env_id, schedule, poison_type, condition, seed)
        for env_id in args.envs
        for schedule in args.schedules
        for poison_type in args.poison_types
        for condition in args.conditions
        for seed in args.seeds
    ]


def _claim_cell(claim_root: Path, key: tuple[str, str, str, str, int]) -> Path | None:
    claim_dir = claim_root / _key_slug(key)
    try:
        claim_dir.mkdir()
    except FileExistsError:
        return None
    (claim_dir / "claimed_at_utc.txt").write_text(datetime.utcnow().isoformat())
    return claim_dir


def _worker(worker_id: int, args, campaign_root: Path, claim_root: Path, queue: mp.Queue) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    while True:
        key = queue.get()
        if key is None:
            return
        completed, active = _observed_cells(
            Path(args.base_root),
            args.total_steps,
            include_active=not args.ignore_incomplete_active,
        )
        if key in completed or key in active:
            continue
        claim_dir = _claim_cell(claim_root, key)
        if claim_dir is None:
            continue
        env_id, schedule, poison_type, condition, seed = key
        try:
            config = build_config(
                env_id=env_id,
                schedule=schedule,
                poison_type=poison_type,
                condition=condition,
                seed=seed,
                output_root=str(campaign_root),
                total_steps=args.total_steps,
            )
            config.detector.window_length = int(args.window_length)
            config.detector.warmup_steps = int(args.warmup_steps)
            run_dir = run_experiment(config)
            (claim_dir / "completed_run_dir.txt").write_text(str(run_dir))
            print(f"worker={worker_id} COMPLETE {key} {run_dir}", flush=True)
        except Exception as exc:
            (claim_dir / "failed.txt").write_text(repr(exc))
            print(f"worker={worker_id} FAILED {key}: {exc!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", default="results/dissertation/parameterized_runs")
    parser.add_argument("--campaign-name", default="window200_rerun_coordinated_pool")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--window-length", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--envs", nargs="+", default=ENVS)
    parser.add_argument("--schedules", nargs="+", default=SCHEDULES)
    parser.add_argument("--poison-types", nargs="+", default=POISON_TYPES)
    parser.add_argument("--conditions", nargs="+", default=["attack_defended"])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--ignore-incomplete-active",
        action="store_true",
        help="Do not treat incomplete prior output directories as active work.",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    campaign_root = base_root / f"{args.campaign_name}_workers{args.workers}_{stamp}"
    claim_root = campaign_root / "_claims"
    claim_root.mkdir(parents=True, exist_ok=True)
    manifest = vars(args).copy()
    manifest["campaign_root"] = str(campaign_root)
    manifest["created_utc"] = datetime.utcnow().isoformat()
    (campaign_root / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2))

    completed, active = _observed_cells(
        base_root,
        args.total_steps,
        include_active=not args.ignore_incomplete_active,
    )
    cells = [cell for cell in _all_cells(args) if cell not in completed and cell not in active]
    queue: mp.Queue = mp.Queue()
    for cell in cells:
        queue.put(cell)
    for _ in range(args.workers):
        queue.put(None)

    print(
        f"starting workers={args.workers} queued={len(cells)} "
        f"completed={len(completed)} active={len(active)} root={campaign_root}",
        flush=True,
    )
    workers = [
        mp.Process(target=_worker, args=(idx, args, campaign_root, claim_root, queue))
        for idx in range(args.workers)
    ]
    for process in workers:
        process.start()
        time.sleep(0.2)
    for process in workers:
        process.join()


if __name__ == "__main__":
    main()
