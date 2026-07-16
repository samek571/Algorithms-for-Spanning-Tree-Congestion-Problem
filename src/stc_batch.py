from __future__ import annotations

import argparse
import csv
import signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import networkx as nx

from src import stc_okamoto, stc_kolman
from src.simple_cache import SimpleCache
from src.stc_core import compute_tree_congestion, graph_statistics, CongestionResult
from src.stc_io import GraphInstance, load_graph_folder


@dataclass
class ExperimentRow:
    graph_label: str
    graph_family: str
    method: str
    n: int
    m: int
    max_degree: int
    max_congestion: int | None
    runtime_seconds: float | None
    status: str

@dataclass
class MethodRun:
    tree: nx.Graph
    runtime_seconds: float
    max_congestion: int
    congestion: CongestionResult | None = None

METHOD_TIMEOUT_SECONDS = 345600 #the wallclock has been set on einfracz and this value was spiked by 3 extra digits so it would not cause any issue but i didnt want to change code much so rather fix it like this

class _MethodTimeout(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _MethodTimeout()

def _run_method(G: nx.Graph, method: str, *, okamoto_upper_bound: int | None = None) -> MethodRun:
    if method == "okamoto_exact":
        result = stc_okamoto.okamoto_exact_stc_simple(G)
        congestion = compute_tree_congestion(G, result.tree)
        if congestion.max_congestion != result.optimum_congestion:
            raise RuntimeError(
                f"Okamoto DP/tree mismatch: dp={result.optimum_congestion}, measured={congestion.max_congestion}"
            )
        return MethodRun(
            tree=result.tree,
            runtime_seconds=result.solve_time_seconds,
            max_congestion=result.optimum_congestion,
            congestion=compute_tree_congestion(G, result.tree),
        )
    if method == "kolman":
        result = stc_kolman.kolman_main(G)
        return MethodRun(
            tree=result.tree,
            runtime_seconds=result.solve_time_seconds,
            max_congestion=result.congestion.max_congestion,
            congestion=result.congestion,
        )

    raise KeyError(f"Unknown method: {method}")


def _make_row(instance: GraphInstance, method: str) -> ExperimentRow:
    stats = graph_statistics(instance.graph)

    return ExperimentRow(
        graph_label=instance.graph_label,
        graph_family=instance.graph_family,
        method=method,
        n=int(stats["n"]),
        m=int(stats["m"]),
        max_degree=int(stats["max_degree"]),
        max_congestion=None,
        runtime_seconds=None,
        status="pending",
    )

def run_method_on_graph(
        instance: GraphInstance,
        method: str,
        *,
        okamoto_upper_bound: int | None = None,
        timeout_seconds: int = METHOD_TIMEOUT_SECONDS,
) -> ExperimentRow:
    row = _make_row(instance, method)

    cached_data = cache.get(instance.graph, method)
    if cached_data:
        row.max_congestion = cached_data["max_congestion"]
        row.runtime_seconds = cached_data["runtime_seconds"]
        row.status = cached_data["status"]
        print("we hit a cache")
        return row

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        run = _run_method(instance.graph, method, okamoto_upper_bound=okamoto_upper_bound)
        signal.setitimer(signal.ITIMER_REAL, 0)

        row.max_congestion = run.max_congestion
        row.runtime_seconds = run.runtime_seconds
        row.status = "ok"

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)

    except _MethodTimeout:
        row.status = "TLE"
        row.runtime_seconds = float(timeout_seconds)
    except ValueError as e:
        msg = str(e)
        if "exponential" in msg.lower() or "too large" in msg.lower() or "n <=" in msg.lower():
            row.status = "skipped_too_large"
        else:
            row.status = "invalid_graph"
    except RuntimeError as e:
        msg = str(e)
        if "SDP" in msg or "ARV" in msg:
            row.status = "sdp_failed"
        else:
            row.status = "failed"
    except Exception:
        row.status = "failed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

    cache.put(instance.graph, method, {
        "max_congestion": row.max_congestion,
        "runtime_seconds": row.runtime_seconds,
        "status": row.status
    })

    return row

def run_graph_instance(
        instance: GraphInstance,
        *,
        input_root: str | Path | None = None,
        timeout_seconds: int = METHOD_TIMEOUT_SECONDS,
        kolman_timeout_seconds: int | None = None,
) -> list[ExperimentRow]:

    rows: list[ExperimentRow] = []
    if kolman_timeout_seconds is None:
        kolman_timeout_seconds = timeout_seconds

    kolman_row = run_method_on_graph(
        instance,
        "kolman",
        input_root=input_root,
        timeout_seconds=kolman_timeout_seconds,
    )
    rows.append(kolman_row)

    okamoto_upper_bound = None
    if kolman_row.status == "ok" and kolman_row.max_congestion is not None:
        okamoto_upper_bound = kolman_row.max_congestion

    okamoto_row = run_method_on_graph(
        instance,
        "okamoto_exact",
        input_root=input_root,
        okamoto_upper_bound=okamoto_upper_bound,
    )
    rows.append(okamoto_row)

    if (okamoto_row.status == "ok"
        and kolman_row.status == "ok"
        and okamoto_row.max_congestion is not None
        and kolman_row.max_congestion is not None
        and okamoto_row.max_congestion > kolman_row.max_congestion):
        raise RuntimeError(
            f"Exact result worse than koolman on "
            f"{instance.graph_family}/{instance.graph_label}: "
            f"exact={okamoto_row.max_congestion}, kolman={kolman_row.max_congestion}"
        )
    return rows


def write_results_csv(rows: Sequence[ExperimentRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["graph_label", "graph_family","method", "n", "m", "max_degree", "max_congestion", "runtime_seconds", "status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            if data["runtime_seconds"] is not None:
                data["runtime_seconds"] = round(data["runtime_seconds"], 2)
            writer.writerow(data)

def run_graph_folder(
        input_root: str | Path,
        output_csv: str | Path,
        *,
        normalize_labels_flag: bool = False,
        timeout_seconds: int = METHOD_TIMEOUT_SECONDS,
) -> list[ExperimentRow]:
    input_root = Path(input_root)
    instances = load_graph_folder(input_root, normalize_labels_flag=normalize_labels_flag,
    )

    all_rows: list[ExperimentRow] = []
    total_graphs = len(instances)
    print(f"Discovered {total_graphs} graph files under {input_root}")
    for i, instance in enumerate(instances, start=1):
        print(f"[{i}/{total_graphs}] {instance.graph_family}/{instance.graph_label}")

        graph_rows = run_graph_instance(
            instance,
            input_root=input_root,
            timeout_seconds=timeout_seconds,
        )

        for row in graph_rows:
            runtime_str = "None" if row.runtime_seconds is None else f"{row.runtime_seconds:.2f}s"
            print(
                f"    method={row.method} "
                f"status={row.status} "
                f"cong={row.max_congestion} "
                f"time={runtime_str}"
            )

        all_rows.extend(graph_rows)


    write_results_csv(all_rows, output_csv)
    return all_rows


if __name__ == "__main__":
    import os
    project_root = Path(__file__).resolve().parents[1]
    CACHE_DIR = project_root / "results" / "json"

    array_id = os.environ.get("PBS_ARRAYID") or os.environ.get("PBS_ARRAY_INDEX")
    job_id = os.environ.get("PBS_JOBID", "").split(".")[0]
    if array_id is not None:
        cache_filename = f"cache_array_{array_id}.json"
    elif job_id:
        cache_filename = f"cache_job_{job_id}.json"
    else:
        cache_filename = "cache.json"

    cache = SimpleCache(CACHE_DIR / cache_filename)
    print(f"Using cache file: {CACHE_DIR / cache_filename}")

    default_input = project_root / "data"
    default_output = project_root / "results" / "csv" / "batch_results.csv" #deprecated but whatever
    default_drawings = project_root / "results" / "drawings" #useless

    parser = argparse.ArgumentParser(description="Run STC methods on a folder of adjacency-list graphs.")
    parser.add_argument("input_root", nargs="?", default=str(default_input), help="Root directory with graph files")
    parser.add_argument("output_csv", nargs="?", default=str(default_output), help="Where to write CSV results")
    parser.add_argument("--normalize-labels", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=METHOD_TIMEOUT_SECONDS, help="Per-method timeout in seconds")
    args = parser.parse_args()

    print(f"Input root: {args.input_root}")
    print(f"Output CSV: {args.output_csv}")
    print("Starting batch run...")

    rows = run_graph_folder(
        args.input_root,
        args.output_csv,
        normalize_labels_flag=args.normalize_labels,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
