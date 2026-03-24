from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from statistics import mean
from pathlib import Path
from typing import Optional, Sequence
import networkx as nx
import stc_algo as algo
from stc_io import GraphInstance, load_graph_folder
from stc_core import compute_tree_congestion, graph_statistics
from stc_draw import draw_graph_with_tree


@dataclass
class ExperimentRow:
    graph_label: str
    graph_family: str
    source_path: str
    method: str
    seed: int
    n: int
    m: int
    min_degree: int
    max_degree: int
    avg_degree: float
    degree_variance: float
    degree_stddev: float
    density: float
    n_articulation_points: int
    n_bridges: int
    n_leaves: int
    is_regular: bool
    max_congestion: int | None
    runtime_seconds: float | None
    status: str
    is_exact: bool = False
    exact_reference: int | None = None
    ratio_to_exact: float | None = None
    notes: str = ""

METHOD_TO_FUNCTION = {
    "bfs": "bfs_tree",
    "dfs": "dfs_tree",
    "wilson": "uniform_spanning_tree",
    "uniform_random": "uniform_spanning_tree",
    "hub_first": "hub_first_tree",
    "cut_recursive": "cut_recursive_tree",
    "okamoto_exact": "okamoto_exact_stc_simple",
    "kolman_exact_cut": "kolman_congspantree_simple",
}

EXACT_METHODS = {"okamoto_exact"}
DEFAULT_METHODS = ["bfs", "dfs", "wilson", "hub_first", "cut_recursive"]


def _resolve_method_builder(method: str):
    if method not in METHOD_TO_FUNCTION:
        raise KeyError(f"Unknown method: {method}")
    function_name = METHOD_TO_FUNCTION[method]
    if not hasattr(algo, function_name):
        raise KeyError(f"Method '{method}' is unavailable. Missing function '{function_name}' in stc_algo.py")
    return getattr(algo, function_name)

def available_methods() -> list[str]:
    methods = []
    for method in METHOD_TO_FUNCTION:
        try:
            _resolve_method_builder(method)
            methods.append(method)
        except KeyError:
            pass
    return methods


def _draw_output_path(drawings_root: str | Path, instance: GraphInstance, method: str, seed: int) -> Path:
    drawings_root = Path(drawings_root)
    subdir = drawings_root / instance.graph_family
    subdir.mkdir(parents=True, exist_ok=True)
    suffix = f"__seed{seed}" if method in {"wilson", "uniform_random"} else ""
    return subdir / f"{instance.graph_label}__{method}{suffix}.png"


def _is_exact_method(method: str) -> bool:
    return method in EXACT_METHODS


def _run_method(G: nx.Graph, method: str, seed: int):
    builder = _resolve_method_builder(method)

    # tree-building methods
    if method in {"bfs", "dfs", "hub_first", "cut_recursive"}:
        start = time.perf_counter()
        T = builder(G)
        runtime = time.perf_counter() - start
        cong = compute_tree_congestion(G, T)
        return T, cong, runtime, None

    if method in {"wilson", "uniform_random"}:
        start = time.perf_counter()
        T = builder(G, seed=seed)
        runtime = time.perf_counter() - start
        cong = compute_tree_congestion(G, T)
        return T, cong, runtime, None

    # advanced result containers
    if method == "okamoto_exact":
        result = builder(G)
        T = result.tree
        cong = compute_tree_congestion(G, T)
        return T, cong, result.solve_time_seconds, result.optimum_congestion

    if method == "kolman_exact_cut":
        result = builder(G)
        T = result.tree
        cong = result.congestion
        return T, cong, result.solve_time_seconds, None

    raise KeyError(f"Unknown method: {method}")

def _make_row(instance: GraphInstance, method: str, seed: int) -> ExperimentRow:
    G = instance.graph
    stats = graph_statistics(G)

    ExperimentRow(
        graph_label=instance.graph_label,
        graph_family=instance.graph_family,
        source_path=instance.source_path,
        method=method,
        seed=seed,
        n=int(stats["n"]),
        m=int(stats["m"]),
        min_degree=int(stats["min_degree"]),
        max_degree=int(stats["max_degree"]),
        avg_degree=float(stats["avg_degree"]),
        degree_variance=float(stats["degree_variance"]),
        degree_stddev=float(stats["degree_stddev"]),
        density=float(stats["density"]),
        n_articulation_points=int(stats["n_articulation_points"]),
        n_bridges=int(stats["n_bridges"]),
        n_leaves=int(stats["n_leaves"]),
        is_regular=bool(stats["is_regular"]),
        max_congestion=None,
        runtime_seconds=None,
        status="pending",
        is_exact=_is_exact_method(method),
    )

def run_method_on_graph(
        instance: GraphInstance,
        method: str,
        *,
        seed: int = 0,
        draw: bool = False,
        drawings_root: str | Path | None = None,
        wilson_repeats: int
) -> ExperimentRow:
    G = instance.graph
    row = _make_row(instance, method, seed)
    try:
        T, cong, runtime, exact_reference = _run_method(G, method, seed)
        row.max_congestion = cong.max_congestion
        row.runtime_seconds = runtime
        row.status = "ok"
        row.exact_reference = exact_reference

        if draw and drawings_root is not None:
            out_path = _draw_output_path(drawings_root, instance, method, seed)
            draw_graph_with_tree(
                G,
                T,
                congestion=cong,
                title=f"{instance.graph_label} [{method}] cong={cong.max_congestion}",
                save_path=out_path,
                show=False,
            )
    except ValueError as e:
        msg = str(e)
        if "exponential" in msg.lower() or "too large" in msg.lower() or "n <=" in msg.lower():
            row.status = "skipped_too_large"
        else:
            row.status = "invalid_graph"
        row.notes = msg
    except Exception as e:
        row.status = "failed"
        row.notes = f"{type(e).__name__}: {e}"

    return row


def run_graph_instance(
    instance: GraphInstance,
    *,
    methods: Optional[Sequence[str]] = None,
    seed: int = 0,
    draw: bool = False,
    drawings_root: str | Path | None = None,
    wilson_repeats: int = 30,
) -> list[ExperimentRow]:
    if methods is None:
        methods = [m for m in DEFAULT_METHODS if m in available_methods()]

    rows: list[ExperimentRow] = []
    exact_value: int | None = None

    # If exact is requested, run it first so ratios can be filled in.
    ordered_methods = list(methods)
    if "okamoto_exact" in ordered_methods:
        ordered_methods.remove("okamoto_exact")
        ordered_methods.insert(0, "okamoto_exact")

    for method in ordered_methods:
        if method in {"wilson", "uniform_random"}:
            for offset in range(wilson_repeats):
                row = run_method_on_graph(
                    instance,
                    method,
                    seed=seed + offset,
                    draw=draw,
                    drawings_root=drawings_root,
                    wilson_repeats = wilson_repeats
                )
                rows.append(row)
        else:
            row = run_method_on_graph(
                instance,
                method,
                seed=seed,
                draw=draw,
                drawings_root=drawings_root,
                wilson_repeats = wilson_repeats
            )
            rows.append(row)
            if method == "okamoto_exact" and row.status == "ok":
                exact_value = row.exact_reference if row.exact_reference is not None else row.max_congestion

    if exact_value is not None:
        for row in rows:
            row.exact_reference = exact_value
            if row.max_congestion is not None and exact_value > 0:
                row.ratio_to_exact = row.max_congestion / exact_value
            elif row.max_congestion == 0 and exact_value == 0:
                row.ratio_to_exact = 1.0

    return rows


def write_results_csv(rows: Sequence[ExperimentRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph_label", "graph_family", "source_path", "method", "seed",
        "n", "m", "min_degree", "max_degree", "avg_degree",   "degree_variance", "degree_stddev", "density",
        "n_articulation_points", "n_bridges", "n_leaves", "is_regular",
        "max_congestion", "max_congestion",
        "runtime_seconds", "status", "is_exact", "exact_reference",  "exact_reference",
        "ratio_to_exact", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

def _family_filename(graph_family: str) -> str:
    safe = graph_family.replace("/", "__").replace("\\", "__")
    if not safe:
        safe = "root"
    return safe


def write_family_summary_csvs(rows: Sequence[ExperimentRow], summaries_root: str | Path) -> None:
    summaries_root = Path(summaries_root)
    summaries_root.mkdir(parents=True, exist_ok=True)

    by_family: dict[str, list[ExperimentRow]] = {}
    for row in rows:
        by_family.setdefault(row.graph_family, []).append(row)

    combined_rows: list[dict[str, object]] = []

    for family, family_rows in by_family.items():
        by_method: dict[str, list[ExperimentRow]] = {}
        for row in family_rows:
            by_method.setdefault(row.method, []).append(row)

        summary_rows: list[dict[str, object]] = []
        for method, method_rows in sorted(by_method.items()):
            ok_rows = [r for r in method_rows if r.status == "ok" and r.max_congestion is not None]
            summary = {
                "graph_family": family,
                "method": method,
                "runs_total": len(method_rows),
                "runs_ok": len(ok_rows),
                "distinct_graphs": len({r.graph_label for r in method_rows}),
                "mean_congestion": mean([r.max_congestion for r in ok_rows]) if ok_rows else None,
                "min_congestion": min([r.max_congestion for r in ok_rows]) if ok_rows else None,
                "max_congestion": max([r.max_congestion for r in ok_rows]) if ok_rows else None,
                "mean_runtime_seconds": mean([r.runtime_seconds for r in ok_rows if r.runtime_seconds is not None]) if ok_rows else None,
                "mean_ratio_to_exact": mean([r.ratio_to_exact for r in ok_rows if r.ratio_to_exact is not None]) if ok_rows and any(r.ratio_to_exact is not None for r in ok_rows) else None,
            }
            summary_rows.append(summary)
            combined_rows.append(summary)

        out_path = summaries_root / f"{_family_filename(family)}__summary.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "graph_family", "method", "runs_total", "runs_ok", "distinct_graphs",
                "mean_congestion", "min_congestion", "max_congestion",
                "mean_runtime_seconds", "mean_ratio_to_exact",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)

    if combined_rows:
        out_path = summaries_root / "all_families_summary.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "graph_family", "method", "runs_total", "runs_ok", "distinct_graphs",
                "mean_congestion", "min_congestion", "max_congestion",
                "mean_runtime_seconds", "mean_ratio_to_exact",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in combined_rows:
                writer.writerow(row)


def run_graph_folder(
        input_root: str | Path,
        output_csv: str | Path,
        *,
        drawings_root: str | Path | None = None,
        summaries_root: str | Path | None = None,
        recursive: bool = True,
        methods: Optional[Sequence[str]] = None,
        seed: int = 0,
        require_connected: bool = True,
        largest_component_only: bool = False,
        normalize_labels_flag: bool = False,
        draw: bool = False,
        wilson_repeats: int = 30,
) -> list[ExperimentRow]:
    instances = load_graph_folder(
        input_root,
        recursive=recursive,
        require_connected=require_connected,
        largest_component_only=largest_component_only,
        normalize_labels_flag=normalize_labels_flag,
    )

    all_rows: list[ExperimentRow] = []
    for instance in instances:
        rows = run_graph_instance(
            instance,
            methods=methods,
            seed=seed,
            draw=draw,
            drawings_root=drawings_root,
            wilson_repeats=wilson_repeats,
        )
        all_rows.extend(rows)

    write_results_csv(all_rows, output_csv)
    if summaries_root is None:
        summaries_root = Path(output_csv).parent / "family_summaries"
    write_family_summary_csvs(all_rows, summaries_root)
    return all_rows


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "data"
    default_output = project_root / "results" / "csv" / "batch_results.csv"
    default_drawings = project_root / "results" / "drawings"
    default_summaries = project_root / "results" / "csv" / "family_summaries"

    parser = argparse.ArgumentParser(description="Run STC methods on a folder of adjacency-list graphs.")
    parser.add_argument("input_root", nargs="?", default=str(default_input), help="Root directory with graph files")
    parser.add_argument("output_csv", nargs="?", default=str(default_output), help="Where to write CSV results")
    parser.add_argument("--drawings-root", default=str(default_drawings), help="Root directory for saved drawings")
    parser.add_argument("--summaries-root", default=str(default_summaries), help="Directory for family summary CSVs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=None, help="Methods to run")
    parser.add_argument("--wilson-repeats", type=int, default=30)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--largest-component-only", action="store_true")
    parser.add_argument("--normalize-labels", action="store_true")
    parser.add_argument("--draw", action="store_true")

    args = parser.parse_args()

    rows = run_graph_folder(
        args.input_root,
        args.output_csv,
        drawings_root=args.drawings_root,
        summaries_root=args.summaries_root,
        recursive=not args.no_recursive,
        methods=args.methods,
        seed=args.seed,
        require_connected=not args.largest_component_only,
        largest_component_only=args.largest_component_only,
        normalize_labels_flag=args.normalize_labels,
        draw=args.draw,
        wilson_repeats=args.wilson_repeats,
    )
    print(f"Wrote {len(rows)} detailed rows to {args.output_csv}")
    print(f"Wrote family summaries under {args.summaries_root}")