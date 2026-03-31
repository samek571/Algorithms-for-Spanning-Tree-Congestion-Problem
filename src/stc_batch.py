from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence
import networkx as nx

from src import stc_okamoto, stc_kolman, stc_core
from stc_core import compute_tree_congestion, graph_statistics
from stc_draw import draw_graph_with_tree
from stc_io import GraphInstance, load_graph_folder


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
    congestion: object | None = None


def _draw_output_path(drawings_root: str | Path, input_root: str | Path, instance: GraphInstance, method: str) -> Path:
    drawings_root = Path(drawings_root)
    input_root = Path(input_root)

    subdir = drawings_root / input_root.name
    if instance.graph_family != ".":
        subdir = subdir / instance.graph_family
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir / f"drawing_of_{instance.graph_label}__{method}.png"


def _run_method(G: nx.Graph, method: str) -> MethodRun:
    # baseline just to verify stuff
    if method == "bfs":
        start = time.perf_counter()
        T = stc_core.bfs_tree(G)
        runtime = time.perf_counter() - start
        congestion = compute_tree_congestion(G, T)
        return MethodRun(tree=T,runtime_seconds=runtime,max_congestion=congestion.max_congestion,congestion=congestion)

    if method == "okamoto_exact":
        result = stc_okamoto.okamoto_exact_stc_simple(G)
        return MethodRun(
            tree=result.tree,
            runtime_seconds=result.solve_time_seconds,
            max_congestion=result.optimum_congestion,
            congestion=compute_tree_congestion(G, result.tree),
        )

    if method == "kolman_exact_cut":
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
        draw: bool = True,
        drawings_root: str | Path | None = None,
        input_root: str | Path | None = None,
) -> ExperimentRow:
    row = _make_row(instance, method)

    try:
        run = _run_method(instance.graph, method)
        row.max_congestion = run.max_congestion
        row.runtime_seconds = run.runtime_seconds
        row.status = "ok"

        if draw and drawings_root is not None and input_root is not None:
            congestion = run.congestion
            if congestion is None:
                congestion = compute_tree_congestion(instance.graph, run.tree)

            out_path = _draw_output_path(drawings_root, input_root, instance, method)
            draw_graph_with_tree(
                instance.graph,
                run.tree,
                congestion=congestion,
                title=f"{instance.graph_label} [{method}] cong={row.max_congestion}",
                save_path=out_path,
                show=False,
            )
    except ValueError as e:
        msg = str(e)
        if "exponential" in msg.lower() or "too large" in msg.lower() or "n <=" in msg.lower():
            row.status = "skipped_too_large"
        else:
            row.status = "invalid_graph"
    except Exception:
        row.status = "failed"

    return row


def run_graph_instance(
        instance: GraphInstance,
        *,
        draw: bool = True,
        drawings_root: str | Path | None = None,
        input_root: str | Path | None = None,
) -> list[ExperimentRow]:
    rows: list[ExperimentRow] = []
    for method in ["bfs", "kolman_exact_cut", "okamoto_exact"]:
        rows.append(
            run_method_on_graph(
                instance,
                method,
                draw=draw,
                drawings_root=drawings_root,
                input_root=input_root,
            )
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
            writer.writerow(asdict(row))

def run_graph_folder(
        input_root: str | Path,
        output_csv: str | Path,
        *,
        drawings_root: str | Path | None = None,
        normalize_labels_flag: bool = False,
        draw: bool = True,
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
            draw=draw,
            drawings_root=drawings_root,
            input_root=input_root,
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
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "data"
    default_output = project_root / "results" / "csv" / "batch_results.csv"
    default_drawings = project_root / "results" / "drawings"

    parser = argparse.ArgumentParser(description="Run STC methods on a folder of adjacency-list graphs.")
    parser.add_argument("input_root", nargs="?", default=str(default_input), help="Root directory with graph files")
    parser.add_argument("output_csv", nargs="?", default=str(default_output), help="Where to write CSV results")
    parser.add_argument("--drawings-root", default=str(default_drawings), help="Root directory for saved drawings")
    parser.add_argument("--normalize-labels", action="store_true")
    parser.add_argument("--draw", action="store_true")

    args = parser.parse_args()

    print(f"Input root: {args.input_root}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Drawings root: {args.drawings_root}")
    print(f"Draw enabled: {args.draw}")
    print("Starting batch run...")

    rows = run_graph_folder(
        args.input_root,
        args.output_csv,
        drawings_root=args.drawings_root,
        normalize_labels_flag=args.normalize_labels,
        draw=args.draw,
    )
    print(f"Wrote {len(rows)} detailed rows to {args.output_csv}")
