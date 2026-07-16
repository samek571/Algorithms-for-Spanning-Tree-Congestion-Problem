# Algorithms for the Spanning Tree Congestion Problem

Implementation accompanying the bachelor thesis *Algorithms for Spanning Tree
Congestion Problem* (Samuel Vaško, Charles University, 2026).

The **spanning tree congestion (STC)** of a connected graph is the minimum, over
all spanning trees, of the largest fundamental cut induced by a tree edge.
This repository implements and compares two algorithms for it:

- an **exact** dynamic program (Okamoto et al.), `O*(3^n)`, used as ground truth
  on small instances;
- **Kolman's** polynomial-time recursive `O(Δ log^{3/2} n)` **approximation**,
  whose balanced-cut oracle is built on the Arora–Rao–Vazirani SDP relaxation.

## Requirements

- Python 3.10+
- [NetworkX](https://networkx.org/) — graph manipulation
- [CVXPY](https://www.cvxpy.org/) with the **SCS** solver — the ARV semidefinite program
- NumPy

```bash
python -m venv venv
source venv/bin/activate
pip install networkx cvxpy numpy
```

## Repository layout

```
src/
  stc_core.py      shared primitives: congestion via LCA, cut-size table, DSU, validation
  stc_okamoto.py   exact DP (decision DP + binary search over cut sizes)
  stc_kolman.py    Kolman's recursive construction (CongSpanTree)
  stc_arv.py       ARV SDP balanced-cut oracle (CVXPY/SCS + projection rounding)
  stc_batch.py     experiment runner: runs both methods over a folder, writes CSV
  stc_io.py        graph loading from adjacency-list files
  simple_cache.py  content-hash result cache
data/              graph instances as adjacency-list files, grouped by family
results/           output CSVs and the JSON cache
```

## Input format

Graphs are adjacency-list text files (`.lst` or `.txt`), one line per vertex:

```
0: 1 2 3
1: 0 2
2: 0 1
3: 0
```

`iter_graph_files` picks up every `.lst`/`.txt` under the input root recursively;
the sub-folder path becomes the graph's *family* label in the output.

## Running

From the repository root:

```bash
python -m src.stc_batch <input_dir> <output_csv>
```

Example — run both methods over everything in `data/` and write `results/csv/batch_results.csv`:

```bash
python -m src.stc_batch data/ results/csv/batch_results.csv
```

Options:

| Flag | Meaning |
|------|---------|
| `--timeout-seconds N` | per-method wall-clock limit (default: 96 h); exceeding it yields status `TLE` |
| `--normalize-labels`  | relabel vertices to a canonical integer ordering on load |

For each graph the runner executes **both** methods (`kolman`, then `okamoto_exact`)
and appends two rows to the CSV.

## Output

The CSV has one row per (graph, method) with columns:

| Column | Meaning |
|--------|---------|
| `graph_label`     | file name (without extension) |
| `graph_family`    | sub-folder path under the input root |
| `method`          | `kolman` or `okamoto_exact` |
| `n`, `m`          | vertices, edges |
| `max_degree`      | maximum degree Δ |
| `max_congestion`  | congestion of the produced/optimal tree (empty if not `ok`) |
| `runtime_seconds` | wall-clock time for that method |
| `status`          | see below |

**Status codes:**

| Status | Meaning |
|--------|---------|
| `ok`                | completed; `max_congestion` is valid |
| `TLE`               | exceeded the wall-clock timeout |
| `skipped_too_large` | instance exceeds the method's size limit (the exact DP is capped at `n ≤ 19`) |
| `sdp_failed`        | the ARV semidefinite solver returned no usable embedding |
| `invalid_graph`     | input was disconnected or otherwise unusable |
| `failed`            | any other error |

## Caching

Results are cached by **graph content**, not by filename: a graph is hashed from
its (sorted) vertex and edge sets, so the same graph appearing under several
folders is computed only once, re-labeled identical graph gets marked as not identical and cache is not triggered. Re-running a completed batch is therefore cheap —
finished (graph, method) pairs are served from the cache. The cache lives at
`results/json/cache.json`; on the PBS cluster a per-job/per-array-task cache file
is used automatically to avoid write contention.

Note: the cache keys on labelled content, **not** isomorphism — two isomorphic
graphs with different vertex labels are treated as distinct.

## Reproducibility

- All instances are committed as adjacency-list files under `data/`.
- The random projection directions in the ARV rounding are seeded.
- Canonical run outputs and the plotting scripts that consume them are committed
  alongside the code.

## A note on the ARV oracle

`stc_arv.py` solves the **full** ARV SDP relaxation (unit vectors + triangle
inequalities + spreading constraint) but uses a **simplified rounding** (repeated
random projection + balanced prefix sweep) rather than the full ARV procedure
(seed-set selection, pair refinement, Fréchet-embedding sweep). The oracle always
returns a valid 2/3-balanced cut, but the proven `O(√log n)` cut-quality bound
attaches to the full rounding, not to this simplification. See the thesis,
Section 3.2, for details.
