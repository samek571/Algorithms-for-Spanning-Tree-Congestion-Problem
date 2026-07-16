"""
Foster census (cubic, Delta=3): congestion and runtime vs n.

We measure wall-clock runtime against n and fit a power law t = a * n^b by
ordinary least squares in log-log space. This is a DESCRIPTIVE summary of
these specific runs on this specific hardware, not a confirmed complexity
bound.
SDP => high-degree polynomial cost in n (O(n^2) variables, O(n^3) triangle constraints)
a handful of points spanning roughly one order of magnitude in n cannot distinguish between polynomials of different degree, nor explain anythin.

The fit is reported as description, not confirmation.
"""
import csv
import math
import matplotlib.pyplot as plt


def load_rows(path="results.csv"):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows]


def completed_only(rows):
    if rows and "status" in rows[0]:
        ok = [r for r in rows if r["status"] == "ok"]
    else:
        ok = rows
    return ok


def fit_power_law(ns, ts):
    """
    Fits t = a * n^b via ordinary least squares on (log n, log t).
    Returns (a, b, r_squared)
    """
    xs = [math.log(n) for n in ns]
    ys = [math.log(t) for t in ts]
    m = len(xs)
    x_bar = sum(xs) / m
    y_bar = sum(ys) / m
    Sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    Sxx = sum((x - x_bar) ** 2 for x in xs)
    b = Sxy / Sxx
    intercept = y_bar - b * x_bar
    a = math.exp(intercept)
    pred = [intercept + b * x for x in xs]
    rss = sum((y - p) ** 2 for y, p in zip(ys, pred))
    tss = sum((y - y_bar) ** 2 for y in ys)
    r2 = 1 - rss / tss if tss > 0 else float("nan")
    return a, b, r2


def main():
    rows = load_rows()
    n_total = len(rows)
    rows = completed_only(rows)
    n_done = len(rows)

    data = sorted(
        (int(r["n"]), int(r["max_congestion"]), float(r["runtime_seconds"]))
        for r in rows
    )
    ns = [d[0] for d in data]
    cs = [d[1] for d in data]
    ts = [d[2] for d in data]

    status_note = (f"{n_done}/{n_total} instances completed"
                   if n_done != n_total else f"{n_done} instances")
    print(f"Foster census: {status_note}")
    print(f"n range: {min(ns)}-{max(ns)}")
    print(f"runtime range: {min(ts):.1f}s - {max(ts):.1f}s "
          f"({max(ts)/3600:.2f} hours)")

    # --- congestion vs n ---
    plt.figure(figsize=(9, 5.5))
    plt.scatter(ns, cs, s=30)
    plt.xlabel("n")
    plt.ylabel("congestion")
    plt.title(f"Foster census (cubic, $\\Delta=3$): congestion vs $n$ "
              f"({status_note})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("foster_congestion.png", dpi=350)
    plt.close()

    # --- runtime vs n, log-log, with explicit least-squares fit ---
    a, b, r2 = fit_power_law(ns, ts)
    print("\nleast-squares fit in log-log space (t = a * n^b):")
    print(f"  slope     b        = {b:.3f}")
    print(f"  constant  a        = {a:.3e}")
    print(f"  R^2 (log space)    = {r2:.3f}")
    print("  descriptive summary of these runs; NOT a claimed complexity bound.")

    plt.figure(figsize=(9, 5.5))
    plt.scatter(ns, ts, s=30, color="tab:red", label="measured")
    ns_sorted = sorted(ns)
    fit_ts = [a * n ** b for n in ns_sorted]
    plt.plot(ns_sorted, fit_ts, color="black", ls="--",
              label=f"fit: $t \\sim n^{{{b:.2f}}}$ ($R^2={r2:.2f}$)")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("runtime (s), log scale")
    plt.title(f"Foster census: runtime vs $n$ ({status_note})")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("foster_runtime.png", dpi=350, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
