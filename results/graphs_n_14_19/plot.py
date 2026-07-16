import csv, random
from statistics import mean, median
from collections import defaultdict
import matplotlib.pyplot as plt

g = defaultdict(dict)
for row in csv.DictReader(open("final.csv")):
    g[row["graph_label"]][row["method"]] = row

ratios = []
per_n = defaultdict(list)
for methods in g.values():
    k, o = methods.get("kolman"), methods.get("okamoto_exact")
    if not k or not o or k["status"] != "ok" or o["status"] != "ok":
        continue
    n = int(k["n"]); r = int(k["max_congestion"]) / int(o["max_congestion"])
    ratios.append(r); per_n[n].append(r)

print(f"graphs: {len(ratios)}  mean: {mean(ratios):.3f}  median: {median(ratios):.3f}  "
      f"max: {max(ratios):.3f}  exact hits: {sum(1 for r in ratios if r==1)}/{len(ratios)}")

xs_all = [n + random.uniform(-0.25, 0.25) for n in [nn for nn in per_n for _ in per_n[nn]]]
ys_all = [r for nn in per_n for r in per_n[nn]]
plt.scatter(xs_all, ys_all, s=8, alpha=0.35)

xs = sorted(per_n)
plt.plot(xs, [mean(per_n[n]) for n in xs],   "r_", markersize=25, markeredgewidth=3, label="mean")
plt.plot(xs, [median(per_n[n]) for n in xs], "k_", markersize=25, markeredgewidth=3, label="median")

plt.axhline(1.0, color="green", ls="--", label="optimum")
plt.xlabel("n"); plt.ylabel("Kolman / optimum")
plt.title("House of Graphs, n=14-19")
plt.legend()
plt.savefig("hog_verification.png", dpi=350)
