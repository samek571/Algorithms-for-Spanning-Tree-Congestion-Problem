import csv, re, statistics
import matplotlib.pyplot as plt

# exact optimum: Kozawa-Otachi-Yamazaki Thm 3.1
def exact(parts):
    p = sorted(parts); n = sum(p)
    return n - p[1] if p[0] == 1 else 2*n - p[-1] - p[-2] - 2

ns, ratios = [], []
for row in csv.DictReader(open("custom_multipartite.csv")):
    if row["method"] != "kolman" or row["status"] != "ok":
        continue
    parts = [int(x) for x in re.findall(r"\d+", row["graph_label"])]
    ns.append(int(row["n"]))
    ratios.append(int(row["max_congestion"]) / exact(parts))

med = statistics.median(ratios)
hits = sum(1 for r in ratios if abs(r - 1.0) < 1e-9) # Using 1e-9 to avoid float precision issues

print(f"graphs: {len(ratios)}  mean ratio: {sum(ratios)/len(ratios):.2f}  max: {max(ratios):.2f}")
print(f"median: {med:.2f}  exact hits: {hits}/{len(ratios)}")

plt.scatter(ns, ratios)
plt.axhline(1.0, color="red", ls="--")
plt.xlabel("n"); plt.ylabel("Kolman / optimum")
plt.title("Complete multipartite")
plt.savefig("multipartite_verification.png", dpi=350)
