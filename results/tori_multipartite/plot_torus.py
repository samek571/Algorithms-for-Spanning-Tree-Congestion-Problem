import csv, re, statistics
import matplotlib.pyplot as plt

# exact optimum: Kozawa-Otachi-Yamazaki Thm 4.2
def exact(m, n):
    return 2 * min(m, n)

ns, ratios = [], []
for row in csv.DictReader(open("custom_torus.csv")):
    if row["method"] != "kolman" or row["status"] != "ok":
        continue
    m, n = map(int, re.search(r"(\d+)x(\d+)", row["graph_label"]).groups())
    ns.append(int(row["n"]))
    ratios.append(int(row["max_congestion"]) / exact(m, n))

med = statistics.median(ratios)
hits = sum(1 for r in ratios if abs(r - 1.0) < 1e-9)

print(f"graphs: {len(ratios)}  mean ratio: {sum(ratios)/len(ratios):.5f}  max: {max(ratios):.5f}")
print(f"median: {med:.2f}  exact hits: {hits}/{len(ratios)}")

plt.scatter(ns, ratios)
plt.axhline(1.0, color="red", ls="--")
plt.xlabel("n"); plt.ylabel("Kolman / optimum")
plt.title("2D torus")
plt.savefig("torus_verification.png", dpi=350)
