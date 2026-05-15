import json, hashlib
from pathlib import Path

class SimpleCache:
    def __init__(self, cache_path="results/cache.json"):
        self.path = Path(cache_path)
        self.cache = {}
        if self.path.exists():
            try:
                self.cache = json.loads(self.path.read_text())
            except: self.cache = {}

    def _get_graph_hash(self, G):
        edges = sorted(tuple(sorted((str(u), str(v)))) for u, v in G.edges())
        nodes = sorted(map(str, G.nodes()))
        payload = f"{nodes}:{edges}"
        return hashlib.md5(payload.encode()).hexdigest()

    def get(self, G, method):
        ghash = self._get_graph_hash(G)
        return self.cache.get(f"{ghash}_{method}")

    def put(self, G, method, data):
        ghash = self._get_graph_hash(G)
        self.cache[f"{ghash}_{method}"] = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=2))