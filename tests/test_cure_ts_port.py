"""Port CURE TypeScript (lib/cure.ts, mode tanpa server di Next.js) harus identik dengan cure.py."""

import json
import random
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cure import CUREClusterer  # noqa: E402
from evaluation import compute_silhouette_score  # noqa: E402


def _toy(n: int, seed: int):
    rnd = random.Random(seed)
    centers = [(-3.2, -2.5, -1.0, 0.4), (-2.3, -1.2, 0.0, 0.5), (0.1, 0.2, 0.3, 0.6), (1.2, 1.4, 1.8, 0.3)]
    X = []
    for i in range(n):
        cx = centers[i % 4]
        X.append([round(v + rnd.gauss(0, 0.6), 2) for v in cx[:3]] + [round(rnd.randint(0, 59) / 60, 4)])
    X += [list(X[0]), list(X[1])]  # titik duplikat -> jarak 0, menguji tie-break
    return X


@unittest.skipIf(shutil.which("node") is None, "Node.js tidak tersedia")
class TestCureTypeScriptPort(unittest.TestCase):
    def test_ts_port_matches_python(self):
        cases = []
        for n, k, c, alpha, seed in [(60, 4, 5, 0.3, 1), (90, 7, 3, 0.0, 2), (40, 3, 1, 1.0, 3), (25, 1, 4, 0.5, 4)]:
            X = _toy(n, seed)
            cases.append({"X": X, "k": k, "c": c, "alpha": alpha, "Xpred": _toy(30, seed + 100)})

        proc = subprocess.run(["node", "--experimental-strip-types", "--no-warnings", str(ROOT / "tests" / "cure_port_check.mjs")],
                              input=json.dumps(cases), capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        ts_results = json.loads(proc.stdout)

        for case, ts in zip(cases, ts_results):
            py = CUREClusterer(case["k"], case["c"], case["alpha"]).fit(case["X"])
            self.assertEqual(list(map(int, py.labels_)), ts["labels"])
            self.assertEqual(py.representatives_, ts["representatives"])  # identik bit-per-bit
            self.assertEqual(list(map(int, py.predict(case["Xpred"]))), ts["predict"])
            self.assertEqual(compute_silhouette_score(py.X_scaled_, py.labels_), ts["silhouette"])


if __name__ == "__main__":
    unittest.main()
