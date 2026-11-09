// Menjalankan port TS (lib/cure.ts) pada JSON dari stdin; dipakai oleh tests/test_cure_ts_port.py.
// Butuh Node >= 22.6 (type stripping bawaan untuk berkas .ts).
import { cureFit, curePredict, silhouetteScore } from '../lib/cure.ts';

let input = '';
process.stdin.on('data', d => { input += d; });
process.stdin.on('end', () => {
  const cases = JSON.parse(input);
  const out = cases.map(({ X, k, c, alpha, Xpred }) => {
    const m = cureFit(X, k, c, alpha);
    return {
      labels: m.labels,
      representatives: m.representatives,
      predict: curePredict(m, Xpred),
      silhouette: silhouetteScore(m.scaled, m.labels),
    };
  });
  process.stdout.write(JSON.stringify(out));
});
