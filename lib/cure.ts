/**
 * Port TypeScript dari cure.py (CUREClusterer, Guha et al. 1998) dan
 * evaluation.compute_silhouette_score. Semantik dibuat sama persis dengan versi Python:
 * standardisasi z (std populasi, std ~0 -> 1), merge hierarkis berbasis jarak minimum antar
 * representatif, c titik tersebar (farthest-first) yang disusutkan alpha ke centroid,
 * tie-break mengikuti urutan penyisipan dict Python, lalu predict = representatif terdekat.
 * Tanpa dependensi agar bisa dijalankan langsung oleh Node (lihat tests/cure_port_check.mjs).
 */

export interface CureModel {
  means: number[];
  stds: number[];
  scaled: number[][];       // matriks sampel terstandar (X_scaled_)
  labels: number[];         // label hasil fit untuk sampel
  representatives: number[][][];
}

function dist(a: number[], b: number[]): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) {
    const t = a[j] - b[j];
    s += t * t;
  }
  return Math.sqrt(s);
}

function selectAndShrink(points: number[][], centroid: number[], c: number, alpha: number): number[][] {
  const n = points.length;
  if (n === 0) return [centroid.slice()];
  let selected: number[][];
  if (n <= c) {
    selected = points;
  } else {
    let first = 0;
    let maxD = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = dist(points[i], centroid);
      if (d > maxD) { maxD = d; first = i; }
    }
    selected = [points[first]];
    const chosen = new Uint8Array(n);
    chosen[first] = 1;
    const minD = new Float64Array(n);
    for (let i = 0; i < n; i++) minD[i] = dist(points[i], points[first]);
    for (let t = 1; t < c; t++) {
      let best = -1;
      let bestD = -1.0;
      for (let i = 0; i < n; i++) {
        if (!chosen[i] && minD[i] > bestD) { bestD = minD[i]; best = i; }
      }
      if (best === -1) break;
      selected.push(points[best]);
      chosen[best] = 1;
      for (let i = 0; i < n; i++) {
        if (!chosen[i]) {
          const d = dist(points[i], points[best]);
          if (d < minD[i]) minD[i] = d;
        }
      }
    }
  }
  return selected.map(p => p.map((v, j) => v + alpha * (centroid[j] - v)));
}

function clusterDistance(a: number[][], b: number[][]): number {
  let m = Infinity;
  for (const p of a) for (const q of b) {
    const d = dist(p, q);
    if (d < m) m = d;
  }
  return m;
}

/** Setara CUREClusterer(k, c, alpha).fit(X). onProgress(merge ke-, total merge). */
export function cureFit(
  X: number[][], k: number, c: number, alpha: number,
  onProgress?: (done: number, total: number) => void
): CureModel {
  if (!(k >= 1) || !(c >= 1) || !(alpha >= 0 && alpha <= 1)) {
    throw new Error(`Parameter CURE tidak valid (k=${k}, c=${c}, alpha=${alpha}).`);
  }
  const N = X.length;
  if (N === 0) throw new Error('Matriks fitur kosong.');
  const D = X[0].length;
  for (const r of X) {
    if (r.length !== D || r.some(v => !Number.isFinite(v))) throw new Error('Fitur CURE tidak lengkap/NaN.');
  }

  const means = new Array(D).fill(0);
  for (const r of X) for (let j = 0; j < D; j++) means[j] += r[j];
  for (let j = 0; j < D; j++) means[j] /= N;
  const stds = new Array(D).fill(0);
  for (const r of X) for (let j = 0; j < D; j++) { const t = r[j] - means[j]; stds[j] += t * t; }
  for (let j = 0; j < D; j++) { const s = Math.sqrt(stds[j] / N); stds[j] = s > 1e-7 ? s : 1.0; }
  const scaled = X.map(r => r.map((v, j) => (v - means[j]) / stds[j]));

  const targetK = Math.min(k, N);
  if (N <= targetK) {
    return { means, stds, scaled, labels: scaled.map((_, i) => i), representatives: scaled.map(p => [p.slice()]) };
  }

  // Slot i = klaster aktif. ids[] = id klaster versi Python (urutan active_ids = id menaik).
  const ids = new Int32Array(N);
  const members: number[][] = [];
  const cent: number[][] = [];
  const reps: number[][][] = [];
  for (let i = 0; i < N; i++) {
    ids[i] = i;
    members.push([i]);
    cent.push(scaled[i].slice());
    reps.push([scaled[i].slice()]);
  }
  // ponytail: matriks jarak N x N (Float64 + Int32 urutan sisip) — sampel 3000 ~ 108 MB; perbesar via jarak on-demand bila perlu
  const dm = new Float64Array(N * N);
  const sq = new Int32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const d = dist(scaled[i], scaled[j]);
      dm[i * N + j] = d; dm[j * N + i] = d;
      sq[i * N + j] = i * N + j; sq[j * N + i] = i * N + j;
    }
  }
  let active: number[] = Array.from({ length: N }, (_, i) => i);
  const nn = new Int32Array(N);
  const nnD = new Float64Array(N);
  const nnS = new Float64Array(N);
  const recompute = (o: number) => {
    let bd = Infinity, bs = Infinity, b = -1;
    for (const p of active) {
      if (p === o) continue;
      const d = dm[o * N + p], s = sq[o * N + p];
      if (d < bd || (d === bd && s < bs)) { bd = d; bs = s; b = p; }
    }
    nn[o] = b; nnD[o] = bd; nnS[o] = bs;
  };
  for (const o of active) recompute(o);

  let nextId = N;
  const totalMerges = N - targetK;
  for (let step = 0; active.length > targetK; step++) {
    // Pasangan global minimum (jarak, lalu urutan sisip) = perilaku min() atas dict Python
    let bi = -1, bd = Infinity, bs = Infinity;
    for (const s of active) {
      if (nnD[s] < bd || (nnD[s] === bd && nnS[s] < bs)) { bd = nnD[s]; bs = nnS[s]; bi = s; }
    }
    const bj = nn[bi];
    const [su, sv] = ids[bi] < ids[bj] ? [bi, bj] : [bj, bi];
    const merged = members[su].concat(members[sv]);
    const nu = members[su].length, nv = members[sv].length;
    const newCent = cent[su].map((v, j) => (nu * v + nv * cent[sv][j]) / (nu + nv));
    const newReps = selectAndShrink(merged.map(i => scaled[i]), newCent, c, alpha);

    active = active.filter(s => s !== su && s !== sv);
    members[su] = merged; cent[su] = newCent; reps[su] = newReps; ids[su] = nextId++;
    members[sv] = []; reps[sv] = [];
    const base = N * N + step * N;
    active.forEach((o, pos) => {
      const d = clusterDistance(newReps, reps[o]);
      dm[su * N + o] = d; dm[o * N + su] = d;
      sq[su * N + o] = base + pos; sq[o * N + su] = base + pos;
    });
    active.push(su);

    recompute(su);
    for (const o of active) {
      if (o === su) continue;
      if (nn[o] === su || nn[o] === sv) {
        recompute(o);
      } else {
        const d = dm[o * N + su], s = sq[o * N + su];
        if (d < nnD[o] || (d === nnD[o] && s < nnS[o])) { nn[o] = su; nnD[o] = d; nnS[o] = s; }
      }
    }
    if (onProgress && (step % 25 === 0)) onProgress(step + 1, totalMerges);
  }

  const labels = new Array<number>(N).fill(0);
  active.forEach((slot, finalIdx) => { for (const p of members[slot]) labels[p] = finalIdx; });
  return { means, stds, scaled, labels, representatives: active.map(s => reps[s]) };
}

/** Setara CUREClusterer.predict: titik ke klaster dengan representatif terdekat (ruang terstandar). */
export function curePredict(model: CureModel, X: number[][]): number[] {
  return X.map(r => {
    const pt = r.map((v, j) => (v - model.means[j]) / model.stds[j]);
    let best = 0, min = Infinity;
    model.representatives.forEach((repsOfC, ci) => {
      for (const rep of repsOfC) {
        const d = dist(pt, rep);
        if (d < min) { min = d; best = ci; }
      }
    });
    return best;
  });
}

/** Sampel acak tanpa pengembalian, terurut, deterministik (mulberry32). Bukan RNG numpy. */
export function seededSample(n: number, size: number, seed = 42): number[] {
  let a = seed >>> 0;
  const rand = () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = 0; i < size; i++) {
    const j = i + Math.floor(rand() * (n - i));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, size).sort((x, y) => x - y);
}

/** Setara evaluation.compute_silhouette_score (subsampel 400 titik berjarak tetap). */
export function silhouetteScore(rows: number[][], labs: number[]): number | null {
  const N = rows.length;
  const unique = Array.from(new Set(labs.filter(l => l >= 0))).sort((x, y) => x - y);
  const k = unique.length;
  if (k <= 1 || k >= N) return null;
  let sr = rows, sl = labs;
  if (N > 400) {
    const step = N / 400;
    const idx = Array.from({ length: 400 }, (_, i) => Math.floor(i * step));
    sr = idx.map(i => rows[i]);
    sl = idx.map(i => labs[i]);
  }
  const vals: number[] = [];
  for (let i = 0; i < sr.length; i++) {
    const ci = sl[i];
    if (ci < 0) continue;
    let sameSum = 0, sameN = 0;
    for (let j = 0; j < sr.length; j++) if (j !== i && sl[j] === ci) { sameSum += dist(sr[i], sr[j]); sameN++; }
    if (sameN === 0) { vals.push(0); continue; }
    const a = sameSum / sameN;
    let b = Infinity;
    for (const oc of unique) {
      if (oc === ci) continue;
      let s = 0, n = 0;
      for (let j = 0; j < sr.length; j++) if (sl[j] === oc) { s += dist(sr[i], sr[j]); n++; }
      if (n > 0 && s / n < b) b = s / n;
    }
    const den = Math.max(a, b);
    vals.push(den > 0 ? (b - a) / den : 0);
  }
  return vals.length ? Math.round((vals.reduce((x, y) => x + y, 0) / vals.length) * 1e4) / 1e4 : null;
}
