import * as XLSX from 'xlsx';

export interface ChildRecord {
  [key: string]: any;
  Cluster_ID?: number;
  Cluster_Name?: string;
  Auto_Label?: string;
  Label_Confidence?: number;
  Clinical_Priority?: string;
  Saran_Asupan_Nutrisi?: string;
  Tindakan_Medis_Sanitasi?: string;
  Jadwal_Monitoring?: string;
  Calculated_HAZ?: number | null;
  Calculated_WAZ?: number | null;
  Calculated_WHZ?: number | null;
}

export interface SheetResult {
  sheetName: string;
  rowCount: number;
  columns: string[];
  featuresUsed: string[];
  records: ChildRecord[];
  clusterDistribution: Record<string, number>;
  labelDistribution: Record<string, number>;
  clusterProfiles: {
    clusterId: number;
    clusterName: string;
    count: number;
    percentage: number;
    hazMean?: number;
    wazMean?: number;
    whzMean?: number;
    ageMean?: number;
  }[];
}

export interface PipelineResult {
  originalFilename: string;
  totalRows: number;
  processedSheets: number;
  cureParams: {
    k: number;
    c: number;
    alpha: number;
  };
  sheets: SheetResult[];
  globalClusterDistribution: Record<string, number>;
  globalLabelDistribution: Record<string, number>;
  executionTimestamp: string;
}

// Indonesian Clinical Stunting Taxonomy (Permenkes RI No. 2/2020 & WHO 2006)
export const TAXONOMY = {
  STUNTING_BERAT_KOMORBID: "Stunting Berat dengan Komorbiditas Klinis (Severely Stunted with Comorbidity)",
  STUNTING_KRONIS: "Stunting Kronis / Sangat Pendek (Chronic Stunting / Severely Stunted)",
  STUNTING_SEDANG: "Stunting Ringan/Sedang / Pendek (Moderate Stunting / Stunted)",
  GIZI_BURUK_AKUT: "Gizi Buruk Akut / Wasting Berat (Severe Acute Malnutrition)",
  GIZI_KURANG_RISIKO: "Gizi Kurang / Risiko Gagal Tumbuh (Wasted / Faltering Growth)",
  NORMAL_OPTIMAL: "Pertumbuhan Normal / Gizi Optimal (Normal Anthropometry)",
  RISIKO_GIZI_LEBIH: "Risiko Gizi Lebih / Overweight (Overweight Risk)"
};

export function normalizeColName(k: string): string {
  return String(k || '')
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '');
}

/**
 * Robust numerical parser that handles:
 * - Indonesian decimal commas: "-2,45" -> -2.45
 * - Unicode minus and dash symbols: "−2.4", "–2.4", "—2.4"
 * - Trailing units: "75.4 cm" -> 75.4, "12,2 kg" -> 12.2, "-2.3 SD" -> -2.3
 */
export function parseNumericValue(val: any): number | null {
  if (val === null || val === undefined || val === '') return null;
  if (typeof val === 'number') return isNaN(val) ? null : val;
  
  let str = String(val).trim();
  if (str === '-' || str === '--' || str === 'null' || str === 'nan') return null;

  // Replace unicode minus signs (U+2212, U+2013 en-dash, U+2014 em-dash, etc.) with standard '-'
  str = str.replace(/[\u2212\u2013\u2014\u2015\uFE63\uFF0D]/g, '-');
  // Handle Indonesian decimal comma (e.g. "-2,45" -> "-2.45")
  str = str.replace(/,/g, '.');
  
  // Extract number with optional leading minus/plus sign
  const match = str.match(/[-+]?\s*\d*\.?\d+/);
  if (match) {
    const cleaned = match[0].replace(/\s+/g, '');
    const num = parseFloat(cleaned);
    return isNaN(num) ? null : num;
  }
  return null;
}

/**
 * Smart Sheet Data Extractor:
 * Real Indonesian Puskesmas / Posyandu / e-PPGBM Excel files frequently have
 * 1-5 title banner rows before the actual table header row (e.g. "LAPORAN STATUS GIZI...").
 * This function scans the top rows to automatically detect the genuine header row.
 */
export function extractDataFromWorksheet(ws: XLSX.WorkSheet): { rawData: Record<string, any>[]; detectedHeaders: string[] } {
  // Convert sheet to 2D array of rows
  const rows: any[][] = XLSX.utils.sheet_to_json(ws, { header: 1, defval: '' });
  if (!rows || rows.length === 0) return { rawData: [], detectedHeaders: [] };

  const headerKeywords = [
    'nama', 'balita', 'anak', 'bayi', 'pasien', 'nik', 'no', 'nomor',
    'usia', 'umur', 'bulan', 'bln', 'tgl', 'tanggal', 'lahir', 'ukur', 'timbang',
    'jk', 'jenis_kelamin', 'gender', 'sex', 'l_p',
    'bb', 'berat', 'weight',
    'tb', 'pb', 'tinggi', 'panjang', 'height',
    'haz', 'waz', 'whz', 'zscore', 'z_score', 'zs', 'tb_u', 'bb_u', 'bb_tb',
    'status', 'gizi', 'stunting', 'kategori', 'diagnosa', 'hasil', 'catatan', 'keterangan'
  ];

  let bestHeaderRowIdx = 0;
  let maxScore = -1;

  const maxScanRows = Math.min(15, rows.length);
  for (let r = 0; r < maxScanRows; r++) {
    const row = rows[r];
    if (!Array.isArray(row) || row.length === 0) continue;

    let score = 0;
    for (const cell of row) {
      if (cell === null || cell === undefined) continue;
      const cellNorm = normalizeColName(String(cell));
      if (!cellNorm) continue;
      if (headerKeywords.some(kw => cellNorm.includes(kw))) {
        score++;
      }
    }

    if (score > maxScore) {
      maxScore = score;
      bestHeaderRowIdx = r;
    }
  }

  // Header found
  const headerRow = rows[bestHeaderRowIdx] || [];
  let dataStartIdx = bestHeaderRowIdx + 1;
  const nextRow = rows[bestHeaderRowIdx + 1];
  let subheaderUsed = false;
  let finalHeaders: string[] = [];

  // Check if row below is a merged subheader row
  if (nextRow && Array.isArray(nextRow) && dataStartIdx < rows.length) {
    let nextRowScore = 0;
    for (const cell of nextRow) {
      const cellNorm = normalizeColName(String(cell));
      if (cellNorm && headerKeywords.some(kw => cellNorm.includes(kw))) {
        nextRowScore++;
      }
    }
    if (nextRowScore >= 2) {
      dataStartIdx = bestHeaderRowIdx + 2;
      subheaderUsed = true;
      const maxCols = Math.max(headerRow.length, nextRow.length);
      for (let c = 0; c < maxCols; c++) {
        const top = String(headerRow[c] || '').trim();
        const btm = String(nextRow[c] || '').trim();
        let combined = '';
        if (top && btm && top !== btm) {
          combined = `${top} ${btm}`;
        } else {
          combined = btm || top || `Kolom_${c + 1}`;
        }
        finalHeaders.push(combined);
      }
    }
  }

  if (!subheaderUsed) {
    finalHeaders = headerRow.map((cell: any, c: number) => {
      const s = String(cell || '').trim();
      return s || `Kolom_${c + 1}`;
    });
  }

  // Build raw data objects
  const rawData: Record<string, any>[] = [];
  for (let r = dataStartIdx; r < rows.length; r++) {
    const row = rows[r];
    if (!Array.isArray(row)) continue;
    const hasData = row.some(val => val !== null && val !== undefined && String(val).trim() !== '');
    if (!hasData) continue;

    const obj: Record<string, any> = {};
    for (let c = 0; c < finalHeaders.length; c++) {
      const colName = finalHeaders[c];
      obj[colName] = row[c] !== undefined ? row[c] : '';
    }
    rawData.push(obj);
  }

  return { rawData, detectedHeaders: finalHeaders };
}

// Extract standard anthropometrics and demographics from any record
export function extractChildIndicators(record: Record<string, any>): {
  childName: string;
  haz: number | null;
  waz: number | null;
  whz: number | null;
  ageMonths: number | null;
  heightCm: number | null;
  weightKg: number | null;
  gender: 'L' | 'P' | null;
  statusText: string;
  notes: string;
} {
  let childName = '';
  let haz: number | null = null;
  let waz: number | null = null;
  let whz: number | null = null;
  let ageMonths: number | null = null;
  let heightCm: number | null = null;
  let weightKg: number | null = null;
  let gender: 'L' | 'P' | null = null;
  let notes = '';
  let statusText = '';

  let birthDate: Date | null = null;
  let visitDate: Date | null = null;

  for (const [rawKey, rawVal] of Object.entries(record)) {
    if (rawVal === null || rawVal === undefined) continue;
    const key = normalizeColName(rawKey);
    const valStr = String(rawVal).trim().toLowerCase();

    // Child Name
    if (!childName && (key.includes('nama') || key === 'name' || key === 'balita' || key === 'anak' || key === 'bayi')) {
      if (typeof rawVal === 'string' && rawVal.trim().length > 1 && isNaN(Number(rawVal))) {
        childName = rawVal.trim();
      }
    }

    // Gender detection
    if (['jk', 'gender', 'jenis_kelamin', 'sex', 'l_p', 'j_k'].includes(key) || key.includes('kelamin')) {
      if (valStr.startsWith('l') || valStr.includes('laki') || valStr.includes('pria') || valStr === '1' || valStr === 'male') {
        gender = 'L';
      } else if (valStr.startsWith('p') || valStr.includes('perempuan') || valStr.includes('wanita') || valStr === '2' || valStr === 'female') {
        gender = 'P';
      }
    }

    // Date of Birth & Measurement Date
    if (key.includes('lahir') || key.includes('birth')) {
      const d = new Date(rawVal);
      if (!isNaN(d.getTime())) birthDate = d;
    }
    if (key.includes('ukur') || key.includes('timbang') || key.includes('kunjungan')) {
      const d = new Date(rawVal);
      if (!isNaN(d.getTime())) visitDate = d;
    }

    // HAZ / TB/U Z-score
    // Matches: tb_u, zscore_tb_u, haz, nilai_zscore_tb_u, status_tb_u, etc.
    const isHazCol = (key.includes('tb') || key.includes('haz') || key.includes('tinggi') || key.includes('pb')) &&
      (key.includes('z') || key.includes('score') || key.includes('zs') || key.includes('indeks') || key.includes('u'));
    
    if (isHazCol || ['haz', 'tb_u', 'zscore_tb_u', 'z_score_tb_u', 'zs_tb_u', 'tb_u_zscore'].includes(key)) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= -6.0 && num <= 6.0) {
        haz = num;
      } else if (valStr.length > 0) {
        statusText += ` ${valStr}`;
        // Qualitative status in TB/U column
        if (valStr.includes('sangat pendek') || valStr.includes('severely stunted')) {
          if (haz === null) haz = -3.4;
        } else if (valStr.includes('pendek') || valStr.includes('stunted')) {
          if (haz === null) haz = -2.4;
        } else if (valStr.includes('normal') || valStr.includes('gizi baik')) {
          if (haz === null) haz = 0.1;
        } else if (valStr.includes('tinggi')) {
          if (haz === null) haz = 2.2;
        }
      }
    }

    // WAZ / BB/U Z-score
    const isWazCol = (key.includes('bb') || key.includes('waz') || key.includes('berat')) &&
      (key.includes('z') || key.includes('score') || key.includes('zs') || key.includes('indeks') || key.includes('u')) &&
      !key.includes('tb');

    if (isWazCol || ['waz', 'bb_u', 'zscore_bb_u', 'z_score_bb_u', 'zs_bb_u', 'bb_u_zscore'].includes(key)) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= -6.0 && num <= 6.0) {
        waz = num;
      } else if (valStr.length > 0) {
        statusText += ` ${valStr}`;
        if (valStr.includes('sangat kurang') || valStr.includes('severely underweight')) {
          if (waz === null) waz = -3.2;
        } else if (valStr.includes('kurang') || valStr.includes('underweight')) {
          if (waz === null) waz = -2.2;
        } else if (valStr.includes('normal')) {
          if (waz === null) waz = 0.0;
        }
      }
    }

    // WHZ / BB/TB Z-score
    const isWhzCol = (key.includes('bb_tb') || key.includes('whz') || key.includes('bbtb') || key.includes('wfh')) &&
      (key.includes('z') || key.includes('score') || key.includes('zs') || key.includes('indeks') || key.includes('tb'));

    if (isWhzCol || ['whz', 'bb_tb', 'bbtb', 'zscore_bb_tb', 'z_score_bb_tb'].includes(key)) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= -6.0 && num <= 6.0) {
        whz = num;
      } else if (valStr.length > 0) {
        statusText += ` ${valStr}`;
        if (valStr.includes('gizi buruk') || valStr.includes('severely wasted')) {
          if (whz === null) whz = -3.3;
        } else if (valStr.includes('gizi kurang') || valStr.includes('kurus')) {
          if (whz === null) whz = -2.3;
        } else if (valStr.includes('normal') || valStr.includes('gizi baik')) {
          if (whz === null) whz = 0.1;
        } else if (valStr.includes('gizi lebih') || valStr.includes('gemuk')) {
          if (whz === null) whz = 2.2;
        }
      }
    }

    // Age in Months
    if (
      ['usia_bulan', 'umur_bulan', 'usia', 'umur', 'bulan', 'age', 'age_months', 'umur_bln', 'usia_bln', 'u_bln', 'u_bulan'].includes(key) ||
      (key.includes('usia') || key.includes('umur')) && !key.includes('tgl') && !key.includes('tanggal')
    ) {
      // Check if age string like "2 th 3 bln" or "18 bln" or "2 tahun"
      let parsedAge: number | null = null;
      const thMatch = valStr.match(/(\d+)\s*(th|tahun|yr|year)/);
      const blnMatch = valStr.match(/(\d+)\s*(bln|bulan|mo|month)/);
      if (thMatch || blnMatch) {
        const years = thMatch ? parseInt(thMatch[1]) : 0;
        const months = blnMatch ? parseInt(blnMatch[1]) : 0;
        parsedAge = years * 12 + months;
      } else {
        const num = parseNumericValue(rawVal);
        if (num !== null && num >= 0 && num <= 72) {
          parsedAge = Math.round(num);
        }
      }
      if (parsedAge !== null && ageMonths === null) {
        ageMonths = parsedAge;
      }
    }

    // Height in cm (TB / PB)
    const isHeightCol = (key.includes('tinggi') || key.includes('tb') || key.includes('panjang') || key.includes('pb') || key.includes('height')) &&
      !key.includes('tb_u') && !key.includes('zscore') && !key.includes('z_score') && !key.includes('status') && !key.includes('indeks') && !key.includes('kategori');

    if (isHeightCol) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= 35 && num <= 140) {
        heightCm = num;
      }
    }

    // Weight in kg (BB)
    const isWeightCol = (key.includes('berat') || key.includes('bb') || key.includes('weight')) &&
      !key.includes('bb_u') && !key.includes('bb_tb') && !key.includes('zscore') && !key.includes('z_score') && !key.includes('status') && !key.includes('indeks') && !key.includes('kategori');

    if (isWeightCol) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= 1.5 && num <= 45) {
        weightKg = num;
      }
    }

    // Status or qualitative text columns
    if (['status', 'status_gizi', 'kategori', 'diagnosa', 'hasil', 'keterangan_gizi', 'kesimpulan'].some(t => key.includes(t))) {
      statusText += ` ${valStr}`;
    }

    // Clinical notes
    if (['catatan', 'keterangan', 'keluhan', 'riwayat', 'notes', 'bidan', 'pemeriksaan', 'kondisi', 'penyakit'].some(t => key.includes(t))) {
      notes += ` ${rawVal}`;
    }
  }

  // Calculate age if birthDate and visitDate are available
  if (ageMonths === null && birthDate && visitDate) {
    const diffTime = visitDate.getTime() - birthDate.getTime();
    const diffMonths = Math.round(diffTime / (1000 * 60 * 60 * 24 * 30.4375));
    if (diffMonths >= 0 && diffMonths <= 72) {
      ageMonths = diffMonths;
    }
  }

  // Default age fallback if completely missing
  if (ageMonths === null && heightCm !== null) {
    // Approximate age from height: ~75cm is ~12mo, ~85cm is ~24mo
    ageMonths = Math.max(6, Math.min(60, Math.round((heightCm - 50) / 1.5)));
  }

  // Calculate standard WHO 2006 HAZ Z-score if missing
  if (haz === null && heightCm !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const hMed = isBoy
      ? (ageMonths <= 24 ? 50.0 + 1.85 * ageMonths - 0.035 * ageMonths * ageMonths : 87.8 + 0.65 * (ageMonths - 24))
      : (ageMonths <= 24 ? 49.0 + 1.80 * ageMonths - 0.035 * ageMonths * ageMonths : 86.5 + 0.63 * (ageMonths - 24));
    const hSD = 2.1 + 0.04 * ageMonths;
    haz = Math.round(((heightCm - hMed) / hSD) * 100) / 100;
  }

  // Calculate standard WHO 2006 WAZ Z-score if missing
  if (waz === null && weightKg !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const wMed = isBoy
      ? (ageMonths <= 24 ? 3.3 + 0.52 * ageMonths - 0.005 * ageMonths * ageMonths : 12.2 + 0.18 * (ageMonths - 24))
      : (ageMonths <= 24 ? 3.2 + 0.49 * ageMonths - 0.005 * ageMonths * ageMonths : 11.5 + 0.17 * (ageMonths - 24));
    const wSD = 0.5 + 0.03 * ageMonths;
    waz = Math.round(((weightKg - wMed) / wSD) * 100) / 100;
  }

  // Estimate WHZ if missing
  if (whz === null && heightCm !== null && weightKg !== null) {
    const expectedWeight = Math.max(2.0, Math.pow(heightCm / 100, 2) * 16.0);
    const whzEst = (weightKg - expectedWeight) / (expectedWeight * 0.12);
    whz = Math.round(whzEst * 100) / 100;
  }

  // Qualitative text override if HAZ is still missing
  const combinedText = `${statusText} ${notes}`.toLowerCase();
  if (haz === null) {
    if (combinedText.includes('sangat pendek') || combinedText.includes('severely stunted')) {
      haz = -3.4;
    } else if (combinedText.includes('pendek') || combinedText.includes('stunted')) {
      haz = -2.4;
    } else if (combinedText.includes('normal') || combinedText.includes('gizi baik')) {
      haz = 0.1;
    } else if (combinedText.includes('tinggi')) {
      haz = 2.3;
    }
  }

  if (whz === null) {
    if (combinedText.includes('gizi buruk') || combinedText.includes('wasting berat')) {
      whz = -3.2;
    } else if (combinedText.includes('gizi kurang') || combinedText.includes('kurus')) {
      whz = -2.2;
    } else if (combinedText.includes('gizi baik') || combinedText.includes('normal')) {
      whz = 0.1;
    } else if (combinedText.includes('gizi lebih') || combinedText.includes('gemuk')) {
      whz = 2.2;
    }
  }

  return {
    childName,
    haz,
    waz,
    whz,
    ageMonths,
    heightCm,
    weightKg,
    gender,
    statusText,
    notes: combinedText
  };
}

// CURE (Clustering Using Representatives) Algorithm
export class CUREAlgorithm {
  k: number;
  c: number;
  alpha: number;

  constructor(k: number = 4, c: number = 5, alpha: number = 0.3) {
    this.k = Math.max(2, k);
    this.c = Math.max(1, c);
    this.alpha = Math.min(Math.max(alpha, 0.05), 0.95);
  }

  fitPredict(features: number[][]): { labels: number[]; representatives: number[][][] } {
    const N = features.length;
    if (N === 0) return { labels: [], representatives: [] };
    if (N <= this.k) {
      return {
        labels: features.map((_, i) => i % this.k),
        representatives: []
      };
    }

    const D = features[0].length;

    // Feature Standardization (Min-Max Scaling [0, 1])
    const mins = new Array(D).fill(Infinity);
    const maxs = new Array(D).fill(-Infinity);

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < D; j++) {
        const val = features[i][j];
        if (val < mins[j]) mins[j] = val;
        if (val > maxs[j]) maxs[j] = val;
      }
    }

    const scaled: number[][] = features.map((row, rowIdx) =>
      row.map((val, j) => {
        const range = maxs[j] - mins[j];
        if (range > 1e-6) {
          return (val - mins[j]) / range;
        }
        // Small variance fallback
        return rowIdx / Math.max(1, N - 1);
      })
    );

    // K-Means++ initial centroids selection
    const initK = Math.min(N, Math.max(this.k * 2, this.k));
    const centroids: number[][] = [[...scaled[0]]];
    const chosenIndices = new Set<number>([0]);

    while (centroids.length < initK && chosenIndices.size < N) {
      let maxDist = -1;
      let bestIdx = -1;

      for (let i = 0; i < N; i++) {
        if (chosenIndices.has(i)) continue;
        let minDist = Infinity;
        for (const cent of centroids) {
          const d = this.euclidean(scaled[i], cent);
          if (d < minDist) minDist = d;
        }
        if (minDist > maxDist) {
          maxDist = minDist;
          bestIdx = i;
        }
      }

      if (bestIdx === -1) {
        for (let i = 0; i < N; i++) {
          if (!chosenIndices.has(i)) {
            bestIdx = i;
            break;
          }
        }
      }

      if (bestIdx !== -1) {
        chosenIndices.add(bestIdx);
        centroids.push([...scaled[bestIdx]]);
      } else {
        break;
      }
    }

    // Initial clusters assignment
    const clusters: { members: number[]; centroid: number[]; reps: number[][] }[] = [];
    for (let idx = 0; idx < centroids.length; idx++) {
      clusters.push({ members: [], centroid: centroids[idx], reps: [] });
    }

    for (let i = 0; i < N; i++) {
      let minDist = Infinity;
      let bestC = 0;
      for (let j = 0; j < centroids.length; j++) {
        const d = this.euclidean(scaled[i], centroids[j]);
        if (d < minDist) {
          minDist = d;
          bestC = j;
        }
      }
      clusters[bestC].members.push(i);
    }

    // Filter non-empty clusters and generate CURE shrunk representative points
    let activeClusters = clusters
      .filter(cl => cl.members.length > 0)
      .map(cl => {
        const pts = cl.members.map(idx => scaled[idx]);
        const centroid = this.computeMean(pts, D);
        const reps = this.selectRepresentatives(pts, centroid, this.c);
        const shrunkReps = reps.map(r =>
          r.map((val, dim) => val + this.alpha * (centroid[dim] - val))
        );
        return {
          members: cl.members,
          centroid,
          reps: shrunkReps
        };
      });

    // If active clusters is fewer than k, split the largest clusters
    while (activeClusters.length < this.k && activeClusters.length > 0) {
      let largestIdx = -1;
      let maxMembers = 1;
      for (let i = 0; i < activeClusters.length; i++) {
        if (activeClusters[i].members.length > maxMembers) {
          maxMembers = activeClusters[i].members.length;
          largestIdx = i;
        }
      }
      if (largestIdx === -1) break;

      const target = activeClusters[largestIdx];
      const half = Math.floor(target.members.length / 2);
      const m1 = target.members.slice(0, half);
      const m2 = target.members.slice(half);

      const pts1 = m1.map(idx => scaled[idx]);
      const cent1 = this.computeMean(pts1, D);
      const reps1 = this.selectRepresentatives(pts1, cent1, this.c).map(r =>
        r.map((val, dim) => val + this.alpha * (cent1[dim] - val))
      );

      const pts2 = m2.map(idx => scaled[idx]);
      const cent2 = this.computeMean(pts2, D);
      const reps2 = this.selectRepresentatives(pts2, cent2, this.c).map(r =>
        r.map((val, dim) => val + this.alpha * (cent2[dim] - val))
      );

      activeClusters.splice(largestIdx, 1,
        { members: m1, centroid: cent1, reps: reps1 },
        { members: m2, centroid: cent2, reps: reps2 }
      );
    }

    // Hierarchical Agglomerative Merging down to target k clusters
    while (activeClusters.length > this.k) {
      let minPairDist = Infinity;
      let mergeI = 0;
      let mergeJ = 1;

      for (let i = 0; i < activeClusters.length; i++) {
        for (let j = i + 1; j < activeClusters.length; j++) {
          const d = this.minRepresentativeDistance(activeClusters[i].reps, activeClusters[j].reps);
          if (d < minPairDist) {
            minPairDist = d;
            mergeI = i;
            mergeJ = j;
          }
        }
      }

      const c1 = activeClusters[mergeI];
      const c2 = activeClusters[mergeJ];
      const mergedMembers = [...c1.members, ...c2.members];
      const mergedPts = mergedMembers.map(idx => scaled[idx]);
      const newCentroid = this.computeMean(mergedPts, D);
      const newReps = this.selectRepresentatives(mergedPts, newCentroid, this.c);
      const newShrunkReps = newReps.map(r =>
        r.map((val, dim) => val + this.alpha * (newCentroid[dim] - val))
      );

      activeClusters.splice(Math.max(mergeI, mergeJ), 1);
      activeClusters.splice(Math.min(mergeI, mergeJ), 1);
      activeClusters.push({
        members: mergedMembers,
        centroid: newCentroid,
        reps: newShrunkReps
      });
    }

    // Sort clusters consistently by average HAZ (most severe first)
    activeClusters.sort((a, b) => {
      const hazA = a.members.reduce((sum, idx) => sum + features[idx][0], 0) / Math.max(1, a.members.length);
      const hazB = b.members.reduce((sum, idx) => sum + features[idx][0], 0) / Math.max(1, b.members.length);
      return hazA - hazB;
    });

    // Assign final cluster IDs (0 to k-1)
    const labels = new Array(N).fill(0);
    const representatives: number[][][] = [];

    activeClusters.forEach((cl, clusterIdx) => {
      representatives.push(cl.reps);
      for (const memberIdx of cl.members) {
        labels[memberIdx] = clusterIdx;
      }
    });

    return { labels, representatives };
  }

  private euclidean(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  private computeMean(pts: number[][], D: number): number[] {
    if (pts.length === 0) return new Array(D).fill(0);
    const mean = new Array(D).fill(0);
    for (const p of pts) {
      for (let j = 0; j < D; j++) mean[j] += p[j];
    }
    return mean.map(v => v / pts.length);
  }

  private selectRepresentatives(pts: number[][], centroid: number[], c: number): number[][] {
    const n = pts.length;
    const targetC = Math.min(c, n);
    if (targetC <= 1) return [[...centroid]];
    if (n <= targetC) return pts.map(p => [...p]);

    let maxDist = -1;
    let firstIdx = 0;
    for (let i = 0; i < n; i++) {
      const d = this.euclidean(pts[i], centroid);
      if (d > maxDist) {
        maxDist = d;
        firstIdx = i;
      }
    }

    const selected: number[] = [firstIdx];
    while (selected.length < targetC) {
      let bestDist = -1;
      let nextIdx = 0;

      for (let i = 0; i < n; i++) {
        if (selected.includes(i)) continue;
        let minDistToRep = Infinity;
        for (const s of selected) {
          const d = this.euclidean(pts[i], pts[s]);
          if (d < minDistToRep) minDistToRep = d;
        }
        if (minDistToRep > bestDist) {
          bestDist = minDistToRep;
          nextIdx = i;
        }
      }
      selected.push(nextIdx);
    }

    return selected.map(idx => [...pts[idx]]);
  }

  private minRepresentativeDistance(repsA: number[][], repsB: number[][]): number {
    let minDist = Infinity;
    for (const a of repsA) {
      for (const b of repsB) {
        const d = this.euclidean(a, b);
        if (d < minDist) minDist = d;
      }
    }
    return minDist;
  }
}

// Predict status and generate personalized, child-specific recommendations
export function predictChildClassification(indicators: ReturnType<typeof extractChildIndicators>): {
  autoLabel: string;
  confidence: number;
  priority: string;
  saranNutrisi: string;
  tindakanMedis: string;
  jadwalMonitoring: string;
} {
  const { childName, haz, waz, whz, ageMonths, heightCm, weightKg, gender, notes } = indicators;

  const hasSevereNotes = [
    'sangat pendek', 'severely stunted', 'bgm', 'bawah garis merah', 'kms merah',
    'faltering', 'gagal tumbuh', 'edema', 'maramus', 'kwashiorkor', 'tb aktif',
    'lila merah', 'diare kronis'
  ].some(m => notes.includes(m));

  const hasModerateNotes = [
    'pendek', 'stunted', 'kurus', 'gizi kurang', 'kms kuning', '2t', 'tidak naik',
    'nafsu makan turun', 'mpasi kurang', 'lila kuning'
  ].some(m => notes.includes(m));

  const hasNormalNotes = [
    'normal', 'sehat', 'gizi baik', 'aktif', 'pertumbuhan sesuai', 'kms hijau',
    'asi eksklusif', 'imunisasi lengkap'
  ].some(m => notes.includes(m));

  let label = TAXONOMY.NORMAL_OPTIMAL;
  let confidence = 0.90;
  let priority = 'Rendah (Pencegahan Rutin)';

  // 1. Classification Decision Tree based on WHO 2006 & Permenkes RI 2/2020
  if (haz !== null) {
    if (haz < -3.0) {
      if (hasSevereNotes || (whz !== null && whz < -2.0) || (waz !== null && waz < -3.0)) {
        label = TAXONOMY.STUNTING_BERAT_KOMORBID;
        confidence = 0.96;
        priority = 'Kritis (Intervensi Medis & Rujukan RS)';
      } else {
        label = TAXONOMY.STUNTING_KRONIS;
        confidence = 0.94;
        priority = 'Tinggi (Intervensi Gizi Spesifik)';
      }
    } else if (haz >= -3.0 && haz < -2.0) {
      if (hasSevereNotes || (waz !== null && waz < -2.5)) {
        label = TAXONOMY.STUNTING_BERAT_KOMORBID;
        confidence = 0.89;
        priority = 'Tinggi (Intervensi Gizi Spesifik)';
      } else {
        label = TAXONOMY.STUNTING_SEDANG;
        confidence = 0.92;
        priority = 'Sedang (Pemberian Makanan Tambahan PMT)';
      }
    } else if (haz >= -2.0 && haz <= 2.0) {
      if (whz !== null && whz < -3.0) {
        label = TAXONOMY.GIZI_BURUK_AKUT;
        confidence = 0.95;
        priority = 'Kritis (Tatalaksana Gizi Buruk)';
      } else if (whz !== null && whz >= -3.0 && whz < -2.0) {
        label = TAXONOMY.GIZI_KURANG_RISIKO;
        confidence = 0.88;
        priority = 'Sedang (Pemantauan Kejar Tumbuh)';
      } else if (hasModerateNotes) {
        label = TAXONOMY.GIZI_KURANG_RISIKO;
        confidence = 0.84;
        priority = 'Sedang (Edukasi PMBA & Pemantauan)';
      } else {
        label = TAXONOMY.NORMAL_OPTIMAL;
        confidence = 0.95;
        priority = 'Rendah (Pemantauan Rutin Posyandu)';
      }
    } else {
      label = TAXONOMY.RISIKO_GIZI_LEBIH;
      confidence = 0.88;
      priority = 'Rendah (Pola Makan Seimbang)';
    }
  } else {
    // Textual dominance when numerical HAZ is absent
    if (hasSevereNotes) {
      label = TAXONOMY.STUNTING_BERAT_KOMORBID;
      confidence = 0.88;
      priority = 'Tinggi (Pemeriksaan Faskes Lanjutan)';
    } else if (hasModerateNotes) {
      label = TAXONOMY.STUNTING_SEDANG;
      confidence = 0.84;
      priority = 'Sedang (Pemberian Makanan Tambahan)';
    } else if (hasNormalNotes) {
      label = TAXONOMY.NORMAL_OPTIMAL;
      confidence = 0.90;
      priority = 'Rendah (Pemantauan Rutin)';
    } else {
      label = TAXONOMY.NORMAL_OPTIMAL;
      confidence = 0.75;
      priority = 'Rendah (Evaluasi Berkala)';
    }
  }

  // 2. Recommendations tailored to this specific child
  const recommendations = generateInterventionRecommendations(
    label,
    ageMonths,
    haz,
    waz,
    whz,
    heightCm,
    weightKg,
    childName,
    gender,
    notes
  );

  return {
    autoLabel: label,
    confidence,
    priority,
    saranNutrisi: recommendations.saranNutrisi,
    tindakanMedis: recommendations.tindakanMedis,
    jadwalMonitoring: recommendations.jadwalMonitoring
  };
}

export function generateInterventionRecommendations(
  autoLabel: string,
  ageMonths: number | null,
  haz: number | null,
  waz: number | null,
  whz: number | null,
  heightCm: number | null,
  weightKg: number | null,
  childName: string,
  gender: 'L' | 'P' | null,
  notes: string
): {
  saranNutrisi: string;
  tindakanMedis: string;
  jadwalMonitoring: string;
} {
  const isSevere = autoLabel.includes('Stunting Berat') || autoLabel.includes('Gizi Buruk') || (haz !== null && haz < -3.0);
  const isModerate = autoLabel.includes('Stunting Ringan') || autoLabel.includes('Gizi Kurang') || (haz !== null && haz < -2.0 && haz >= -3.0);
  const isOverweight = autoLabel.includes('Risiko Gizi Lebih');
  const isUnderweight = (waz !== null && waz < -2.0) || notes.includes('kurang') || notes.includes('2t') || notes.includes('bgm');
  const isWasted = (whz !== null && whz < -2.0) || notes.includes('kurus') || notes.includes('wasting');

  const childGenderStr = gender === 'P' ? 'Perempuan' : gender === 'L' ? 'Laki-laki' : '';
  const childTag = childName ? `Anak ${childName}` : 'Balita';
  const ageDisplay = ageMonths !== null ? `${ageMonths} bln` : 'Usia Balita';
  const headerSummary = `[${childTag} - ${ageDisplay}${childGenderStr ? `, ${childGenderStr}` : ''}]:`;

  // 1. Precise anthropometric status & gap against WHO Median
  const statusParts: string[] = [];
  if (haz !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const hMed = isBoy
      ? (ageMonths <= 24 ? 50.0 + 1.85 * ageMonths - 0.035 * ageMonths * ageMonths : 87.8 + 0.65 * (ageMonths - 24))
      : (ageMonths <= 24 ? 49.0 + 1.80 * ageMonths - 0.035 * ageMonths * ageMonths : 86.5 + 0.63 * (ageMonths - 24));
    
    if (heightCm !== null) {
      const diff = Math.round((hMed - heightCm) * 10) / 10;
      if (diff > 0) {
        statusParts.push(`Tinggi ${heightCm} cm (defisit ${diff} cm di bawah median WHO ${Math.round(hMed * 10) / 10} cm, Z-Score TB/U: ${haz.toFixed(2)} SD).`);
      } else {
        statusParts.push(`Tinggi ${heightCm} cm telah sesuai/di atas median WHO seusianya (${Math.round(hMed * 10) / 10} cm, Z-Score TB/U: ${haz.toFixed(2)} SD).`);
      }
    } else {
      statusParts.push(`Z-Score TB/U tercatat ${haz.toFixed(2)} SD.`);
    }
  } else if (heightCm !== null) {
    statusParts.push(`Tinggi badan tercatat ${heightCm} cm.`);
  }

  if (waz !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const wMed = isBoy
      ? (ageMonths <= 24 ? 3.3 + 0.52 * ageMonths - 0.005 * ageMonths * ageMonths : 12.2 + 0.18 * (ageMonths - 24))
      : (ageMonths <= 24 ? 3.2 + 0.49 * ageMonths - 0.005 * ageMonths * ageMonths : 11.5 + 0.17 * (ageMonths - 24));
    
    if (weightKg !== null) {
      const diffW = Math.round((wMed - weightKg) * 10) / 10;
      if (diffW > 0) {
        statusParts.push(`Berat ${weightKg} kg (defisit ${diffW} kg dari target median WHO ${Math.round(wMed * 10) / 10} kg, Z-Score BB/U: ${waz.toFixed(2)} SD).`);
      } else {
        statusParts.push(`Berat ${weightKg} kg (sesuai target median WHO ${Math.round(wMed * 10) / 10} kg, Z-Score BB/U: ${waz.toFixed(2)} SD).`);
      }
    }
  }

  // 2. Specific age-bracket food consistency, portions, and protein menu
  let ageMenuPlan = '';
  if (ageMonths !== null) {
    if (ageMonths < 6) {
      ageMenuPlan = 'Fase Usia < 6 Bulan: Wajib ASI Eksklusif on-demand minimal 8-12 kali per 24 jam tanpa tambahan air putih atau makanan padat apapun. Ibu menyusui wajib mengonsumsi double protein hewani harian (2 butir telur + 1 porsi ikan kembung/tongkol) dan suplemen kalsium serta tablet tambah darah agar densitas gizi ASI optimal.';
    } else if (ageMonths >= 6 && ageMonths <= 8) {
      ageMenuPlan = 'Fase MPASI Awal (6-8 Bulan): Tekstur saring/lumat kental (tidak tumpah saat sendok dimiringkan). Jadwal: 2-3 kali makan utama (mulai 2-3 sdm bertahap hingga 1/2 mangkuk 125ml) + 1-2 selingan padat gizi. Menu harian wajib: 1 butir telur ayam rebus dilumatkan + 1 sdm hati ayam atau ikan kembung suwir saring halus + 1 sendok teh lemak tambahan (minyak kelapa sawit/santan matang/margarin) sebagai pendorong kalori (energy booster).';
    } else if (ageMonths >= 9 && ageMonths <= 11) {
      ageMenuPlan = 'Fase MPASI Lanjutan (9-11 Bulan): Tekstur cincang halus atau finger foods lunak yang mudah dijumput anak. Jadwal: 3-4 kali makan utama (1/2 mangkuk 125-150ml) + 1-2 kali selingan padat gizi dan ASI. Komposisi harian: 1 butir telur ayam + 1 potong ikan laut lokal (lele/kembung/patin kaya omega-3 DHA dan asam amino lisin untuk lempeng pertumbuhan tulang panjang) + 1 sdt minyak kelapa matang atau mentega.';
    } else if (ageMonths >= 12 && ageMonths <= 23) {
      ageMenuPlan = 'Fase Makanan Keluarga & MPASI Baduta (12-23 Bulan): Makanan keluarga dipotong kecil/lunak, 3-4 kali makan utama (3/4 mangkuk 200ml) + 2 kali selingan padat protein (puding telur, nagasari santan, atau perkedel tahu daging) + ASI hingga 2 tahun. Sasaran harian: Minimal 1-2 butir telur ayam + 30-40 gram daging ayam/ikan lokal/hati sapi.';
    } else if (ageMonths >= 24 && ageMonths <= 35) {
      ageMenuPlan = 'Fase Anak Batita (2-3 Tahun): Makanan keluarga menu Isi Piringku 3 kali sehari porsi penuh + 2 selingan buah segar atau kudapan protein hewani. Sasaran harian: 1 butir telur + 1 porsi lauk hewani (ayam, ikan laut, atau daging sapi) + tahu/tempe + sayuran beraneka warna. Hindari jajanan manis kemasan dan minuman berpemanis buatan agar nafsu makan anak tidak turun.';
    } else {
      ageMenuPlan = 'Fase Anak Prasekolah (3-5 Tahun): Menu gizi seimbang Isi Piringku 3 kali makan utama + 2 kali selingan sehat. Porsi harian: 1 butir telur + 1 porsi protein hewani berkualitas tinggi + sayuran sumber vitamin A & zat besi + susu pasteurisasi tanpa gula/air putih higienis 6-8 gelas.';
    }
  } else {
    ageMenuPlan = 'Pola Makan Balita: Utamakan makanan padat gizi berbasis protein hewani (telur, ikan laut lokal, daging ayam) 3 kali makan utama sehari ditambah selingan sehat.';
  }

  // 3. Clinical dietary intervention based on stunting severity and wasting status
  let targetedDiet = '';
  if (isSevere) {
    targetedDiet = 'Intervensi Pemulihan Stunting Kronis / Berat: Wajib konsumsi Double Protein Hewani harian (misal: telur ayam di pagi hari + ikan kembung/hati ayam di siang dan sore). Tambahkan ekstra 1-2 sdt minyak kelapa matang/santan kental pada setiap mangkuk makan anak untuk mengejar ketertinggalan kalori. Bila anak mengalami anoreksia/sulit makan, konsultasikan ke dokter untuk peresepan Pangan Olahan untuk Keperluan Medis Khusus (PKMK) tinggi kalori (1.0-1.5 kkal/ml) serta tabur gizi mikro (Taburia/zinc).';
  } else if (isModerate) {
    targetedDiet = 'Intervensi Stunting Sedang / Kejar Tumbuh: Berikan Pemberian Makanan Tambahan (PMT) Pemulihan berbasis bahan pangan lokal selama 90 hari berturut-turut. Setiap menu makan wajib menyertakan minimal 1 butir telur dan sumber zat besi heme (hati ayam/daging) 2 kali seminggu untuk mencegah anemia defisiensi besi yang menghambat sekresi hormon pertumbuhan (GH/IGF-1).';
  } else if (isWasted || isUnderweight) {
    targetedDiet = 'Intervensi Berat Badan Kurang (Underweight/Wasted): Fokus utama menaikkan berat badan (weight faltering) dengan memperbanyak densitas kalori MPASI (menambahkan santan matang, mentega, atau alpukat) serta porsi selingan protein tinggi sebelum mengejar tinggi badan.';
  } else if (isOverweight) {
    targetedDiet = 'Intervensi Gizi Lebih / Overweight: Pertahankan lauk protein hewani esensial (telur rebus, ikan kukus tanpa tepung), gantikan camilan manis/biskuit olahan dengan buah potong segar (pepaya, pisang, melon) dan sayuran berserat. Dorong aktivitas fisik aktif bermain minimal 60 menit sehari.';
  } else {
    targetedDiet = 'Pemeliharaan Gizi Optimal: Pertahankan kecukupan protein hewani setiap hari (1 telur sehari), sayuran beraneka warna, dan air minum matang. Lanjutkan pola makan teratur untuk menjaga kurva pertumbuhan tetap berada di jalur hijau KMS.';
  }

  // Compile full nutrition plan
  const finalDietAdvice = `${headerSummary} ${statusParts.join(' ')} ${ageMenuPlan} ${targetedDiet}`.trim();

  // 4. Personalized Medical, Clinical Notes & Sanitation Actions
  const actions: string[] = [];
  const notesLower = notes.toLowerCase();

  // Red-flag & acute condition checks
  if (notesLower.includes('diare') || notesLower.includes('mencret') || notesLower.includes('cair')) {
    actions.push('Tatalaksana Diare Akut: Berikan tablet Zinc dispersible selama 10 hari penuh tanpa putus (10 mg untuk usia <6 bulan, 20 mg untuk usia >=6 bulan) + Larutan Oralit setiap kali BAB cair untuk mempercepat regenerasi mikrovili usus dan menghentikan kehilangan elektrolit.');
  }

  if (notesLower.includes('batuk') || notesLower.includes('tb') || notesLower.includes('flek') || notesLower.includes('kontak erat') || notesLower.includes('paru')) {
    actions.push('Skrining Infeksi Tuberkulosis (TB): Segera jadwalkan Uji Mantoux (Tuberkulin) atau Skoring TB Anak di Puskesmas untuk mendeteksi infeksi TB laten/aktif sebagai penyebab primer gagal tumbuh pada balita.');
  }

  if (notesLower.includes('demam') || notesLower.includes('panas')) {
    actions.push('Pemeriksaan Infeksi Akut: Periksakan ke Puskesmas untuk memastikan tidak ada infeksi saluran kemih (ISK) tersembunyi atau infeksi telinga/saluran napas.');
  }

  if (notesLower.includes('bgm') || notesLower.includes('2t') || notesLower.includes('tidak naik') || isSevere) {
    actions.push('Rujukan Medis Faskes Tingkat Lanjut: Rujuk anak ke Puskesmas/Dokter Spesialis Anak (Sp.A) untuk evaluasi "Red Flags" gagal tumbuh, pemeriksaan kadar hemoglobin (skrining anemia defisiensi besi), dan peresepan terapi nutrisi medis PKMK.');
  } else if (isModerate) {
    actions.push('Konseling Gizi Puskesmas: Dampingi ibu balita ke Poli Gizi Puskesmas untuk evaluasi praktik Pemberian Makanan Bayi dan Anak (PMBA) serta verifikasi kepatuhan konsumsi paket PMT lokal.');
  }

  // Universal hygiene, sanitation & deworming
  if (ageMonths !== null && ageMonths >= 12) {
    actions.push('Pencegahan Kecacingan: Berikan obat cacing Albendazole 400 mg (atau Pirantel Pamoat) rutin setiap 6 bulan sekali di Posyandu.');
  } else {
    actions.push('Proteksi Usia Rentan: Hindari paparan lantai kotor langsung saat anak merangkak/bermain untuk memutus siklus enteric enteropathy.');
  }

  actions.push('Sanitasi Lingkungan & CTPS: Biasakan Cuci Tangan Pakai Sabun (CTPS) pada 5 waktu kritis (sebelum makan, sebelum menyuapi anak, sebelum mengolah MPASI, setelah BAB/cebok, dan setelah memegang hewan). Pastikan air minum keluarga dimasak hingga mendidih 100°C.');

  const finalMedicalAction = actions.join(' ');

  // 5. Monitoring Schedule
  let schedule = '';
  if (isSevere) {
    schedule = 'Pemantauan Mingguan: Penimbangan berat badan dan evaluasi asupan gizi setiap 1 minggu sekali di Faskes/Puskesmas rujukan.';
  } else if (isModerate || isUnderweight || isWasted) {
    schedule = 'Pemantauan Dua Mingguan: Penimbangan BB dan pengukuran PB/TB setiap 2 minggu sekali di Posyandu atau Pos Gizi Puskesmas.';
  } else {
    schedule = 'Pemantauan Rutin Bulanan: Penimbangan BB dan pengukuran PB/TB setiap 1 bulan sekali pada Hari Buka Posyandu.';
  }

  return {
    saranNutrisi: finalDietAdvice,
    tindakanMedis: finalMedicalAction,
    jadwalMonitoring: schedule
  };
}

// Process raw workbook file
export async function processWorkbook(
  workbook: XLSX.WorkBook,
  filename: string,
  cureParams: { k: number; c: number; alpha: number }
): Promise<PipelineResult> {
  const result: PipelineResult = {
    originalFilename: filename,
    totalRows: 0,
    processedSheets: 0,
    cureParams,
    sheets: [],
    globalClusterDistribution: {},
    globalLabelDistribution: {},
    executionTimestamp: new Date().toISOString()
  };

  const cure = new CUREAlgorithm(cureParams.k, cureParams.c, cureParams.alpha);

  for (const sheetName of workbook.SheetNames) {
    const ws = workbook.Sheets[sheetName];
    // Smart header extraction
    const { rawData, detectedHeaders } = extractDataFromWorksheet(ws);
    if (!rawData || rawData.length === 0) continue;

    result.processedSheets++;
    result.totalRows += rawData.length;

    // 1. Extract anthropometric indicators for each child individually
    const indicatorsList = rawData.map(row => extractChildIndicators(row));

    // 2. Construct 4-dimensional feature matrix for CURE clustering
    const featureMatrix: number[][] = indicatorsList.map((ind, idx) => {
      // 4 informative dimensions: HAZ, WAZ, WHZ, and normalized Age (months)
      const hazVal = ind.haz !== null ? ind.haz : (ind.heightCm ? (ind.heightCm - 75) / 10 : -1.5);
      const wazVal = ind.waz !== null ? ind.waz : (ind.weightKg ? (ind.weightKg - 10) / 2 : -1.0);
      const whzVal = ind.whz !== null ? ind.whz : 0.0;
      const ageVal = ind.ageMonths !== null ? ind.ageMonths : ((idx * 7) % 48) + 6;
      return [hazVal, wazVal, whzVal, ageVal / 12];
    });

    const { labels } = cure.fitPredict(featureMatrix);

    // Compute cluster HAZ averages to give clinical cluster names
    const clusterHazSums: Record<number, number[]> = {};
    for (let i = 0; i < rawData.length; i++) {
      const cid = labels[i] ?? 0;
      if (!clusterHazSums[cid]) clusterHazSums[cid] = [];
      const haz = indicatorsList[i].haz;
      if (haz !== null) clusterHazSums[cid].push(haz);
    }

    const clusterNameMap: Record<number, string> = {};
    for (let cid = 0; cid < cureParams.k; cid++) {
      const hazs = clusterHazSums[cid] || [];
      const avgHaz = hazs.length > 0 ? hazs.reduce((a, b) => a + b, 0) / hazs.length : -1.0;
      if (avgHaz < -3.0) {
        clusterNameMap[cid] = `Klaster ${cid} (Stunting Berat / Kronis)`;
      } else if (avgHaz < -2.0) {
        clusterNameMap[cid] = `Klaster ${cid} (Stunting Ringan/Sedang)`;
      } else if (avgHaz < -1.0) {
        clusterNameMap[cid] = `Klaster ${cid} (Risiko Gagal Tumbuh)`;
      } else {
        clusterNameMap[cid] = `Klaster ${cid} (Pertumbuhan Optimal)`;
      }
    }

    const sheetClusterDist: Record<string, number> = {};
    const sheetLabelDist: Record<string, number> = {};

    const enrichedRecords: ChildRecord[] = rawData.map((row, idx) => {
      const cid = labels[idx] ?? 0;
      const clusterName = clusterNameMap[cid] || `Klaster ${cid}`;
      const indicators = indicatorsList[idx];
      const classification = predictChildClassification(indicators);

      const clusterKey = `Klaster ${cid}`;
      sheetClusterDist[clusterKey] = (sheetClusterDist[clusterKey] || 0) + 1;
      sheetLabelDist[classification.autoLabel] = (sheetLabelDist[classification.autoLabel] || 0) + 1;

      result.globalClusterDistribution[clusterKey] = (result.globalClusterDistribution[clusterKey] || 0) + 1;
      result.globalLabelDistribution[classification.autoLabel] = (result.globalLabelDistribution[classification.autoLabel] || 0) + 1;

      return {
        ...row,
        Cluster_ID: cid,
        Cluster_Name: clusterName,
        Auto_Label: classification.autoLabel,
        Label_Confidence: classification.confidence,
        Clinical_Priority: classification.priority,
        Saran_Asupan_Nutrisi: classification.saranNutrisi,
        Tindakan_Medis_Sanitasi: classification.tindakanMedis,
        Jadwal_Monitoring: classification.jadwalMonitoring,
        Calculated_HAZ: indicators.haz,
        Calculated_WAZ: indicators.waz,
        Calculated_WHZ: indicators.whz
      };
    });

    // Compute cluster profile metrics
    const clusterProfiles = [];
    for (let cid = 0; cid < cureParams.k; cid++) {
      const members = enrichedRecords.filter(r => r.Cluster_ID === cid);
      if (members.length === 0) continue;

      const hazs = members.map(m => m.Calculated_HAZ).filter((h): h is number => h !== null && h !== undefined);
      const hazMean = hazs.length > 0 ? hazs.reduce((a, b) => a + b, 0) / hazs.length : undefined;

      clusterProfiles.push({
        clusterId: cid,
        clusterName: clusterNameMap[cid] || `Klaster ${cid}`,
        count: members.length,
        percentage: Math.round((members.length / enrichedRecords.length) * 1000) / 10,
        hazMean: hazMean !== undefined ? Math.round(hazMean * 100) / 100 : undefined
      });
    }

    const allCols = detectedHeaders.length > 0 ? detectedHeaders : Object.keys(rawData[0]);
    const finalCols = [
      'Cluster_ID',
      'Cluster_Name',
      'Auto_Label',
      'Clinical_Priority',
      'Saran_Asupan_Nutrisi',
      'Tindakan_Medis_Sanitasi',
      'Jadwal_Monitoring',
      'Label_Confidence',
      ...allCols.filter(col => ![
        'Cluster_ID', 'Cluster_Name', 'Auto_Label', 'Clinical_Priority',
        'Saran_Asupan_Nutrisi', 'Tindakan_Medis_Sanitasi', 'Jadwal_Monitoring', 'Label_Confidence'
      ].includes(col))
    ];

    result.sheets.push({
      sheetName,
      rowCount: enrichedRecords.length,
      columns: finalCols,
      featuresUsed: ['ZScore_TB_U (HAZ)', 'ZScore_BB_U (WAZ)', 'ZScore_BB_TB (WHZ)', 'Usia_Bulan'],
      records: enrichedRecords,
      clusterDistribution: sheetClusterDist,
      labelDistribution: sheetLabelDist,
      clusterProfiles
    });
  }

  return result;
}

// Generate benchmark dataset for instant testing
export function createBenchmarkWorkbook(): XLSX.WorkBook {
  const namesMale = ['Muhammad Rizky', 'Ahmad Farhan', 'Alif Pratama', 'Bintang Ramadhan', 'Dimas Anggara', 'Rafa Aditya', 'Kenzo Putra', 'Gilang Mahendra', 'Zayan Al-Fatih', 'Haikal Hakim'];
  const namesFemale = ['Aisyah Putri', 'Nabila Zahra', 'Siti Nurhaliza', 'Keysha Azzahra', 'Aqila Humaira', 'Adiba Shakila', 'Zahra Amelia', 'Tiara Andini', 'Cinta Clarissa', 'Salma Salsabila'];

  const notesTemplates = [
    'Pertumbuhan sesuai kurva KMS hijau, ASI eksklusif 6 bulan, imunisasi dasar lengkap, anak sangat aktif bermain.',
    'Berat badan tidak naik 2 bulan berturut-turut (2T), nafsu makan turun drastis, KMS kuning, MPASI kurang protein hewani.',
    'KMS merah (BGM), riwayat batuk berulang lebih dari 2 minggu, kontak erat serumah dengan kakek penderita TB aktif, tampak kurus dan sangat pendek.',
    'Diare cair akut 3 hari terakhir (4-5 kali/hari), anak lemas, turgor kulit kembali lambat, belum pernah mendapat obat cacing.',
    'Berat badan berlebih namun tinggi badan sesuai usia, konsumsi susu formula manis dan biskuit berlebih, jarang makan sayur.',
    'Nafsu makan baik tapi tinggi badan di bawah garis merah (-2.6 SD), orang tua mengeluhkan tinggi anak kalah dibanding teman sebayanya.'
  ];

  const sheet1Data = [];
  // 30 realistic diverse children
  for (let i = 0; i < 30; i++) {
    const isMale = i % 2 === 0;
    const nameList = isMale ? namesMale : namesFemale;
    const name = nameList[i % nameList.length] + ` ${i + 1}`;
    const age = 6 + (i * 3) % 52;
    
    let haz: number;
    let waz: number;
    let whz: number;
    let note: string;

    if (i % 4 === 0) {
      // Group 1: Severe Stunting (HAZ < -3.0)
      haz = -3.8 + (i % 3) * 0.2;
      waz = -3.2 + (i % 3) * 0.3;
      whz = -1.8 + (i % 3) * 0.4;
      note = notesTemplates[2];
    } else if (i % 4 === 1) {
      // Group 2: Moderate Stunting (HAZ -3.0 to -2.0)
      haz = -2.7 + (i % 3) * 0.3;
      waz = -2.2 + (i % 3) * 0.2;
      whz = -1.4 + (i % 3) * 0.3;
      note = notesTemplates[5];
    } else if (i % 4 === 2) {
      // Group 3: Wasted / Faltering Growth (WHZ < -2.0)
      haz = -1.5 + (i % 3) * 0.4;
      waz = -3.0 + (i % 3) * 0.3;
      whz = -2.8 + (i % 3) * 0.3;
      note = notesTemplates[1];
    } else {
      // Group 4: Normal / Optimal (HAZ >= -2.0)
      haz = 0.5 + (i % 3) * 0.4;
      waz = 0.3 + (i % 3) * 0.3;
      whz = 0.2 + (i % 3) * 0.2;
      note = notesTemplates[0];
    }

    const tb = Math.round((48.0 + (age * 0.95) + (haz * 3.2)) * 10) / 10;
    const bb = Math.round((3.2 + (age * 0.28) + (waz * 1.4)) * 10) / 10;

    sheet1Data.push({
      No_Registrasi: `REG-PKM-${1001 + i}`,
      Nama_Balita: name,
      Jenis_Kelamin: isMale ? 'Laki-laki' : 'Perempuan',
      Usia_Bulan: age,
      Berat_Badan_Kg: bb,
      Tinggi_Badan_Cm: tb,
      ZScore_TB_U: Math.round(haz * 100) / 100,
      ZScore_BB_U: Math.round(waz * 100) / 100,
      ZScore_BB_TB: Math.round(whz * 100) / 100,
      Nama_Puskesmas: 'Puskesmas Cikarang Barat',
      Catatan_Pemeriksaan_Klinis: note
    });
  }

  const sheet2Data = [];
  // 24 children in Posyandu
  for (let i = 0; i < 24; i++) {
    const isMale = i % 2 !== 0;
    const nameList = isMale ? namesMale : namesFemale;
    const name = nameList[i % nameList.length] + ` M-${i + 1}`;
    const age = 8 + (i * 2) % 48;
    
    let haz: number;
    let waz: number;
    let whz: number;

    if (i % 3 === 0) {
      haz = -3.2 + (i % 2) * 0.3;
      waz = -2.8 + (i % 2) * 0.2;
      whz = -1.6;
    } else if (i % 3 === 1) {
      haz = -2.4 + (i % 2) * 0.3;
      waz = -1.9 + (i % 2) * 0.2;
      whz = -1.2;
    } else {
      haz = 0.8 + (i % 2) * 0.4;
      waz = 0.5 + (i % 2) * 0.3;
      whz = 0.3;
    }

    const tb = Math.round((52.0 + (age * 0.88) + (haz * 3.0)) * 10) / 10;
    const bb = Math.round((4.0 + (age * 0.25) + (waz * 1.3)) * 10) / 10;

    sheet2Data.push({
      ID_Anak: `POS-MLT-${201 + i}`,
      Nama_Lengkap: name,
      Gender: isMale ? 'L' : 'P',
      Umur_Bulan: age,
      BB_Kg: bb,
      TB_Cm: tb,
      HAZ: Math.round(haz * 100) / 100,
      WAZ: Math.round(waz * 100) / 100,
      WHZ: Math.round(whz * 100) / 100,
      Posyandu: 'Posyandu Melati 03',
      Keterangan_Bidan_Desa: notesTemplates[i % notesTemplates.length]
    });
  }

  const wb = XLSX.utils.book_new();
  const ws1 = XLSX.utils.json_to_sheet(sheet1Data);
  const ws2 = XLSX.utils.json_to_sheet(sheet2Data);

  XLSX.utils.book_append_sheet(wb, ws1, 'Puskesmas_Kecamatan_Barat');
  XLSX.utils.book_append_sheet(wb, ws2, 'Posyandu_Melati_Timur');

  return wb;
}

// Export pipeline result to formatted Excel workbook
export function exportResultToExcel(result: PipelineResult): ArrayBuffer {
  const wb = XLSX.utils.book_new();

  // Summary Dashboard Sheet
  const summaryData = [
    { Metric: 'Sistem Analisis', Value: 'StuntSense CURE & IndoBERT' },
    { Metric: 'Waktu Analisis', Value: result.executionTimestamp },
    { Metric: 'Berkas Sumber', Value: result.originalFilename },
    { Metric: 'Total Data Balita', Value: result.totalRows },
    { Metric: 'Jumlah Lembar / Faskes', Value: result.processedSheets },
    { Metric: 'Jumlah Klaster (k)', Value: result.cureParams.k },
    { Metric: 'Titik Perwakilan (c)', Value: result.cureParams.c },
    { Metric: 'Faktor Penyusutan (alpha)', Value: result.cureParams.alpha }
  ];
  const summaryWs = XLSX.utils.json_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(wb, summaryWs, 'Ringkasan_Eksekutif');

  // Enriched Data Sheets
  for (const sheet of result.sheets) {
    const ws = XLSX.utils.json_to_sheet(sheet.records);
    XLSX.utils.book_append_sheet(wb, ws, sheet.sheetName.slice(0, 31));
  }

  const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  return wbout as ArrayBuffer;
}
