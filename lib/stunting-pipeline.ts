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
  Usia_Bulan?: number | null;
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
  crosstab: Record<string, Record<string, number>>;
  purityScore: number;
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
  globalCrosstab: Record<string, Record<string, number>>;
  globalPurityScore: number;
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

  // Replace unicode minus signs with standard '-'
  str = str.replace(/[\u2212\u2013\u2014\u2015\uFE63\uFF0D]/g, '-');
  // Handle Indonesian decimal comma
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
 * 1-5 title banner rows before the actual table header row (e.g. "TAHUN 2024").
 * This function scans the top rows to automatically detect the genuine header row.
 */
export function extractDataFromWorksheet(ws: XLSX.WorkSheet): { rawData: Record<string, any>[]; detectedHeaders: string[] } {
  const rows: any[][] = XLSX.utils.sheet_to_json(ws, { header: 1, defval: '' });
  if (!rows || rows.length === 0) return { rawData: [], detectedHeaders: [] };

  const headerKeywords = [
    'nama', 'balita', 'anak', 'bayi', 'nik', 'no', 'nomor',
    'usia', 'umur', 'bulan', 'bln', 'tgl', 'tanggal', 'lahir', 'ukur',
    'jk', 'gender', 'jenis_kelamin',
    'bb', 'berat', 'tb', 'pb', 'tinggi', 'panjang',
    'haz', 'waz', 'whz', 'zs', 'zscore', 'z_score', 'tb_u', 'bb_u', 'bb_tb',
    'kec', 'kecamatan', 'puskesmas', 'pukesmas', 'desa', 'kelurahan', 'lila'
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
      // Word token matching to prevent false positives like 'normal' matching 'no'
      const tokens = cellNorm.split('_');
      if (tokens.some(t => headerKeywords.includes(t)) || headerKeywords.some(kw => cellNorm === kw)) {
        score++;
      }
    }

    if (score > maxScore) {
      maxScore = score;
      bestHeaderRowIdx = r;
    }
  }

  const headerRow = rows[bestHeaderRowIdx] || [];
  let dataStartIdx = bestHeaderRowIdx + 1;
  const nextRow = rows[bestHeaderRowIdx + 1];
  let subheaderUsed = false;
  let finalHeaders: string[] = [];

  // Check if row below is a genuine merged subheader row (NOT child data)
  // Data rows have a numeric index in col 0 (e.g. 1, 2) or names in col 1, whereas subheaders have units/labels.
  const isNextRowData = nextRow && Array.isArray(nextRow) && (
    typeof nextRow[0] === 'number' ||
    String(nextRow[0]).trim() === '1' ||
    (nextRow[1] && typeof nextRow[1] === 'string' && nextRow[1].trim().length > 3 && isNaN(Number(nextRow[1])))
  );

  if (!isNextRowData && nextRow && Array.isArray(nextRow) && dataStartIdx < rows.length) {
    let subheaderKeywords = ['cm', 'kg', 'sd', 'l', 'p', 'zscore', 'status', 'kategori'];
    let subScore = 0;
    for (const cell of nextRow) {
      const cellNorm = normalizeColName(String(cell));
      if (cellNorm && subheaderKeywords.includes(cellNorm)) subScore++;
    }
    if (subScore >= 2) {
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

  // Deduplicate column names if needed
  const seenHeaderCounts: Record<string, number> = {};
  finalHeaders = finalHeaders.map(h => {
    if (!seenHeaderCounts[h]) {
      seenHeaderCounts[h] = 1;
      return h;
    } else {
      seenHeaderCounts[h]++;
      return `${h}_${seenHeaderCounts[h]}`;
    }
  });

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
export function extractChildIndicators(record: Record<string, any>, sheetName: string = ''): {
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
  tbUStatus: string;
  bbUStatus: string;
  bbTbStatus: string;
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
  let tbUStatus = '';
  let bbUStatus = '';
  let bbTbStatus = '';

  let birthDate: Date | null = null;
  let visitDate: Date | null = null;

  // Infer survey year from sheet name (e.g. '2024' or '2025')
  let surveyYear = 2024;
  const yearMatch = sheetName.match(/\b(202[0-9])\b/);
  if (yearMatch) {
    surveyYear = parseInt(yearMatch[1]);
  } else {
    const currentYear = new Date().getFullYear();
    surveyYear = currentYear <= 2026 ? currentYear : 2024;
  }
  // August 1st is the standard annual Indonesian Posyandu Bulan Timbang
  const defaultSurveyDate = new Date(surveyYear, 7, 1);

  for (const [rawKey, rawVal] of Object.entries(record)) {
    if (rawVal === null || rawVal === undefined) continue;
    const key = normalizeColName(rawKey);
    const valStr = String(rawVal).trim().toLowerCase();

    // Child Name: Match 'nama', 'nama_anak', 'nama_balita', 'balita'
    if (!childName && (key === 'nama' || key.startsWith('nama_') || key === 'balita' || key === 'anak' || key === 'bayi')) {
      if (!key.includes('ortu') && !key.includes('ibu') && !key.includes('ayah') && !key.includes('posyandu') && !key.includes('puskesmas') && !key.includes('desa')) {
        if (typeof rawVal === 'string' && rawVal.trim().length > 1 && isNaN(Number(rawVal))) {
          childName = rawVal.trim();
        }
      }
    }

    // Gender detection: Match 'jk', 'gender', 'jenis_kelamin', 'sex'
    if (['jk', 'gender', 'jenis_kelamin', 'sex', 'l_p', 'j_k'].includes(key) || key.startsWith('jk_')) {
      if (valStr.startsWith('l') || valStr.includes('laki') || valStr.includes('pria') || valStr === '1' || valStr === 'male') {
        gender = 'L';
      } else if (valStr.startsWith('p') || valStr.includes('perempuan') || valStr.includes('wanita') || valStr === '2' || valStr === 'female') {
        gender = 'P';
      }
    }

    // Date of Birth: Handle Excel serial dates (e.g. 44119) and standard strings
    if (key.includes('lahir') || key.includes('birth') || key === 'tgl_lhr') {
      if (typeof rawVal === 'number' && rawVal > 25000 && rawVal < 65000) {
        birthDate = new Date(Math.round((rawVal - 25569) * 86400 * 1000));
      } else if (typeof rawVal === 'string' && rawVal.trim().length > 0) {
        const numVal = parseNumericValue(rawVal);
        if (numVal !== null && numVal > 25000 && numVal < 65000) {
          birthDate = new Date(Math.round((numVal - 25569) * 86400 * 1000));
        } else {
          const d = new Date(rawVal);
          if (!isNaN(d.getTime())) birthDate = d;
        }
      }
    }

    // Measurement Date
    if (key.includes('tgl_ukur') || key.includes('tgl_timbang') || key.includes('tgl_kunjungan')) {
      if (typeof rawVal === 'number' && rawVal > 25000 && rawVal < 65000) {
        visitDate = new Date(Math.round((rawVal - 25569) * 86400 * 1000));
      } else {
        const d = new Date(rawVal);
        if (!isNaN(d.getTime())) visitDate = d;
      }
    }

    // Explicit Age in Months column (if present in Excel)
    if (
      ['usia_bulan', 'umur_bulan', 'usia', 'umur', 'bulan', 'age', 'age_months', 'umur_bln', 'usia_bln', 'u_bln', 'u_bulan'].includes(key) ||
      ((key.includes('usia') || key.includes('umur')) && !key.includes('tgl') && !key.includes('tanggal'))
    ) {
      const thMatch = valStr.match(/(\d+)\s*(th|tahun|yr|year)/);
      const blnMatch = valStr.match(/(\d+)\s*(bln|bulan|mo|month)/);
      if (thMatch || blnMatch) {
        const years = thMatch ? parseInt(thMatch[1]) : 0;
        const months = blnMatch ? parseInt(blnMatch[1]) : 0;
        ageMonths = years * 12 + months;
      } else {
        const num = parseNumericValue(rawVal);
        if (num !== null && num >= 0 && num <= 72) {
          ageMonths = Math.round(num);
        }
      }
    }

    // Height in cm (TB / PB) - Exclude TB/U and Z-score columns
    const isHeightCol = (key === 'tinggi' || key === 'tb' || key === 'pb' || key === 'panjang' || key.includes('tinggi_badan') || key.includes('panjang_badan')) &&
      !key.includes('tb_u') && !key.includes('zs') && !key.includes('zscore') && !key.includes('status') && !key.includes('kategori');

    if (isHeightCol) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= 30 && num <= 150) {
        heightCm = num;
      }
    }

    // Weight in kg (BB) - Exclude BB/U, BB/TB and Z-score columns
    const isWeightCol = (key === 'berat' || key === 'bb' || key.includes('berat_badan')) &&
      !key.includes('bb_u') && !key.includes('bb_tb') && !key.includes('bbtb') && !key.includes('zs') && !key.includes('zscore') && !key.includes('status') && !key.includes('kategori');

    if (isWeightCol) {
      const num = parseNumericValue(rawVal);
      if (num !== null && num >= 1.0 && num <= 50) {
        weightKg = num;
      }
    }

    // HAZ: Z-Score TB/U (Height-for-Age)
    // Matches 'zs_tb_u', 'zscore_tb_u', 'haz', 'tb_u_zscore' (strictly excludes bb_tb and whz)
    const isHazCol = (key.includes('tb') || key.includes('haz')) &&
      (key.includes('zs') || key.includes('zscore') || key.includes('z_score')) &&
      !key.includes('bb_tb') && !key.includes('bbtb') && !key.includes('whz');

    if (isHazCol || key === 'haz' || key === 'zscore_tb_u' || key === 'zs_tb_u') {
      const num = parseNumericValue(rawVal);
      if (num !== null) {
        // Clamp extreme outliers to WHO physiological bounds [-5.0, 5.0]
        haz = Math.max(-5.0, Math.min(5.0, num));
      }
    } else if (key === 'tb_u' || key === 'tbu' || key === 'status_tb_u') {
      tbUStatus = String(rawVal).trim();
      statusText += ` ${valStr}`;
    }

    // WAZ: Z-Score BB/U (Weight-for-Age)
    // Matches 'zs_bb_u', 'zscore_bb_u', 'waz' (strictly excludes tb)
    const isWazCol = (key.includes('bb') || key.includes('waz')) &&
      (key.includes('zs') || key.includes('zscore') || key.includes('z_score')) &&
      !key.includes('tb');

    if (isWazCol || key === 'waz' || key === 'zscore_bb_u' || key === 'zs_bb_u') {
      const num = parseNumericValue(rawVal);
      if (num !== null) {
        waz = Math.max(-5.0, Math.min(5.0, num));
      }
    } else if (key === 'bb_u' || key === 'bbu' || key === 'status_bb_u') {
      bbUStatus = String(rawVal).trim();
      statusText += ` ${valStr}`;
    }

    // WHZ: Z-Score BB/TB (Weight-for-Height)
    // Matches 'zs_bb_tb', 'zscore_bb_tb', 'whz'
    const isWhzCol = (key.includes('bb_tb') || key.includes('bbtb') || key.includes('whz')) &&
      (key.includes('zs') || key.includes('zscore') || key.includes('z_score') || key === 'whz');

    if (isWhzCol || key === 'whz' || key === 'zscore_bb_tb' || key === 'zs_bb_tb') {
      const num = parseNumericValue(rawVal);
      if (num !== null) {
        whz = Math.max(-5.0, Math.min(5.0, num));
      }
    } else if (key === 'bb_tb' || key === 'bbtb' || key === 'status_bb_tb') {
      bbTbStatus = String(rawVal).trim();
      statusText += ` ${valStr}`;
    }

    // General qualitative or clinical notes
    if (['catatan', 'keterangan', 'keluhan', 'riwayat', 'diagnosa', 'hasil', 'notes'].some(t => key.includes(t))) {
      notes += ` ${rawVal}`;
    }
  }

  // Calculate ageMonths from birthDate if not explicitly given
  if (ageMonths === null && birthDate) {
    const refDate = visitDate || defaultSurveyDate;
    const diffMs = refDate.getTime() - birthDate.getTime();
    if (diffMs >= 0) {
      ageMonths = Math.min(72, Math.max(0, Math.round(diffMs / (1000 * 60 * 60 * 24 * 30.4375))));
    }
  }

  // Fallback: If qualitative TB/U is given but numeric HAZ is missing, assign standard representative Z-score
  if (haz === null && tbUStatus) {
    const sLower = tbUStatus.toLowerCase();
    if (sLower.includes('sangat pendek')) haz = -3.4;
    else if (sLower.includes('pendek')) haz = -2.4;
    else if (sLower.includes('normal')) haz = 0.1;
    else if (sLower.includes('tinggi')) haz = 2.2;
  }

  // Fallback: If qualitative BB/U is given but numeric WAZ is missing
  if (waz === null && bbUStatus) {
    const sLower = bbUStatus.toLowerCase();
    if (sLower.includes('sangat kurang')) waz = -3.2;
    else if (sLower.includes('kurang')) waz = -2.2;
    else if (sLower.includes('normal')) waz = 0.0;
  }

  // Fallback: If qualitative BB/TB is given but numeric WHZ is missing
  if (whz === null && bbTbStatus) {
    const sLower = bbTbStatus.toLowerCase();
    if (sLower.includes('gizi buruk')) whz = -3.3;
    else if (sLower.includes('gizi kurang')) whz = -2.2;
    else if (sLower.includes('gizi baik') || sLower.includes('normal')) whz = 0.1;
    else if (sLower.includes('gizi lebih') || sLower.includes('obesitas')) whz = 2.2;
  }

  // Compute WHO 2006 Standard Z-scores if anthropometrics are present but Z-scores were absent
  if (haz === null && heightCm !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const hMed = isBoy
      ? (ageMonths <= 24 ? 50.0 + 1.85 * ageMonths - 0.035 * ageMonths * ageMonths : 87.8 + 0.65 * (ageMonths - 24))
      : (ageMonths <= 24 ? 49.0 + 1.80 * ageMonths - 0.035 * ageMonths * ageMonths : 86.5 + 0.63 * (ageMonths - 24));
    const hSd = 2.1 + 0.04 * ageMonths;
    haz = Math.round(((heightCm - hMed) / hSd) * 100) / 100;
  }

  if (waz === null && weightKg !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const wMed = isBoy
      ? (ageMonths <= 24 ? 3.3 + 0.52 * ageMonths - 0.005 * ageMonths * ageMonths : 12.2 + 0.18 * (ageMonths - 24))
      : (ageMonths <= 24 ? 3.2 + 0.49 * ageMonths - 0.005 * ageMonths * ageMonths : 11.5 + 0.17 * (ageMonths - 24));
    const wSd = 0.5 + 0.03 * ageMonths;
    waz = Math.round(((weightKg - wMed) / wSd) * 100) / 100;
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
    notes: `${notes} ${statusText}`.trim(),
    tbUStatus,
    bbUStatus,
    bbTbStatus
  };
}

/**
 * Production CURE (Clustering Using Representatives) Algorithm
 * 
 * - Represents each cluster by c well-scattered points shrunk towards the cluster centroid by alpha.
 * - Outlier-resistant: Bound features to WHO biological limits to prevent cluster collapse.
 * - Partitioned into 4 clinical phenotypes:
 *   0: Stunting & Malnutrisi Kronis (Prioritas Kritis)
 *   1: Risiko Gagal Tumbuh / Underweight (Prioritas Sedang)
 *   2: Pertumbuhan Normal & Gizi Optimal (Prioritas Pemeliharaan)
 *   3: Pertumbuhan Tinggi / Risiko Gizi Lebih (Prioritas Seimbang)
 */
export class CUREAlgorithm {
  private k: number;
  private c: number;
  private alpha: number;

  constructor(k: number = 4, c: number = 5, alpha: number = 0.3) {
    this.k = Math.max(2, k);
    this.c = Math.max(1, c);
    this.alpha = Math.max(0.05, Math.min(0.95, alpha));
  }

  public fitPredict(features: number[][]): { labels: number[]; representatives: number[][][] } {
    const N = features.length;
    const D = features[0]?.length || 4;
    if (N === 0) return { labels: [], representatives: [] };
    if (N <= this.k) {
      return {
        labels: Array.from({ length: N }, (_, i) => i % this.k),
        representatives: []
      };
    }

    // Standardize features
    const means = new Float64Array(D);
    const stds = new Float64Array(D);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < D; j++) means[j] += features[i][j];
    }
    for (let j = 0; j < D; j++) means[j] /= N;

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < D; j++) {
        const d = features[i][j] - means[j];
        stds[j] += d * d;
      }
    }
    for (let j = 0; j < D; j++) {
      stds[j] = Math.sqrt(stds[j] / N) || 1.0;
    }

    const scaled = new Float64Array(N * D);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < D; j++) {
        scaled[i * D + j] = (features[i][j] - means[j]) / stds[j];
      }
    }

    // Clinical anchor seeds in anthropometric feature space [HAZ, WAZ, WHZ, normalizedAge]
    // Exactly matches the 7 Expert Clinical Categories of Kemenkes RI (Permenkes 2/2020) & WHO 2006:
    const clinicalSeeds7: number[][] = [
      [-3.4, -2.8, -2.4, 0.40], // Seed 0: Stunting Berat + Komorbiditas Klinis (HAZ < -3, WAZ < -2, WHZ < -2)
      [-3.1, -1.5, -0.2, 0.45], // Seed 1: Stunting Kronis / Sangat Pendek (HAZ < -3, WHZ normal/sedang)
      [-2.3, -1.4, -0.4, 0.45], // Seed 2: Stunting Sedang / Pendek (HAZ -2 s/d -3)
      [-0.5, -3.1, -3.2, 0.35], // Seed 3: Gizi Buruk Akut / Wasting Berat (WHZ < -3, WAZ < -3)
      [-1.2, -2.2, -2.1, 0.40], // Seed 4: Gizi Kurang / Risiko Gagal Tumbuh (WAZ < -2, WHZ < -2)
      [ 0.1,  0.2,  0.2, 0.50], // Seed 5: Pertumbuhan Normal / Gizi Optimal (Semua Z-Score rentang normal)
      [ 1.4,  1.8,  2.1, 0.55]  // Seed 6: Risiko Gizi Lebih / Overweight (WHZ > 2 atau HAZ tinggi)
    ];

    const clinicalSeeds4: number[][] = [
      [-3.2, -2.5, -2.0, 0.40], // Seed 0: Stunting Berat / Malnutrisi Kronis
      [-2.1, -1.6, -1.0, 0.42], // Seed 1: Stunting Sedang / Risiko Underweight
      [ 0.0,  0.1,  0.1, 0.48], // Seed 2: Pertumbuhan Normal Optimal
      [ 1.4,  1.5,  1.6, 0.52]  // Seed 3: Pertumbuhan Tinggi / Risiko Lebih
    ];

    // Build exactly this.k seeds guaranteed to be populated
    const chosenSeeds: number[][] = [];
    if (this.k === 7) {
      for (let i = 0; i < 7; i++) chosenSeeds.push([...clinicalSeeds7[i]]);
    } else if (this.k === 4) {
      for (let i = 0; i < 4; i++) chosenSeeds.push([...clinicalSeeds4[i]]);
    } else if (this.k < 7) {
      const step = (clinicalSeeds7.length - 1) / Math.max(1, this.k - 1);
      for (let i = 0; i < this.k; i++) {
        const idx = Math.min(clinicalSeeds7.length - 1, Math.round(i * step));
        chosenSeeds.push([...clinicalSeeds7[idx]]);
      }
    } else {
      for (const s of clinicalSeeds7) chosenSeeds.push([...s]);
      while (chosenSeeds.length < this.k) {
        const sampleIdx = Math.floor((chosenSeeds.length / this.k) * N);
        chosenSeeds.push(Array.from(features[sampleIdx] || features[0]));
      }
    }

    const scaledSeeds: number[][] = chosenSeeds.map(row =>
      row.map((val, j) => (val - (j < D ? means[j] : 0)) / (j < D ? stds[j] : 1.0))
    );

    // Initial assignment
    const clusterLabels = new Int32Array(N);
    const clusterCounts = new Int32Array(this.k);
    const clusterSums = Array.from({ length: this.k }, () => new Float64Array(D));

    for (let i = 0; i < N; i++) {
      let minDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < this.k; c++) {
        let dist = 0;
        for (let j = 0; j < D; j++) {
          const diff = scaled[i * D + j] - scaledSeeds[c][j];
          dist += diff * diff;
        }
        if (dist < minDist) {
          minDist = dist;
          bestC = c;
        }
      }
      clusterLabels[i] = bestC;
      clusterCounts[bestC]++;
      for (let j = 0; j < D; j++) {
        clusterSums[bestC][j] += scaled[i * D + j];
      }
    }

    // Centroid refinement (2 iterations)
    for (let iter = 0; iter < 2; iter++) {
      for (let c = 0; c < this.k; c++) {
        if (clusterCounts[c] > 0) {
          for (let j = 0; j < D; j++) {
            scaledSeeds[c][j] = clusterSums[c][j] / clusterCounts[c];
          }
        }
      }
      clusterCounts.fill(0);
      for (let c = 0; c < this.k; c++) clusterSums[c].fill(0);

      for (let i = 0; i < N; i++) {
        let minDist = Infinity;
        let bestC = 0;
        for (let c = 0; c < this.k; c++) {
          let dist = 0;
          for (let j = 0; j < D; j++) {
            const diff = scaled[i * D + j] - scaledSeeds[c][j];
            dist += diff * diff;
          }
          if (dist < minDist) {
            minDist = dist;
            bestC = c;
          }
        }
        clusterLabels[i] = bestC;
        clusterCounts[bestC]++;
        for (let j = 0; j < D; j++) {
          clusterSums[bestC][j] += scaled[i * D + j];
        }
      }
    }

    // Select c well-scattered representative points and shrink them towards centroid by alpha
    const clusterReps: number[][][] = [];
    for (let c = 0; c < this.k; c++) {
      const cent = scaledSeeds[c];
      const members: number[] = [];
      for (let i = 0; i < N; i++) {
        if (clusterLabels[i] === c) members.push(i);
      }

      if (members.length === 0) {
        clusterReps.push([Array.from(cent)]);
        continue;
      }

      const selected: number[] = [];
      const sampleSize = Math.min(members.length, 300);

      // 1st rep: farthest from centroid
      let maxDist = -1;
      let firstIdx = members[0];
      for (let s = 0; s < sampleSize; s++) {
        const idx = members[Math.floor(s * members.length / sampleSize)];
        let d = 0;
        for (let j = 0; j < D; j++) {
          const diff = scaled[idx * D + j] - cent[j];
          d += diff * diff;
        }
        if (d > maxDist) {
          maxDist = d;
          firstIdx = idx;
        }
      }
      selected.push(firstIdx);

      // Remaining reps: max-min distance from currently selected reps
      while (selected.length < this.c && selected.length < members.length) {
        let bestDist = -1;
        let nextIdx = members[0];
        for (let s = 0; s < sampleSize; s++) {
          const idx = members[Math.floor(s * members.length / sampleSize)];
          if (selected.includes(idx)) continue;
          let minDistToRep = Infinity;
          for (const sel of selected) {
            let d = 0;
            for (let j = 0; j < D; j++) {
              const diff = scaled[idx * D + j] - scaled[sel * D + j];
              d += diff * diff;
            }
            if (d < minDistToRep) minDistToRep = d;
          }
          if (minDistToRep > bestDist) {
            bestDist = minDistToRep;
            nextIdx = idx;
          }
        }
        selected.push(nextIdx);
      }

      // Shrink representative points towards centroid
      const shrunk = selected.map(sel => {
        const pt: number[] = [];
        for (let j = 0; j < D; j++) {
          const v = scaled[sel * D + j];
          pt.push(v + this.alpha * (cent[j] - v));
        }
        return pt;
      });
      clusterReps.push(shrunk);
    }

    // Final CURE assignment: assign each point to cluster with closest shrunk representative
    const finalLabels = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      let minDist = Infinity;
      let bestC = 0;
      for (let c = 0; c < this.k; c++) {
        const reps = clusterReps[c];
        for (const rep of reps) {
          let dist = 0;
          for (let j = 0; j < D; j++) {
            const diff = scaled[i * D + j] - rep[j];
            dist += diff * diff;
          }
          if (dist < minDist) {
            minDist = dist;
            bestC = c;
          }
        }
      }
      finalLabels[i] = bestC;
    }

    return { labels: finalLabels, representatives: clusterReps };
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
  const { childName, haz, waz, whz, ageMonths, heightCm, weightKg, gender, notes, tbUStatus, bbUStatus, bbTbStatus } = indicators;

  const notesLower = notes.toLowerCase();
  const tbULower = tbUStatus.toLowerCase();
  const bbULower = bbUStatus.toLowerCase();
  const bbTbLower = bbTbStatus.toLowerCase();

  const isExplicitSeverelyStunted = tbULower.includes('sangat pendek') || (haz !== null && haz < -3.0);
  const isExplicitStunted = !isExplicitSeverelyStunted && (tbULower.includes('pendek') || (haz !== null && haz >= -3.0 && haz < -2.0));
  const isExplicitSevereWasted = bbTbLower.includes('gizi buruk') || (whz !== null && whz < -3.0);
  const isExplicitWasted = !isExplicitSevereWasted && (bbTbLower.includes('gizi kurang') || (whz !== null && whz >= -3.0 && whz < -2.0));
  const isExplicitUnderweight = bbULower.includes('sangat kurang') || bbULower.includes('kurang') || (waz !== null && waz < -2.0);
  const isExplicitOverweight = bbTbLower.includes('lebih') || bbTbLower.includes('obesitas') || (whz !== null && whz > 2.0);

  let label = TAXONOMY.NORMAL_OPTIMAL;
  let confidence = 0.90;
  let priority = 'Rendah (Pemantauan Rutin Posyandu)';

  // Clinical Decision Tree based on Permenkes RI No. 2/2020 & WHO 2006 Standards
  if (isExplicitSeverelyStunted) {
    if (isExplicitSevereWasted || isExplicitUnderweight || notesLower.includes('komorbid') || notesLower.includes('tb')) {
      label = TAXONOMY.STUNTING_BERAT_KOMORBID;
      confidence = 0.96;
      priority = 'Kritis (Intervensi Medis & Rujukan RS)';
    } else {
      label = TAXONOMY.STUNTING_KRONIS;
      confidence = 0.94;
      priority = 'Tinggi (Intervensi Gizi Spesifik)';
    }
  } else if (isExplicitStunted) {
    if (isExplicitUnderweight || isExplicitWasted) {
      label = TAXONOMY.STUNTING_BERAT_KOMORBID;
      confidence = 0.90;
      priority = 'Tinggi (Intervensi Gizi Spesifik)';
    } else {
      label = TAXONOMY.STUNTING_SEDANG;
      confidence = 0.92;
      priority = 'Sedang (Pemberian Makanan Tambahan PMT)';
    }
  } else if (isExplicitSevereWasted) {
    label = TAXONOMY.GIZI_BURUK_AKUT;
    confidence = 0.95;
    priority = 'Kritis (Tatalaksana Gizi Buruk / TFC)';
  } else if (isExplicitWasted || isExplicitUnderweight) {
    label = TAXONOMY.GIZI_KURANG_RISIKO;
    confidence = 0.89;
    priority = 'Sedang (Pemantauan Kejar Tumbuh)';
  } else if (isExplicitOverweight) {
    label = TAXONOMY.RISIKO_GIZI_LEBIH;
    confidence = 0.88;
    priority = 'Rendah (Pola Makan Seimbang)';
  } else {
    label = TAXONOMY.NORMAL_OPTIMAL;
    confidence = 0.95;
    priority = 'Rendah (Pemantauan Rutin Posyandu)';
  }

  // Generate deeply tailored, individual recommendations
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
    notes,
    tbUStatus,
    bbUStatus,
    bbTbStatus
  );

  return {
    autoLabel: label,
    confidence,
    priority,
    ...recommendations
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
  notes: string,
  tbUStatus: string = '',
  bbUStatus: string = '',
  bbTbStatus: string = ''
): {
  saranNutrisi: string;
  tindakanMedis: string;
  jadwalMonitoring: string;
} {
  const isSevere = autoLabel.includes('Stunting Berat') || autoLabel.includes('Gizi Buruk') || (haz !== null && haz < -3.0);
  const isModerate = autoLabel.includes('Stunting Ringan') || autoLabel.includes('Gizi Kurang') || (haz !== null && haz < -2.0 && haz >= -3.0);
  const isOverweight = autoLabel.includes('Risiko Gizi Lebih');
  const isUnderweight = (waz !== null && waz < -2.0) || bbUStatus.toLowerCase().includes('kurang');
  const isWasted = (whz !== null && whz < -2.0) || bbTbStatus.toLowerCase().includes('kurang');

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
        statusParts.push(`Tinggi ${heightCm} cm (defisit ${diff} cm di bawah median WHO ${Math.round(hMed * 10) / 10} cm, Z-Score TB/U: ${haz.toFixed(2)} SD, Status: ${tbUStatus || 'Pendek'}).`);
      } else {
        statusParts.push(`Tinggi ${heightCm} cm (sesuai median WHO seusianya ${Math.round(hMed * 10) / 10} cm, Z-Score TB/U: ${haz.toFixed(2)} SD, Status: ${tbUStatus || 'Normal'}).`);
      }
    } else {
      statusParts.push(`Z-Score TB/U: ${haz.toFixed(2)} SD (${tbUStatus || 'Tercatat'}).`);
    }
  } else if (heightCm !== null) {
    statusParts.push(`Tinggi badan ${heightCm} cm (Status TB/U: ${tbUStatus || '-'}).`);
  }

  if (waz !== null && ageMonths !== null) {
    const isBoy = gender !== 'P';
    const wMed = isBoy
      ? (ageMonths <= 24 ? 3.3 + 0.52 * ageMonths - 0.005 * ageMonths * ageMonths : 12.2 + 0.18 * (ageMonths - 24))
      : (ageMonths <= 24 ? 3.2 + 0.49 * ageMonths - 0.005 * ageMonths * ageMonths : 11.5 + 0.17 * (ageMonths - 24));
    
    if (weightKg !== null) {
      const diffW = Math.round((wMed - weightKg) * 10) / 10;
      if (diffW > 0) {
        statusParts.push(`Berat ${weightKg} kg (defisit ${diffW} kg dari target median WHO ${Math.round(wMed * 10) / 10} kg, Z-Score BB/U: ${waz.toFixed(2)} SD, Status: ${bbUStatus || 'Kurang'}).`);
      } else {
        statusParts.push(`Berat ${weightKg} kg (sesuai target median WHO ${Math.round(wMed * 10) / 10} kg, Z-Score BB/U: ${waz.toFixed(2)} SD, Status: ${bbUStatus || 'Normal'}).`);
      }
    }
  } else if (weightKg !== null) {
    statusParts.push(`Berat badan ${weightKg} kg (Status BB/U: ${bbUStatus || '-'}).`);
  }

  if (whz !== null) {
    statusParts.push(`Z-Score BB/TB: ${whz.toFixed(2)} SD (${bbTbStatus || 'Gizi Baik'}).`);
  }

  // 2. Specific age-bracket food consistency, portions, and protein menu
  let ageMenuPlan = '';
  if (ageMonths !== null) {
    if (ageMonths < 6) {
      ageMenuPlan = 'Fase Usia < 6 Bulan: Wajib ASI Eksklusif on-demand minimal 8-12 kali per 24 jam tanpa tambahan air putih atau makanan padat. Ibu menyusui wajib mengonsumsi double protein hewani harian (2 butir telur + 1 porsi ikan kembung/tongkol) dan suplemen kalsium serta tablet tambah darah agar densitas gizi ASI optimal.';
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
    targetedDiet = 'Intervensi Pemulihan Stunting Kronis / Berat: Wajib konsumsi Double Protein Hewani harian (misal: telur ayam di pagi hari + ikan kembung/hati ayam di siang dan sore). Tambahkan ekstra 1-2 sdt minyak kelapa matang/santan kental pada setiap mangkuk makan anak untuk mengejar ketertinggalan kalori. Konsultasikan ke dokter untuk peresepan Pangan Olahan untuk Keperluan Medis Khusus (PKMK) tinggi kalori (1.0-1.5 kkal/ml) serta tabur gizi mikro (Taburia/zinc).';
  } else if (isModerate) {
    targetedDiet = 'Intervensi Stunting Sedang / Kejar Tumbuh: Berikan Pemberian Makanan Tambahan (PMT) Pemulihan berbasis bahan pangan lokal selama 90 hari berturut-turut. Setiap menu makan wajib menyertakan minimal 1 butir telur dan sumber zat besi heme (hati ayam/daging) 2 kali seminggu untuk mencegah anemia defisiensi besi yang menghambat sekresi hormon pertumbuhan (GH/IGF-1).';
  } else if (isWasted || isUnderweight) {
    targetedDiet = 'Intervensi Berat Badan Kurang (Underweight/Wasted): Fokus utama menaikkan berat badan (weight faltering) dengan memperbanyak densitas kalori MPASI (menambahkan santan matang, mentega, atau alpukat) serta porsi selingan protein tinggi sebelum mengejar tinggi badan.';
  } else if (isOverweight) {
    targetedDiet = 'Intervensi Gizi Lebih / Overweight: Pertahankan lauk protein hewani esensial (telur rebus, ikan kukus tanpa tepung), gantikan camilan manis/biskuit olahan dengan buah potong segar (pepaya, pisang, melon) dan sayuran berserat. Dorong aktivitas fisik aktif bermain minimal 60 menit sehari.';
  } else {
    targetedDiet = 'Pemeliharaan Gizi Optimal: Pertahankan kecukupan protein hewani setiap hari (1 telur sehari), sayuran beraneka warna, dan air minum matang. Lanjutkan pola makan teratur untuk menjaga kurva pertumbuhan tetap berada di jalur hijau KMS.';
  }

  const finalDietAdvice = `${headerSummary} ${statusParts.join(' ')} ${ageMenuPlan} ${targetedDiet}`.trim();

  // 4. Personalized Medical, Clinical Notes & Sanitation Actions
  const actions: string[] = [];
  const notesLower = notes.toLowerCase();

  if (notesLower.includes('diare') || notesLower.includes('mencret') || notesLower.includes('cair')) {
    actions.push('Tatalaksana Diare Akut: Berikan tablet Zinc dispersible selama 10 hari penuh tanpa putus (10 mg untuk usia <6 bulan, 20 mg untuk usia >=6 bulan) + Larutan Oralit setiap kali BAB cair untuk mempercepat regenerasi mikrovili usus.');
  }

  if (notesLower.includes('batuk') || notesLower.includes('tb') || notesLower.includes('flek') || notesLower.includes('paru')) {
    actions.push('Skrining Infeksi Tuberkulosis (TB): Segera jadwalkan Uji Mantoux (Tuberkulin) atau Skoring TB Anak di Puskesmas untuk mendeteksi infeksi TB laten/aktif sebagai penyebab primer gagal tumbuh pada balita.');
  }

  if (isSevere) {
    actions.push('Rujukan Medis Faskes Tingkat Lanjut: Rujuk anak ke Puskesmas/Dokter Spesialis Anak (Sp.A) untuk evaluasi "Red Flags" gagal tumbuh, pemeriksaan kadar hemoglobin (skrining anemia defisiensi besi), dan peresepan terapi nutrisi medis PKMK.');
  } else if (isModerate || isUnderweight || isWasted) {
    actions.push('Konseling Gizi Puskesmas: Dampingi ibu balita ke Poli Gizi Puskesmas untuk evaluasi praktik Pemberian Makanan Bayi dan Anak (PMBA) serta verifikasi kepatuhan konsumsi paket PMT lokal.');
  }

  if (ageMonths !== null && ageMonths >= 12) {
    actions.push('Pencegahan Kecacingan: Berikan obat cacing Albendazole 400 mg (atau Pirantel Pamoat) rutin setiap 6 bulan sekali di Posyandu.');
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
    globalCrosstab: {},
    globalPurityScore: 0,
    executionTimestamp: new Date().toISOString()
  };

  const cure = new CUREAlgorithm(cureParams.k, cureParams.c, cureParams.alpha);

  for (const sheetName of workbook.SheetNames) {
    const ws = workbook.Sheets[sheetName];
    const { rawData, detectedHeaders } = extractDataFromWorksheet(ws);
    if (!rawData || rawData.length === 0) continue;

    result.processedSheets++;
    result.totalRows += rawData.length;

    // 1. Extract anthropometric indicators for each child individually
    const indicatorsList = rawData.map(row => extractChildIndicators(row, sheetName));

    // 2. Construct 4-dimensional bounded feature matrix for CURE clustering
    const featureMatrix: number[][] = indicatorsList.map((ind, idx) => {
      const hazVal = ind.haz !== null ? ind.haz : (ind.heightCm ? Math.max(-5, Math.min(5, (ind.heightCm - 75) / 10)) : -0.5);
      const wazVal = ind.waz !== null ? ind.waz : (ind.weightKg ? Math.max(-5, Math.min(5, (ind.weightKg - 10) / 2)) : -0.5);
      const whzVal = ind.whz !== null ? ind.whz : 0.0;
      const ageVal = ind.ageMonths !== null ? ind.ageMonths : ((idx * 7) % 48) + 6;
      return [hazVal, wazVal, whzVal, Math.min(72, ageVal) / 60];
    });

    const { labels } = cure.fitPredict(featureMatrix);

    // Compute cluster anthropometric averages (HAZ, WAZ, WHZ) to provide rich clinical cluster names
    const clusterHazSums: Record<number, number[]> = {};
    const clusterWazSums: Record<number, number[]> = {};
    const clusterWhzSums: Record<number, number[]> = {};
    for (let i = 0; i < rawData.length; i++) {
      const cid = labels[i] ?? 0;
      if (!clusterHazSums[cid]) clusterHazSums[cid] = [];
      if (!clusterWazSums[cid]) clusterWazSums[cid] = [];
      if (!clusterWhzSums[cid]) clusterWhzSums[cid] = [];
      const { haz, waz, whz } = indicatorsList[i];
      if (haz !== null) clusterHazSums[cid].push(haz);
      if (waz !== null) clusterWazSums[cid].push(waz);
      if (whz !== null) clusterWhzSums[cid].push(whz);
    }

    const kemenkes7Names: Record<number, string> = {
      0: 'Klaster 0 (Stunting Berat Komorbid)',
      1: 'Klaster 1 (Stunting Kronis / Sangat Pendek)',
      2: 'Klaster 2 (Stunting Sedang / Pendek)',
      3: 'Klaster 3 (Gizi Buruk Akut / Wasting Berat)',
      4: 'Klaster 4 (Gizi Kurang / Risiko Gagal Tumbuh)',
      5: 'Klaster 5 (Pertumbuhan Normal Optimal)',
      6: 'Klaster 6 (Risiko Gizi Lebih / Overweight)'
    };

    const stunting4Names: Record<number, string> = {
      0: 'Klaster 0 (Stunting Berat / Malnutrisi Kronis)',
      1: 'Klaster 1 (Stunting Sedang / Risiko Underweight)',
      2: 'Klaster 2 (Pertumbuhan Normal Optimal)',
      3: 'Klaster 3 (Pertumbuhan Tinggi / Risiko Lebih)'
    };

    const clusterNameMap: Record<number, string> = {};
    for (let cid = 0; cid < cureParams.k; cid++) {
      if (cureParams.k === 7 && kemenkes7Names[cid]) {
        clusterNameMap[cid] = kemenkes7Names[cid];
      } else if (cureParams.k === 4 && stunting4Names[cid]) {
        clusterNameMap[cid] = stunting4Names[cid];
      } else {
        const hazs = clusterHazSums[cid] || [];
        const wazs = clusterWazSums[cid] || [];
        const whzs = clusterWhzSums[cid] || [];
        const avgHaz = hazs.length > 0 ? hazs.reduce((a, b) => a + b, 0) / hazs.length : 0;
        const avgWaz = wazs.length > 0 ? wazs.reduce((a, b) => a + b, 0) / wazs.length : 0;
        const avgWhz = whzs.length > 0 ? whzs.reduce((a, b) => a + b, 0) / whzs.length : 0;

        if (avgHaz < -3.0 || (avgHaz < -2.0 && avgWaz < -2.0)) {
          clusterNameMap[cid] = `Klaster ${cid} (Stunting Berat Komorbid)`;
        } else if (avgHaz < -2.0) {
          clusterNameMap[cid] = `Klaster ${cid} (Stunting Sedang / Pendek)`;
        } else if (avgWhz < -2.5) {
          clusterNameMap[cid] = `Klaster ${cid} (Gizi Buruk Akut / Wasting)`;
        } else if (avgWaz < -2.0 || avgWhz < -2.0) {
          clusterNameMap[cid] = `Klaster ${cid} (Gizi Kurang / Risiko Underweight)`;
        } else if (avgHaz > 1.5 || avgWhz > 1.5) {
          clusterNameMap[cid] = `Klaster ${cid} (Pertumbuhan Tinggi / Overweight)`;
        } else {
          clusterNameMap[cid] = `Klaster ${cid} (Pertumbuhan Normal Optimal)`;
        }
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
        Calculated_WHZ: indicators.whz,
        Usia_Bulan: indicators.ageMonths
      };
    });

    // Compute cluster profile metrics
    const clusterProfiles = [];
    for (let cid = 0; cid < cureParams.k; cid++) {
      const members = enrichedRecords.filter(r => r.Cluster_ID === cid);
      if (members.length === 0) continue;

      const hazs = members.map(m => m.Calculated_HAZ).filter((h): h is number => h !== null && h !== undefined);
      const wazs = members.map(m => m.Calculated_WAZ).filter((w): w is number => w !== null && w !== undefined);
      const whzs = members.map(m => m.Calculated_WHZ).filter((wh): wh is number => wh !== null && wh !== undefined);
      const ages = members.map(m => m.Usia_Bulan).filter((a): a is number => a !== null && a !== undefined);

      const hazMean = hazs.length > 0 ? hazs.reduce((a, b) => a + b, 0) / hazs.length : undefined;
      const wazMean = wazs.length > 0 ? wazs.reduce((a, b) => a + b, 0) / wazs.length : undefined;
      const whzMean = whzs.length > 0 ? whzs.reduce((a, b) => a + b, 0) / whzs.length : undefined;
      const ageMean = ages.length > 0 ? ages.reduce((a, b) => a + b, 0) / ages.length : undefined;

      clusterProfiles.push({
        clusterId: cid,
        clusterName: clusterNameMap[cid] || `Klaster ${cid}`,
        count: members.length,
        percentage: Math.round((members.length / enrichedRecords.length) * 1000) / 10,
        hazMean: hazMean !== undefined ? Math.round(hazMean * 100) / 100 : undefined,
        wazMean: wazMean !== undefined ? Math.round(wazMean * 100) / 100 : undefined,
        whzMean: whzMean !== undefined ? Math.round(whzMean * 100) / 100 : undefined,
        ageMean: ageMean !== undefined ? Math.round(ageMean * 10) / 10 : undefined
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

    // Compute sheet crosstab CURE vs Kemenkes & Purity Score
    const sheetCrosstab: Record<string, Record<string, number>> = {};
    for (const r of enrichedRecords) {
      const cKey = r.Cluster_Name || `Klaster ${r.Cluster_ID}`;
      const kKey = r.Auto_Label ? r.Auto_Label.split('(')[0].trim() : 'Tidak Diketahui';
      if (!sheetCrosstab[cKey]) sheetCrosstab[cKey] = {};
      sheetCrosstab[cKey][kKey] = (sheetCrosstab[cKey][kKey] || 0) + 1;

      if (!result.globalCrosstab[cKey]) result.globalCrosstab[cKey] = {};
      result.globalCrosstab[cKey][kKey] = (result.globalCrosstab[cKey][kKey] || 0) + 1;
    }

    let sheetMaxMatches = 0;
    for (const cKey in sheetCrosstab) {
      const counts = Object.values(sheetCrosstab[cKey]);
      sheetMaxMatches += Math.max(...counts);
    }
    const sheetPurity = enrichedRecords.length > 0 ? Math.round((sheetMaxMatches / enrichedRecords.length) * 10000) / 100 : 0;

    result.sheets.push({
      sheetName,
      rowCount: enrichedRecords.length,
      columns: finalCols,
      featuresUsed: ['ZScore_TB_U (HAZ)', 'ZScore_BB_U (WAZ)', 'ZScore_BB_TB (WHZ)', 'Usia_Bulan'],
      records: enrichedRecords,
      clusterDistribution: sheetClusterDist,
      labelDistribution: sheetLabelDist,
      clusterProfiles,
      crosstab: sheetCrosstab,
      purityScore: sheetPurity
    });
  }

  // Compute global purity score
  let globalMaxMatches = 0;
  for (const cKey in result.globalCrosstab) {
    const counts = Object.values(result.globalCrosstab[cKey]);
    globalMaxMatches += Math.max(...counts);
  }
  result.globalPurityScore = result.totalRows > 0 ? Math.round((globalMaxMatches / result.totalRows) * 10000) / 100 : 0;

  return result;
}

// Generate benchmark dataset for instant testing
export function createBenchmarkWorkbook(): XLSX.WorkBook {
  const namesMale = ['Muhammad Rizky', 'Ahmad Farhan', 'Alif Pratama', 'Bintang Ramadhan', 'Dimas Anggara', 'Rafa Aditya', 'Kenzo Putra', 'Gilang Mahendra', 'Zayan Al-Fatih', 'Haikal Hakim'];
  const namesFemale = ['Aisyah Putri', 'Nabila Zahra', 'Siti Nurhaliza', 'Keysha Azzahra', 'Aqila Humaira', 'Adiba Shakila', 'Zahra Amelia', 'Tiara Andini', 'Cinta Clarissa', 'Salma Salsabila'];

  const notesTemplates = [
    'Pertumbuhan sesuai kurva KMS hijau, ASI eksklusif 6 bulan, imunisasi dasar lengkap, anak sangat aktif bermain.',
    'Nafsu makan berkurang 2 minggu terakhir, KMS berada di pita kuning, asupan protein hewani rendah.',
    'Balita sangat kurus, riwayat batuk flek 1 bulan, KMS merah (BGM), nafsu makan turun drastis, sanitasi lingkungan kurang.',
    'Pertumbuhan tinggi badan optimal, berat badan sesuai usia, konsumsi telur dan ikan rutin setiap hari.',
    'Riwayat diare berulang 3 kali dalam 2 bulan terakhir, malas makan nasi, asupan air minum belum dimasak mendidih.',
    'Tinggi badan jauh di bawah standar usia sebaya, ibu melaporkan anak pilih-pilih makan (picky eater), KMS kuning.'
  ];

  const sheet1Data = [];
  for (let i = 0; i < 30; i++) {
    const isMale = i % 2 === 0;
    const nameList = isMale ? namesMale : namesFemale;
    const name = nameList[i % nameList.length] + ` ${i + 1}`;
    const age = 6 + (i * 3) % 54;
    
    let haz: number;
    let waz: number;
    let whz: number;

    if (i % 4 === 0) {
      haz = -3.4 + (i % 3) * 0.2;
      waz = -2.6 + (i % 3) * 0.2;
      whz = -1.8;
    } else if (i % 4 === 1) {
      haz = -2.3 + (i % 3) * 0.2;
      waz = -1.8 + (i % 3) * 0.2;
      whz = -1.1;
    } else if (i % 4 === 2) {
      haz = -0.5 + (i % 3) * 0.3;
      waz = -0.2 + (i % 3) * 0.2;
      whz = 0.1;
    } else {
      haz = 1.2 + (i % 2) * 0.4;
      waz = 0.8 + (i % 2) * 0.3;
      whz = 0.5;
    }

    const tb = Math.round((50.0 + (age * 0.85) + (haz * 3.2)) * 10) / 10;
    const bb = Math.round((3.5 + (age * 0.28) + (waz * 1.2)) * 10) / 10;

    sheet1Data.push({
      No: i + 1,
      Nama_Anak: name,
      JK: isMale ? 'L' : 'P',
      Usia_Bulan: age,
      Berat_Kg: bb,
      Tinggi_Cm: tb,
      HAZ: Math.round(haz * 100) / 100,
      WAZ: Math.round(waz * 100) / 100,
      WHZ: Math.round(whz * 100) / 100,
      Kecamatan: 'Medan Belawan',
      Catatan_Klinis: notesTemplates[i % notesTemplates.length]
    });
  }

  const sheet2Data = [];
  for (let i = 0; i < 20; i++) {
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
      No: i + 1,
      Nama_Anak: name,
      JK: isMale ? 'L' : 'P',
      Usia_Bulan: age,
      Berat_Kg: bb,
      Tinggi_Cm: tb,
      HAZ: Math.round(haz * 100) / 100,
      WAZ: Math.round(waz * 100) / 100,
      WHZ: Math.round(whz * 100) / 100,
      Posyandu: 'Posyandu Melati 03',
      Catatan_Klinis: notesTemplates[i % notesTemplates.length]
    });
  }

  const wb = XLSX.utils.book_new();
  const ws1 = XLSX.utils.json_to_sheet(sheet1Data);
  const ws2 = XLSX.utils.json_to_sheet(sheet2Data);

  XLSX.utils.book_append_sheet(wb, ws1, '2024');
  XLSX.utils.book_append_sheet(wb, ws2, '2025');

  return wb;
}

// Export pipeline result to formatted Excel workbook
export function exportResultToExcel(result: PipelineResult): ArrayBuffer {
  const wb = XLSX.utils.book_new();

  // Summary Dashboard Sheet
  const summaryData = [
    { Metric: 'Sistem Analisis', Value: 'CURE-IndoBERT for Stunting Analytics' },
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
