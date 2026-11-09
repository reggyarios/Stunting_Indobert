import * as XLSX from 'xlsx';
import { cureFit, curePredict, seededSample, silhouetteScore } from './cure';

/**
 * Dua mode analisis dengan keluaran yang sama bentuknya:
 * - server: Flask /api/analyze (CURE cure.py + IndoBERT-RCNN bila checkpoint tersedia)
 * - client: port CURE (lib/cure.ts) + kategori Permenkes berbasis ambang z-score (BUKAN IndoBERT)
 * Validasi baris/kolom di sini meniru app.py agar kedua mode menolak data yang sama.
 */

export type ChildRecord = Record<string, any>;

export interface AnalysisParams {
  k: number;
  c: number;
  alpha: number;
  sampleSize: number;
  referenceDate: string; // YYYY-MM-DD
}

export interface ExcludedRow { row: number; reasons: string[] }

export interface ClusterProfile {
  clusterId: number;
  name: string;                 // diturunkan dari isi klaster
  count: number;
  percentage: number;
  hazMean: number | null;
  wazMean: number | null;
  whzMean: number | null;
  ageMean: number | null;
  majorityCategory: string;
  majorityShare: number;        // %
  semanticLabel?: string | null;       // konsensus IndoBERT-RCNN (server)
  semanticConfidence?: number | null;
}

export interface SheetResult {
  sheetName: string;
  inputRows: number;
  rowCount: number;
  columns: string[];            // seluruh kolom record (urutan ekspor), tanpa kolom identitas
  records: ChildRecord[];
  excluded: ExcludedRow[];
  ageSource: string;
  clusterProfiles: ClusterProfile[];
  labelDistribution: Record<string, number>;
  crosstab: Record<string, Record<string, number>>;
  purityScore: number | null;   // terhadap kategori acuan Permenkes
  silhouette: number | null;
}

export interface PipelineResult {
  mode: 'server' | 'client';
  indobertActive: boolean;
  originalFilename: string;
  totalRows: number;
  processedSheets: number;
  cureParams: AnalysisParams;
  sheets: SheetResult[];
  skippedSheets: { sheetName: string; reason: string }[];
  labelField: string;
  labelTitle: string;
  executionTimestamp: string;
}

export class AnalysisError extends Error {
  details: string[];
  constructor(message: string, details: string[] = []) {
    super(message);
    this.details = details;
  }
}

// Sama dengan indobert_rcnn.DEFAULT_LABEL_DESCRIPTIONS
export const PERMENKES_LABELS: Record<string, string> = {
  STUNTING_BERAT_KOMORBID: 'Stunting Berat dengan Komorbiditas Antropometri',
  STUNTING_KRONIS: 'Stunting Kronis / Sangat Pendek (HAZ < -3 SD)',
  STUNTING_SEDANG: 'Stunting Sedang / Pendek (-3 SD <= HAZ < -2 SD)',
  GIZI_BURUK_AKUT: 'Gizi Buruk Akut / Wasting Berat (WHZ < -3 SD)',
  GIZI_KURANG_RISIKO: 'Gizi Kurang / Risiko Gagal Tumbuh',
  NORMAL_OPTIMAL: 'Pertumbuhan Normal / Gizi Optimal',
  RISIKO_GIZI_LEBIH: 'Risiko Gizi Lebih / Overweight',
};
export const RULE_LABEL_FIELD = 'Kategori_Permenkes_7Kelas';
export const RULE_LABEL_TITLE = 'Kategori Permenkes (aturan ambang)';
export const MODEL_LABEL_FIELD = 'IndoBERT_Record_Prediksi';

const PRIORITY: Record<string, string> = {
  STUNTING_BERAT_KOMORBID: 'Kritis (Intervensi Medis & Rujukan)',
  STUNTING_KRONIS: 'Tinggi (Intervensi Gizi Spesifik)',
  STUNTING_SEDANG: 'Sedang (Pemberian Makanan Tambahan PMT)',
  GIZI_BURUK_AKUT: 'Kritis (Tatalaksana Gizi Buruk)',
  GIZI_KURANG_RISIKO: 'Sedang (Pemantauan Kejar Tumbuh)',
  RISIKO_GIZI_LEBIH: 'Rendah (Pola Makan Seimbang)',
  NORMAL_OPTIMAL: 'Rendah (Pemantauan Rutin Posyandu)',
};

// Batas yang sama dengan app.py
const WEIGHT_RANGE: [number, number] = [1.5, 35];
const HEIGHT_RANGE: [number, number] = [38, 130];
const AGE_MAX_MONTHS = 60;
const DAYS_PER_MONTH = 30.4375;
const IDENTIFIER_TOKENS = new Set(['nama', 'name', 'nik', 'kk', 'alamat', 'address', 'ibu', 'ayah', 'ortu', 'wali', 'telp', 'telepon', 'hp']);
const HEADER_KEYWORDS = new Set(['nama', 'balita', 'anak', 'bayi', 'nik', 'no', 'nomor', 'usia', 'umur', 'bulan', 'bln', 'jk', 'gender', 'bb', 'tb',
  'pb', 'haz', 'waz', 'whz', 'puskesmas', 'posyandu', 'desa', 'berat', 'tinggi', 'panjang', 'catatan', 'keterangan']);
export const MAX_SAMPLE_SIZE = 3000;

export function normalizeColName(k: unknown): string {
  return String(k ?? '').trim().toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
}

type Role = 'haz' | 'waz' | 'whz' | 'age' | 'dob' | 'measured' | 'height' | 'weight' | 'gender' | 'identifier' | 'notes' | 'status_tbu' | 'status_bbu' | 'status_bbtb';

const roleCache = new Map<string, Role | null>();

/** Cermin app.column_role (+ kolom catatan & status teks untuk teks saran). */
export function columnRole(col: unknown): Role | null {
  const key = String(col ?? '');
  if (!roleCache.has(key)) roleCache.set(key, computeRole(key));
  return roleCache.get(key)!;
}

function computeRole(col: string): Role | null {
  const k = normalizeColName(col);
  const toks = new Set(k.split('_'));
  const has = (...xs: string[]) => xs.some(x => toks.has(x));
  if ([...toks].some(t => IDENTIFIER_TOKENS.has(t))) return 'identifier';
  if (has('status', 'kategori')) return null;
  const isZ = has('zs', 'zscore', 'z');
  if (toks.has('haz') || (isZ && (k.includes('tb_u') || k.includes('pb_u')))) return 'haz';
  if (toks.has('waz') || (isZ && k.includes('bb_u'))) return 'waz';
  if (toks.has('whz') || (isZ && (k.includes('bb_tb') || k.includes('bb_pb')))) return 'whz';
  if (k === 'tb_u' || k === 'pb_u') return 'status_tbu';
  if (k === 'bb_u') return 'status_bbu';
  if (k === 'bb_tb' || k === 'bb_pb') return 'status_bbtb';
  if (['tb_u', 'pb_u', 'bb_u', 'bb_tb', 'bb_pb'].some(p => k.includes(p))) return null;
  if (has('tgl', 'tanggal')) {
    if (has('lahir', 'lhr')) return 'dob';
    if (has('ukur', 'timbang', 'pengukuran', 'kunjungan')) return 'measured';
    return null;
  }
  if (has('usia', 'umur')) return 'age';
  if (has('tinggi', 'panjang', 'tb', 'pb')) return 'height';
  if (has('berat', 'bb')) return 'weight';
  if (has('jk', 'gender', 'sex') || k.includes('jenis_kelamin')) return 'gender';
  if (['catatan', 'keterangan', 'keluhan', 'diagnosa', 'riwayat', 'notes'].some(t => k.includes(t))) return 'notes';
  return null;
}

/** Cermin pipeline.parse_indonesian_numeric: koma desimal, minus unicode, satuan. */
export function parseNumericValue(val: unknown): number | null {
  if (val === null || val === undefined || val === '') return null;
  if (typeof val === 'number') return Number.isFinite(val) ? val : null;
  let str = String(val).trim();
  if (['-', '--', 'null', 'nan', 'None', ''].includes(str)) return null;
  str = str.replace(/[−–—―﹣－]/g, '-').replace(/,/g, '.');
  const match = str.match(/[-+]?\s*\d*\.?\d+/);
  if (!match) return null;
  const num = parseFloat(match[0].replace(/\s+/g, ''));
  return Number.isFinite(num) ? num : null;
}

function parseGender(v: unknown): 'L' | 'P' | null {
  if (v === null || v === undefined) return null;
  const s = String(v).trim().toUpperCase();
  if (s.includes('L') || s.includes('PRIA') || s.includes('LAKI')) return 'L';
  if (s.includes('P') || s.includes('WANITA') || s.includes('PEREMPUAN')) return 'P';
  return null;
}

/** Hari sejak 1970-01-01 (UTC) dari serial Excel, Date, 'YYYY-MM-DD[...]' atau 'DD/MM/YYYY'. */
function toUtcDay(v: unknown): number | null {
  const ymd = (y: number, m: number, d: number) => {
    const t = Date.UTC(y, m - 1, d);
    const dt = new Date(t);
    return dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d ? t / 86400000 : null;
  };
  if (typeof v === 'number') return v >= 1 && v < 2958466 ? Math.floor(v) - 25569 : null;
  if (v instanceof Date) return isNaN(v.getTime()) ? null : ymd(v.getFullYear(), v.getMonth() + 1, v.getDate());
  if (typeof v !== 'string') return null;
  const s = v.trim();
  let m = s.match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
  if (m) return ymd(+m[1], +m[2], +m[3]);
  m = s.match(/^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$/);
  if (m) return ymd(+m[3], +m[2], +m[1]);
  return null;
}

function dayToIso(day: number): string {
  return new Date(day * 86400000).toISOString().slice(0, 10);
}

/** Cermin app.row_issues: alasan baris dikeluarkan (pesan sama). */
export function rowIssues(haz: number | null, waz: number | null, whz: number | null, weight: number | null,
  height: number | null, age: number | null, gender: string | null): string[] {
  const missing = ([['JK', gender], ['Berat', weight], ['Tinggi', height], ['ZS TB/U', haz], ['ZS BB/U', waz],
    ['ZS BB/TB', whz], ['Usia/Tgl Lahir', age]] as const).filter(([, v]) => v === null).map(([l]) => l);
  const g = (x: number) => String(x);
  const issues = missing.length ? [`Data tidak lengkap/tidak terbaca: ${missing.join(', ')}`] : [];
  if (haz !== null && Math.abs(haz) > 6) issues.push(`ZS TB/U ${g(haz)} di luar batas WHO (|z| > 6)`);
  if (waz !== null && !(waz >= -6 && waz <= 5)) issues.push(`ZS BB/U ${g(waz)} di luar batas WHO (-6 s.d. +5)`);
  if (whz !== null && Math.abs(whz) > 5) issues.push(`ZS BB/TB ${g(whz)} di luar batas WHO (|z| > 5)`);
  if (weight !== null && !(weight >= WEIGHT_RANGE[0] && weight <= WEIGHT_RANGE[1]))
    issues.push(`Berat ${g(weight)} kg di luar rentang wajar (${WEIGHT_RANGE[0]}-${WEIGHT_RANGE[1]} kg)`);
  if (height !== null && !(height >= HEIGHT_RANGE[0] && height <= HEIGHT_RANGE[1]))
    issues.push(`Tinggi ${g(height)} cm di luar rentang wajar (${HEIGHT_RANGE[0]}-${HEIGHT_RANGE[1]} cm)`);
  if (age !== null && !(age >= 0 && age <= AGE_MAX_MONTHS))
    issues.push(`Usia ${g(age)} bulan di luar rentang balita 0-60 bulan (periksa tanggal acuan)`);
  return issues;
}

/** Cermin pipeline.permenkes_reference_label (ambang Permenkes 2/2020, prioritas TB/U). */
export function permenkesKey(haz: number, waz: number, whz: number): string {
  if (haz < -3) return waz < -2 || whz < -2 ? 'STUNTING_BERAT_KOMORBID' : 'STUNTING_KRONIS';
  if (haz < -2) return 'STUNTING_SEDANG';
  if (whz < -3) return 'GIZI_BURUK_AKUT';
  if (whz < -2 || waz < -2) return 'GIZI_KURANG_RISIKO';
  if (whz > 1) return 'RISIKO_GIZI_LEBIH';
  return 'NORMAL_OPTIMAL';
}

const shortLabel = (s: string) => String(s ?? '-').split(' (')[0];
const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/** Nilai numerik indikator dari record berdasarkan peran kolom (dipakai modal detail). */
export function roleValue(record: ChildRecord, role: Role): number | null {
  for (const [k, v] of Object.entries(record)) if (columnRole(k) === role) return parseNumericValue(v);
  return null;
}

// ---------------------------------------------------------------------------
// Saran gizi umum (mode tanpa server) — panduan umum berbasis kategori, bukan diagnosis
// ---------------------------------------------------------------------------
function recommendations(key: string, age: number, haz: number, waz: number, whz: number, height: number, weight: number,
  gender: 'L' | 'P' | null, notes: string, st: { tbu: string; bbu: string; bbtb: string }) {
  const isSevere = key === 'STUNTING_BERAT_KOMORBID' || key === 'STUNTING_KRONIS' || key === 'GIZI_BURUK_AKUT';
  const isModerate = key === 'STUNTING_SEDANG' || key === 'GIZI_KURANG_RISIKO';
  const isOverweight = key === 'RISIKO_GIZI_LEBIH';
  const isUnderweight = waz < -2;
  const isWasted = whz < -2;
  const m = Math.floor(age);
  const sex = gender === 'P' ? ', Perempuan' : gender === 'L' ? ', Laki-laki' : '';
  const fileStatus = (s: string) => (s ? `, status di berkas: ${s}` : '');
  // Median WHO tidak ditampilkan: tabel LMS WHO 2006 tidak tersedia secara andal di aplikasi ini
  const status = `Tinggi ${height} cm (Z-Score TB/U ${haz.toFixed(2)} SD${fileStatus(st.tbu)}). ` +
    `Berat ${weight} kg (Z-Score BB/U ${waz.toFixed(2)} SD${fileStatus(st.bbu)}). ` +
    `Z-Score BB/TB ${whz.toFixed(2)} SD${fileStatus(st.bbtb)}.`;

  let ageMenuPlan: string;
  if (m < 6) {
    ageMenuPlan = 'Fase Usia < 6 Bulan: ASI eksklusif on-demand minimal 8-12 kali per 24 jam tanpa tambahan air putih atau makanan padat. Ibu menyusui dianjurkan mengonsumsi protein hewani harian dan tablet tambah darah agar kualitas gizi ASI optimal.';
  } else if (m <= 8) {
    ageMenuPlan = 'Fase MPASI Awal (6-8 Bulan): Tekstur saring/lumat kental. 2-3 kali makan utama (mulai 2-3 sdm bertahap hingga 1/2 mangkuk 125 ml) + 1-2 selingan. Sertakan protein hewani setiap hari (telur, hati ayam, atau ikan) dan sedikit lemak tambahan sebagai sumber energi.';
  } else if (m <= 11) {
    ageMenuPlan = 'Fase MPASI Lanjutan (9-11 Bulan): Tekstur cincang halus atau finger food lunak. 3-4 kali makan utama (1/2 mangkuk 125-150 ml) + 1-2 selingan dan ASI. Sertakan telur dan ikan/daging setiap hari.';
  } else if (m <= 23) {
    ageMenuPlan = 'Fase Makanan Keluarga (12-23 Bulan): Makanan keluarga dipotong kecil/lunak, 3-4 kali makan utama (3/4 mangkuk 200 ml) + 2 selingan padat protein + ASI hingga 2 tahun.';
  } else if (m <= 35) {
    ageMenuPlan = 'Fase Batita (2-3 Tahun): Menu Isi Piringku 3 kali sehari + 2 selingan buah atau kudapan protein hewani. Batasi jajanan manis kemasan dan minuman berpemanis.';
  } else {
    ageMenuPlan = 'Fase Prasekolah (3-5 Tahun): Menu gizi seimbang Isi Piringku 3 kali makan utama + 2 selingan sehat, protein hewani setiap hari, sayur dan buah beragam, air minum matang.';
  }

  let targetedDiet: string;
  if (isSevere) {
    targetedDiet = 'Kategori berat: perbanyak protein hewani harian dan densitas energi makanan; konsultasikan ke tenaga kesehatan/Puskesmas untuk tata laksana gizi sesuai pedoman.';
  } else if (isModerate) {
    targetedDiet = 'Kategori sedang: Pemberian Makanan Tambahan (PMT) berbahan pangan lokal sesuai program Puskesmas, dengan sumber protein hewani dan zat besi setiap hari.';
  } else if (isWasted || isUnderweight) {
    targetedDiet = 'Berat badan kurang: utamakan kenaikan berat badan dengan menambah densitas kalori makanan dan selingan tinggi protein.';
  } else if (isOverweight) {
    targetedDiet = 'Risiko gizi lebih: pertahankan protein hewani, ganti camilan manis dengan buah segar dan sayur, dorong aktivitas fisik bermain minimal 60 menit sehari.';
  } else {
    targetedDiet = 'Pemeliharaan gizi: pertahankan protein hewani harian, sayur beragam, dan air minum matang agar pertumbuhan tetap di jalur normal.';
  }

  const actions: string[] = [];
  const nl = notes.toLowerCase();
  if (/diare|mencret|\bcair\b/.test(nl)) actions.push('Riwayat diare pada catatan: tata laksana diare dengan oralit dan zinc sesuai anjuran tenaga kesehatan.');
  if (/batuk|\btb\b|tuberkulosis|flek|paru/.test(nl)) actions.push('Keluhan batuk/riwayat TB pada catatan: rujuk ke Puskesmas untuk skrining TB anak.');
  if (isSevere) actions.push('Rujuk ke Puskesmas/dokter anak untuk evaluasi penyebab gagal tumbuh dan pemeriksaan anemia.');
  else if (isModerate || isUnderweight || isWasted) actions.push('Konseling gizi di Puskesmas: evaluasi praktik Pemberian Makan Bayi dan Anak (PMBA) dan kepatuhan PMT.');
  if (m >= 12) actions.push('Obat cacing sesuai jadwal program (setiap 6 bulan) di Posyandu.');
  actions.push('Sanitasi & CTPS: cuci tangan pakai sabun pada waktu kritis dan pastikan air minum dimasak hingga mendidih.');

  const schedule = isSevere
    ? 'Pemantauan ketat sesuai arahan Puskesmas/faskes rujukan.'
    : isModerate || isUnderweight || isWasted
      ? 'Pemantauan berkala BB dan PB/TB di Posyandu atau Pos Gizi Puskesmas.'
      : 'Pemantauan rutin bulanan BB dan PB/TB pada hari buka Posyandu.';

  return {
    Prioritas_Intervensi: PRIORITY[key],
    Saran_Asupan_Nutrisi: `[Balita ${m} bln${sex}]: ${status} ${ageMenuPlan} ${targetedDiet}`,
    Tindakan_Medis_Sanitasi: actions.join(' '),
    Jadwal_Monitoring: schedule,
  };
}

// ---------------------------------------------------------------------------
// Ringkasan per lembar (dipakai kedua mode)
// ---------------------------------------------------------------------------
function summarizeSheet(base: Omit<SheetResult, 'clusterProfiles' | 'labelDistribution' | 'crosstab' | 'purityScore'>,
  labelField: string, semantic: { cluster_id: number; cluster_label: string; confidence: number | null }[] | null): SheetResult {
  const { records } = base;
  const byCluster = new Map<number, ChildRecord[]>();
  for (const r of records) {
    const cid = Number(r.CURE_Cluster_ID);
    if (cid >= 0) byCluster.set(cid, [...(byCluster.get(cid) ?? []), r]);
  }
  const clustered = [...byCluster.values()].reduce((a, m) => a + m.length, 0);
  const clusterProfiles: ClusterProfile[] = [...byCluster.entries()].sort((a, b) => a[0] - b[0]).map(([cid, members]) => {
    const counts: Record<string, number> = {};
    for (const r of members) counts[r[RULE_LABEL_FIELD]] = (counts[r[RULE_LABEL_FIELD]] ?? 0) + 1;
    const [majority, majorityCount] = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    const nums = (role: Role) => members.map(r => roleValue(r, role)).filter((v): v is number => v !== null);
    const hazMean = mean(nums('haz'));
    const share = round((majorityCount / members.length) * 100, 1);
    const sem = semantic?.find(s => s.cluster_id === cid);
    return {
      clusterId: cid,
      name: `Klaster ${cid}: mayoritas ${shortLabel(majority)} (${share}%)${hazMean !== null ? `, rerata HAZ ${hazMean.toFixed(2)}` : ''}`,
      count: members.length,
      percentage: round((members.length / records.length) * 100, 1),
      hazMean: hazMean === null ? null : round(hazMean, 2),
      wazMean: round(mean(nums('waz')) ?? NaN, 2),
      whzMean: round(mean(nums('whz')) ?? NaN, 2),
      ageMean: round(mean(members.map(r => Number(r.Usia_Bulan_Dihitung)).filter(Number.isFinite)) ?? NaN, 1),
      majorityCategory: majority,
      majorityShare: share,
      semanticLabel: sem?.cluster_label ?? null,
      semanticConfidence: sem?.confidence ?? null,
    };
  });

  const nameOf = new Map(clusterProfiles.map(p => [p.clusterId, p.name]));
  const crosstab: Record<string, Record<string, number>> = {};
  const labelDistribution: Record<string, number> = {};
  let maxSum = 0;
  for (const r of records) {
    const lbl = String(r[labelField]);
    labelDistribution[lbl] = (labelDistribution[lbl] ?? 0) + 1;
    const cid = Number(r.CURE_Cluster_ID);
    r.Profil_Klaster = nameOf.get(cid) ?? 'Tidak terklaster';
    if (cid < 0) continue;
    const cKey = r.Profil_Klaster;
    const kKey = shortLabel(r[RULE_LABEL_FIELD]);
    crosstab[cKey] ??= {};
    crosstab[cKey][kKey] = (crosstab[cKey][kKey] ?? 0) + 1;
  }
  for (const row of Object.values(crosstab)) maxSum += Math.max(...Object.values(row));
  const columns = base.columns.includes('Profil_Klaster') ? base.columns : [...base.columns, 'Profil_Klaster'];
  return {
    ...base, columns, clusterProfiles, labelDistribution, crosstab,
    purityScore: clustered ? round((maxSum / clustered) * 100, 2) : null,
  };
}

// ---------------------------------------------------------------------------
// Mode tanpa server
// ---------------------------------------------------------------------------
const XLSX_MAGIC = [0x50, 0x4b, 0x03, 0x04];
const XLS_MAGIC = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];
export function hasExcelSignature(head: Uint8Array): boolean {
  const starts = (sig: number[]) => sig.every((b, i) => head[i] === b);
  return starts(XLSX_MAGIC) || starts(XLS_MAGIC);
}

function detectHeaderRow(rows: unknown[][]): number {
  let best = 0, bestScore = -1;
  for (let r = 0; r < Math.min(15, rows.length); r++) {
    let score = 0;
    for (const cell of rows[r] ?? []) {
      if (cell === null || cell === undefined || cell === '') continue;
      if (normalizeColName(cell).split('_').some(t => HEADER_KEYWORDS.has(t))) score++;
    }
    if (score > bestScore) { bestScore = score; best = r; }
  }
  return best;
}

export function analyzeWorkbook(buffer: ArrayBuffer, filename: string, params: AnalysisParams,
  onProgress: (pct: number, message: string) => void = () => {}): PipelineResult {
  if (!hasExcelSignature(new Uint8Array(buffer.slice(0, 8)))) {
    throw new AnalysisError('Isi berkas bukan workbook Excel (.xlsx/.xls). Berkas mungkin rusak, atau berupa CSV/PDF/berkas lain yang diganti ekstensinya.');
  }
  let wb: XLSX.WorkBook;
  try {
    wb = XLSX.read(buffer, { type: 'array' });
  } catch {
    throw new AnalysisError('Workbook Excel tidak dapat dibaca: berkas rusak, terenkripsi, atau bukan berkas Excel.');
  }
  const refDay = toUtcDay(params.referenceDate);
  if (refDay === null) throw new AnalysisError('Tanggal acuan harus berformat YYYY-MM-DD.');

  const sheets: SheetResult[] = [];
  const skipped: { sheetName: string; reason: string }[] = [];
  let anyData = false;
  let validSheets = 0;
  const nSheets = wb.SheetNames.length;

  wb.SheetNames.forEach((sheetName, si) => {
    const pBase = 5 + (90 * si) / nSheets;
    const pSpan = 90 / nSheets;
    onProgress(pBase, `Validasi lembar ${sheetName}...`);
    const ws = wb.Sheets[sheetName];
    const rows = XLSX.utils.sheet_to_json<unknown[]>(ws, { header: 1, defval: null, raw: true, blankrows: true });
    const origin = ws['!ref'] ? XLSX.utils.decode_range(ws['!ref']).s.r : 0;
    if (rows.length < 2) { skipped.push({ sheetName, reason: 'Lembar kosong' }); return; }

    const h = detectHeaderRow(rows);
    const seen: Record<string, number> = {};
    const headers = (rows[h] ?? []).map((cell, i) => {
      const name = cell === null || cell === undefined || String(cell).trim() === '' ? `Kolom_${i + 1}` : String(cell).trim();
      seen[name] = (seen[name] ?? 0) + 1;
      return seen[name] === 1 ? name : `${name}_${seen[name]}`;
    });
    const dataRows = rows.slice(h + 1)
      .map((row, i) => ({ row: row ?? [], excelRow: origin + h + 2 + i }))
      .filter(({ row }) => row.some(v => v !== null && v !== undefined && String(v).trim() !== ''));
    if (!dataRows.length) { skipped.push({ sheetName, reason: 'Tidak ada baris data balita' }); return; }
    anyData = true;

    const roleIdx: Partial<Record<Role, number>> = {};
    headers.forEach((hd, i) => {
      const role = columnRole(hd);
      if (role && role !== 'identifier' && roleIdx[role] === undefined) roleIdx[role] = i;
    });
    const required: [Role, string][] = [['gender', 'JK'], ['weight', 'Berat'], ['height', 'Tinggi'], ['haz', 'ZS TB/U'], ['waz', 'ZS BB/U'], ['whz', 'ZS BB/TB']];
    const missing = required.filter(([r]) => roleIdx[r] === undefined).map(([, l]) => l);
    if (roleIdx.age === undefined && roleIdx.dob === undefined) missing.push('Usia (bulan) atau Tgl Lahir');
    if (missing.length) { skipped.push({ sheetName, reason: `Kolom wajib tidak ditemukan: ${missing.join(', ')}` }); return; }
    validSheets++;

    const useMeasured = roleIdx.age === undefined && roleIdx.measured !== undefined;
    const ageSource = roleIdx.age !== undefined ? 'kolom usia'
      : useMeasured ? 'Tgl Lahir terhadap kolom tanggal ukur' : `Tgl Lahir terhadap tanggal acuan ${params.referenceDate}`;
    const keepIdx = headers.map((_, i) => i).filter(i => columnRole(headers[i]) !== 'identifier');
    const dateIdx = new Set([roleIdx.dob, roleIdx.measured].filter((i): i is number => i !== undefined));
    const cell = (row: unknown[], role: Role) => (roleIdx[role] === undefined ? null : row[roleIdx[role]!]);
    const num = (row: unknown[], role: Role) => parseNumericValue(cell(row, role));

    const excluded: ExcludedRow[] = [];
    const records: ChildRecord[] = [];
    const features: number[][] = [];
    for (const { row, excelRow } of dataRows) {
      let age: number | null;
      if (roleIdx.age !== undefined) {
        age = num(row, 'age');
      } else {
        const dob = toUtcDay(cell(row, 'dob'));
        const ref = (useMeasured ? toUtcDay(cell(row, 'measured')) : null) ?? refDay;
        age = dob === null ? null : round((ref - dob) / DAYS_PER_MONTH, 2);
      }
      const haz = num(row, 'haz'), waz = num(row, 'waz'), whz = num(row, 'whz');
      const weight = num(row, 'weight'), height = num(row, 'height');
      const gender = parseGender(cell(row, 'gender'));
      const issues = rowIssues(haz, waz, whz, weight, height, age, gender);
      if (issues.length) { excluded.push({ row: excelRow, reasons: issues }); continue; }

      const rec: ChildRecord = { Baris_Excel: excelRow };
      for (const i of keepIdx) {
        const v = row[i] ?? null;
        const day = dateIdx.has(i) ? toUtcDay(v) : null;
        rec[headers[i]] = day !== null ? dayToIso(day) : v;
      }
      const key = permenkesKey(haz!, waz!, whz!);
      rec.Usia_Bulan_Dihitung = age;
      rec[RULE_LABEL_FIELD] = PERMENKES_LABELS[key];
      rec.CURE_Cluster_ID = -1;
      Object.assign(rec, recommendations(key, age!, haz!, waz!, whz!, height!, weight!, gender,
        String(cell(row, 'notes') ?? ''), {
          tbu: String(cell(row, 'status_tbu') ?? ''), bbu: String(cell(row, 'status_bbu') ?? ''), bbtb: String(cell(row, 'status_bbtb') ?? ''),
        }));
      records.push(rec);
      features.push([haz!, waz!, whz!, age! / 60]);
    }

    let silhouette: number | null = null;
    if (records.length >= 2) {
      const k = Math.min(params.k, records.length);
      const sampled = records.length > params.sampleSize;
      const sampleIdx = sampled ? seededSample(records.length, params.sampleSize, 42) : null;
      const fitX = sampleIdx ? sampleIdx.map(i => features[i]) : features;
      onProgress(pBase + pSpan * 0.1, `CURE lembar ${sheetName}: ${fitX.length} titik sampel, k=${k}...`);
      const model = cureFit(fitX, k, params.c, params.alpha, (done, total) =>
        onProgress(pBase + pSpan * (0.1 + 0.8 * (done / total)), `CURE lembar ${sheetName}: penggabungan ${done}/${total}`));
      const labels = sampled ? curePredict(model, features) : model.labels;
      labels.forEach((l, i) => { records[i].CURE_Cluster_ID = l; });
      silhouette = silhouetteScore(model.scaled, model.labels);
    }

    if (!records.length) {
      skipped.push({ sheetName, reason: `Semua ${excluded.length} baris dikeluarkan saat validasi (contoh baris ${excluded[0].row}: ${excluded[0].reasons.join('; ')})` });
      return;
    }
    const columns = ['Baris_Excel', ...keepIdx.map(i => headers[i]), 'Usia_Bulan_Dihitung', RULE_LABEL_FIELD, 'CURE_Cluster_ID',
      'Prioritas_Intervensi', 'Saran_Asupan_Nutrisi', 'Tindakan_Medis_Sanitasi', 'Jadwal_Monitoring'];
    sheets.push(summarizeSheet({ sheetName, inputRows: dataRows.length, rowCount: records.length, columns, records, excluded, ageSource, silhouette },
      RULE_LABEL_FIELD, null));
  });

  if (!anyData) throw new AnalysisError('Tidak ada data balita pada berkas ini.', skipped.map(s => `Lembar ${s.sheetName}: ${s.reason}`));
  const details = skipped.map(s => `Lembar ${s.sheetName}: ${s.reason}`);
  if (!validSheets) {
    throw new AnalysisError('Tidak ada lembar dengan kolom wajib (JK, Berat, Tinggi, ZS TB/U, ZS BB/U, ZS BB/TB, serta Usia atau Tgl Lahir). Data tidak diberi label.', details);
  }
  if (!sheets.length) throw new AnalysisError('Tidak ada data balita yang lolos validasi (semua baris tidak lengkap atau tidak wajar).', details);
  onProgress(100, 'Selesai');
  return {
    mode: 'client', indobertActive: false, originalFilename: filename,
    totalRows: sheets.reduce((a, s) => a + s.rowCount, 0), processedSheets: sheets.length,
    cureParams: params, sheets, skippedSheets: skipped,
    labelField: RULE_LABEL_FIELD, labelTitle: RULE_LABEL_TITLE, executionTimestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Mode server: adaptor respons /api/analyze
// ---------------------------------------------------------------------------
interface ServerSheet {
  sheet_name: string; input_rows: number; excluded_rows: ExcludedRow[]; age_source: string;
  columns: string[]; rows: unknown[][];
}
interface ServerSummarySheet {
  sheet_name: string; silhouette_score: number | null;
  cluster_semantic_labels: { cluster_id: number; cluster_label: string; confidence: number | null }[];
}
export interface ServerResponse {
  status: string; message?: string; indobert_active: boolean; reference_date: string;
  summary: { sheets: ServerSummarySheet[] };
  sheets: ServerSheet[];
  skipped_sheets: { sheet_name: string; reason: string }[];
}

export function fromServerResponse(body: ServerResponse, filename: string, params: AnalysisParams): PipelineResult {
  const labelField = body.indobert_active ? MODEL_LABEL_FIELD : RULE_LABEL_FIELD;
  const sheets = body.sheets.map(s => {
    const dateCols = new Set(s.columns.filter(c => ['dob', 'measured'].includes(columnRole(c) ?? '')));
    const records = s.rows.map(row => {
      const rec: ChildRecord = {};
      s.columns.forEach((c, i) => {
        const v = row[i];
        rec[c] = dateCols.has(c) && typeof v === 'string' ? v.slice(0, 10) : v;
      });
      return rec;
    });
    const summ = body.summary.sheets.find(x => x.sheet_name === s.sheet_name);
    return summarizeSheet({
      sheetName: s.sheet_name, inputRows: s.input_rows, rowCount: records.length, columns: s.columns, records,
      excluded: s.excluded_rows, ageSource: s.age_source, silhouette: summ?.silhouette_score ?? null,
    }, labelField, body.indobert_active ? summ?.cluster_semantic_labels ?? null : null);
  });
  return {
    mode: 'server', indobertActive: body.indobert_active, originalFilename: filename,
    totalRows: sheets.reduce((a, s) => a + s.rowCount, 0), processedSheets: sheets.length,
    cureParams: { ...params, referenceDate: body.reference_date }, sheets,
    skippedSheets: body.skipped_sheets.map(s => ({ sheetName: s.sheet_name, reason: s.reason })),
    labelField, labelTitle: body.indobert_active ? 'Prediksi IndoBERT-RCNN (per anak)' : RULE_LABEL_TITLE,
    executionTimestamp: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Interpretasi dihitung dari angka hasil (bukan teks statis)
// ---------------------------------------------------------------------------
export function interpretSheet(sheet: SheetResult, result: PipelineResult): string[] {
  const out: string[] = [];
  const p = result.cureParams;
  const clustered = sheet.clusterProfiles.reduce((a, c) => a + c.count, 0);
  if (sheet.purityScore !== null) {
    const level = sheet.purityScore >= 80 ? 'tinggi' : sheet.purityScore >= 60 ? 'sedang' : 'rendah';
    out.push(`Purity klaster CURE terhadap ${RULE_LABEL_TITLE} = ${sheet.purityScore.toFixed(2)}%: ` +
      `${Math.round((sheet.purityScore / 100) * clustered).toLocaleString('id-ID')} dari ${clustered.toLocaleString('id-ID')} balita ` +
      `berada pada kategori mayoritas klasternya (keselarasan ${level}). Kategori acuan berasal dari ambang z-score, ` +
      `sehingga purity bukan akurasi model.`);
  }
  const sorted = [...sheet.clusterProfiles].sort((a, b) => b.count - a.count);
  if (sorted.length) {
    const big = sorted[0], small = sorted[sorted.length - 1];
    out.push(`CURE (k=${p.k}, c=${p.c}, α=${p.alpha.toFixed(2)}) membentuk ${sorted.length} klaster; terbesar Klaster ${big.clusterId} ` +
      `(${big.count.toLocaleString('id-ID')} anak, ${big.percentage}%), terkecil Klaster ${small.clusterId} ` +
      `(${small.count.toLocaleString('id-ID')} anak, ${small.percentage}%).`);
  }
  for (const c of sheet.clusterProfiles) {
    out.push(`Klaster ${c.clusterId}: ${c.majorityShare}% anggota berkategori ${shortLabel(c.majorityCategory)}; rerata HAZ ${c.hazMean ?? '-'}, ` +
      `WAZ ${c.wazMean ?? '-'}, WHZ ${c.whzMean ?? '-'}, usia ${c.ageMean ?? '-'} bulan.` +
      (c.semanticLabel ? ` Label konsensus IndoBERT-RCNN: ${c.semanticLabel}${c.semanticConfidence !== null && c.semanticConfidence !== undefined ? ` (${c.semanticConfidence})` : ''}.` : ''));
  }
  if (sheet.silhouette !== null) {
    const s = sheet.silhouette;
    const q = s > 0.5 ? 'struktur klaster kuat' : s > 0.25 ? 'struktur klaster lemah-sedang' : 'struktur klaster lemah/tumpang tindih';
    out.push(`Silhouette (ruang fitur terstandar, sampel CURE) = ${s.toFixed(4)}: ${q}.`);
  }
  if (sheet.excluded.length) {
    out.push(`${sheet.excluded.length.toLocaleString('id-ID')} dari ${sheet.inputRows.toLocaleString('id-ID')} baris dikeluarkan sebelum analisis ` +
      `(data tidak lengkap, WHO flag, atau ukuran tidak wajar). Usia dihitung dari ${sheet.ageSource}.`);
  }
  if (result.mode === 'client' && sheet.inputRows > p.sampleSize) {
    out.push(`Mode browser: tahap hierarkis CURE memakai sampel acak ${p.sampleSize} titik (seed 42); hasil dapat berbeda dari server yang memakai RNG numpy.`);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Ekspor Excel (terkompresi, tanggal sebagai tanggal Excel)
// ---------------------------------------------------------------------------
export function exportResultToExcel(result: PipelineResult): ArrayBuffer {
  const wb = XLSX.utils.book_new();
  const p = result.cureParams;
  const summary: [string, string | number][] = [
    ['Sistem Analisis', 'CURE-IndoBERT for Stunting Analytics'],
    ['Mode', result.mode === 'server'
      ? (result.indobertActive ? 'Server: CURE + model IndoBERT-RCNN' : 'Server: CURE (checkpoint IndoBERT belum tersedia)')
      : 'Browser tanpa model: CURE + kategori Permenkes (aturan ambang)'],
    ['Label per anak', result.labelTitle],
    ['Waktu Analisis', result.executionTimestamp],
    ['Berkas Sumber', result.originalFilename],
    ['Total Data Balita Dianalisis', result.totalRows],
    ['Jumlah Klaster (k)', p.k],
    ['Titik Perwakilan (c)', p.c],
    ['Faktor Penyusutan (alpha)', p.alpha],
    ['Ukuran Sampel CURE', p.sampleSize],
    ['Tanggal Acuan Usia', p.referenceDate],
    ...result.sheets.flatMap(s => [
      [`[${s.sheetName}] Baris dianalisis / dikeluarkan`, `${s.rowCount} / ${s.excluded.length}`],
      [`[${s.sheetName}] Purity vs ${RULE_LABEL_TITLE} (%)`, s.purityScore ?? '-'],
      [`[${s.sheetName}] Silhouette`, s.silhouette ?? '-'],
    ] as [string, string | number][]),
  ];
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['Metrik', 'Nilai'], ...summary]), 'Ringkasan');

  const used = new Set(['Ringkasan']);
  for (const sheet of result.sheets) {
    const dateCols = sheet.columns.filter(c => ['dob', 'measured'].includes(columnRole(c) ?? ''));
    const rows = sheet.records.map(r => {
      const o: ChildRecord = {};
      for (const c of sheet.columns) {
        const v = r[c];
        const m = dateCols.includes(c) && typeof v === 'string' ? v.match(/^(\d{4})-(\d{2})-(\d{2})$/) : null;
        o[c] = m ? new Date(+m[1], +m[2] - 1, +m[3]) : v;
      }
      return o;
    });
    let name = sheet.sheetName.slice(0, 31);
    while (used.has(name)) name = `${name.slice(0, 28)}_${used.size}`;
    used.add(name);
    XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(rows, { header: sheet.columns, cellDates: true, dateNF: 'yyyy-mm-dd' }), name);
  }
  const excludedRows = result.sheets.flatMap(s => s.excluded.map(e => ({ Lembar: s.sheetName, Baris_Excel: e.row, Alasan: e.reasons.join('; ') })));
  if (excludedRows.length) XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(excludedRows), 'Baris_Dikeluarkan');

  return XLSX.write(wb, { bookType: 'xlsx', type: 'array', compression: true }) as ArrayBuffer;
}

// Data contoh sintetis untuk uji perangkat lunak (bukan data klinis riil)
export function createBenchmarkWorkbook(): XLSX.WorkBook {
  const notesTemplates = [
    'Pertumbuhan sesuai kurva KMS hijau, ASI eksklusif 6 bulan, imunisasi dasar lengkap.',
    'Nafsu makan berkurang 2 minggu terakhir, KMS berada di pita kuning.',
    'Balita kurus, riwayat batuk 1 bulan, nafsu makan turun, sanitasi lingkungan kurang.',
    'Pertumbuhan tinggi badan optimal, konsumsi telur dan ikan rutin setiap hari.',
    'Riwayat diare berulang 3 kali dalam 2 bulan terakhir.',
    'Tinggi badan di bawah standar usia sebaya, anak pilih-pilih makan.',
  ];
  const make = (n: number, sheet: 1 | 2) => Array.from({ length: n }, (_, i) => {
    const age = sheet === 1 ? 6 + (i * 3) % 54 : 8 + (i * 2) % 48;
    const g = sheet === 1 ? i % 4 : [0, 1, 2][i % 3];
    const [haz, waz, whz] = g === 0 ? [-3.4 + (i % 3) * 0.2, -2.6 + (i % 3) * 0.2, -1.8]
      : g === 1 ? [-2.3 + (i % 3) * 0.2, -1.8 + (i % 3) * 0.2, -1.1]
        : g === 2 ? [-0.5 + (i % 3) * 0.3, -0.2 + (i % 3) * 0.2, 0.1]
          : [1.2 + (i % 2) * 0.4, 0.8 + (i % 2) * 0.3, 0.5];
    return {
      No: i + 1,
      Nama_Anak: `Anak Contoh ${sheet}-${i + 1}`,
      JK: (i + sheet) % 2 === 1 ? 'L' : 'P',
      Usia_Bulan: age,
      Berat_Kg: round(3.5 + age * 0.28 + waz * 1.2, 1),
      Tinggi_Cm: round(50 + age * 0.85 + haz * 3.2, 1),
      HAZ: round(haz, 2),
      WAZ: round(waz, 2),
      WHZ: round(whz, 2),
      Posyandu: sheet === 1 ? 'Posyandu Contoh A' : 'Posyandu Contoh B',
      Catatan_Klinis: notesTemplates[i % notesTemplates.length],
    };
  });
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(make(30, 1)), '2024');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.json_to_sheet(make(20, 2)), '2025');
  return wb;
}
