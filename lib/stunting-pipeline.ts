import * as XLSX from 'xlsx';

export interface ChildRecord {
  [key: string]: any;
  Cluster_ID?: number;
  Auto_Label?: string;
  Label_Confidence?: number;
  Clinical_Priority?: string;
  Saran_Asupan_Nutrisi?: string;
  Tindakan_Medis_Sanitasi?: string;
  Jadwal_Monitoring?: string;
  Calculated_HAZ?: number | null;
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

const CLINICAL_LEXICON = {
  severe: [
    "sangat pendek", "severely stunted", "bgm", "bawah garis merah", "kms merah",
    "gizi buruk", "edema", "maramus", "kwashiorkor", "sangat kurus", "lila merah",
    "diare kronis", "tb/u < -3", "zscore < -3", "tb paru", "batuk kronis", "anoreksia"
  ],
  moderate: [
    "pendek", "stunted", "kurus", "gizi kurang", "kms kuning", "tidak naik bb",
    "2t", "dua kali tidak naik", "nafsu makan turun", "mpasi tidak adekuat",
    "protein hewani rendah", "lila kuning", "tb/u -2 sd -3"
  ],
  faltering: [
    "asi tidak eksklusif", "formula dini", "sanitasi buruk", "air tidak bersih",
    "riwayat bblr", "berat lahir rendah", "prematur", "sering demam", "ispa"
  ],
  normal: [
    "normal", "sehat", "gizi baik", "aktif", "pertumbuhan sesuai", "kms hijau",
    "nafsu makan baik", "asi eksklusif", "imunisasi lengkap", "lila hijau"
  ]
};

// CURE (Clustering Using Representatives)
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
    if (N < this.k) {
      return {
        labels: features.map((_, i) => i % this.k),
        representatives: []
      };
    }

    const D = features[0].length;

    // 1. Feature Standardization (Z-score normalization)
    const means = new Array(D).fill(0);
    const stds = new Array(D).fill(0);

    for (let j = 0; j < D; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += features[i][j];
      means[j] = sum / N;

      let varSum = 0;
      for (let i = 0; i < N; i++) {
        const diff = features[i][j] - means[j];
        varSum += diff * diff;
      }
      stds[j] = Math.sqrt(varSum / N) || 1.0;
    }

    const scaled = features.map(row =>
      row.map((val, j) => (val - means[j]) / stds[j])
    );

    // 2. Initialize clusters (Grid / sub-sampling)
    const initK = Math.min(N, Math.max(this.k * 3, Math.min(N, 30)));
    // Simple fast k-means++ style initialization for initial subclusters
    const initialCentroids: number[][] = [];
    initialCentroids.push([...scaled[0]]);
    while (initialCentroids.length < initK) {
      let maxDist = -1;
      let bestIdx = 0;
      for (let i = 0; i < N; i++) {
        let minDist = Infinity;
        for (const c of initialCentroids) {
          const d = this.euclidean(scaled[i], c);
          if (d < minDist) minDist = d;
        }
        if (minDist > maxDist) {
          maxDist = minDist;
          bestIdx = i;
        }
      }
      initialCentroids.push([...scaled[bestIdx]]);
    }

    // Assign to closest initial centroid
    const clusters: { members: number[]; centroid: number[]; reps: number[][] }[] = [];
    for (let k = 0; k < initK; k++) {
      clusters.push({ members: [], centroid: initialCentroids[k], reps: [] });
    }

    for (let i = 0; i < N; i++) {
      let minDist = Infinity;
      let bestC = 0;
      for (let k = 0; k < initK; k++) {
        const d = this.euclidean(scaled[i], clusters[k].centroid);
        if (d < minDist) {
          minDist = d;
          bestC = k;
        }
      }
      clusters[bestC].members.push(i);
    }

    // Filter empty clusters and compute initial representatives
    let activeClusters = clusters.filter(c => c.members.length > 0).map(c => {
      const pts = c.members.map(idx => scaled[idx]);
      const centroid = this.computeMean(pts, D);
      const reps = this.selectRepresentatives(pts, centroid, this.c);
      const shrunkReps = reps.map(r =>
        r.map((val, dim) => val + this.alpha * (centroid[dim] - val))
      );
      return {
        members: c.members,
        centroid,
        reps: shrunkReps
      };
    });

    // 3. Hierarchical Agglomerative Merging of CURE Representative Sets
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

      // Merge clusters mergeI and mergeJ
      const c1 = activeClusters[mergeI];
      const c2 = activeClusters[mergeJ];
      const mergedMembers = [...c1.members, ...c2.members];
      const mergedPts = mergedMembers.map(idx => scaled[idx]);
      const newCentroid = this.computeMean(mergedPts, D);
      const newReps = this.selectRepresentatives(mergedPts, newCentroid, this.c);
      const newShrunkReps = newReps.map(r =>
        r.map((val, dim) => val + this.alpha * (newCentroid[dim] - val))
      );

      const newCluster = {
        members: mergedMembers,
        centroid: newCentroid,
        reps: newShrunkReps
      };

      activeClusters.splice(Math.max(mergeI, mergeJ), 1);
      activeClusters.splice(Math.min(mergeI, mergeJ), 1);
      activeClusters.push(newCluster);
    }

    // 4. Assign labels
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

// IndoBERT contextual inference simulation / semantic rules
export function predictIndoBERTLabel(row: Record<string, any>): {
  autoLabel: string;
  confidence: number;
  priority: string;
  saranNutrisi: string;
  tindakanMedis: string;
  jadwalMonitoring: string;
  extractedHaz: number | null;
  extractedWaz: number | null;
  extractedWhz: number | null;
} {
  const haz = extractFloat(row, ["haz", "zscore_tb_u", "z_score_tb", "zscore_tbu", "tb_u", "zscore"]);
  const waz = extractFloat(row, ["waz", "zscore_bb_u", "z_score_bb", "zscore_bbu", "bb_u"]);
  const whz = extractFloat(row, ["whz", "zscore_bb_tb", "z_score_bb_tb", "bb_tb"]);

  let notes = "";
  for (const [k, v] of Object.entries(row)) {
    const kLow = k.toLowerCase();
    if (["catatan", "keterangan", "diagnosa", "keluhan", "notes", "riwayat", "status_gizi"].some(t => kLow.includes(t))) {
      notes += ` ${v}`;
    }
  }
  const cleanNotes = notes.toLowerCase();

  const hasSevere = CLINICAL_LEXICON.severe.some(m => cleanNotes.includes(m));
  const hasMod = CLINICAL_LEXICON.moderate.some(m => cleanNotes.includes(m));
  const hasFaltering = CLINICAL_LEXICON.faltering.some(m => cleanNotes.includes(m));
  const hasNormal = CLINICAL_LEXICON.normal.some(m => cleanNotes.includes(m));

  let label = TAXONOMY.NORMAL_OPTIMAL;
  let confidence = 0.86;
  let priority = "Rendah (Pencegahan Primer)";

  if (haz !== null) {
    if (haz < -3.0) {
      if (hasSevere || (whz !== null && whz < -2.0)) {
        label = TAXONOMY.STUNTING_BERAT_KOMORBID;
        confidence = 0.96;
        priority = "Kritis (Intervensi Medis & Rujukan RS)";
      } else {
        label = TAXONOMY.STUNTING_KRONIS;
        confidence = 0.93;
        priority = "Tinggi (Intervensi Gizi Spesifik)";
      }
    } else if (haz >= -3.0 && haz < -2.0) {
      if (hasSevere || (waz !== null && waz < -2.5)) {
        label = TAXONOMY.STUNTING_BERAT_KOMORBID;
        confidence = 0.89;
        priority = "Tinggi (Intervensi Gizi Spesifik)";
      } else {
        label = TAXONOMY.STUNTING_SEDANG;
        confidence = 0.91;
        priority = "Sedang (Pemberian Makanan Tambahan PMT)";
      }
    } else if (haz >= -2.0 && haz <= 2.0) {
      if (whz !== null && whz < -3.0) {
        label = TAXONOMY.GIZI_BURUK_AKUT;
        confidence = 0.94;
        priority = "Kritis (Terapi Gizi Buruk TFC)";
      } else if (whz !== null && whz >= -3.0 && whz < -2.0) {
        label = TAXONOMY.GIZI_KURANG_RISIKO;
        confidence = 0.88;
        priority = "Sedang (Pemantauan Pertumbuhan Intensif)";
      } else if (hasSevere || hasFaltering) {
        label = TAXONOMY.GIZI_KURANG_RISIKO;
        confidence = 0.82;
        priority = "Sedang (Edukasi PMBA & Pemantauan)";
      } else {
        label = TAXONOMY.NORMAL_OPTIMAL;
        confidence = 0.95;
        priority = "Rendah (Pemantauan Rutin Posyandu)";
      }
    } else {
      label = TAXONOMY.RISIKO_GIZI_LEBIH;
      confidence = 0.87;
      priority = "Rendah (Pola Makan Seimbang)";
    }
  } else {
    // Textual dominance
    if (hasSevere) {
      label = TAXONOMY.STUNTING_BERAT_KOMORBID;
      confidence = 0.88;
      priority = "Tinggi (Pemeriksaan Lanjutan Diperlukan)";
    } else if (hasMod) {
      label = TAXONOMY.STUNTING_SEDANG;
      confidence = 0.83;
      priority = "Sedang (Pemeriksaan Antropometri Validasi)";
    } else if (hasNormal) {
      label = TAXONOMY.NORMAL_OPTIMAL;
      confidence = 0.85;
      priority = "Rendah (Pemantauan Rutin)";
    } else {
      label = TAXONOMY.GIZI_KURANG_RISIKO;
      confidence = 0.70;
      priority = "Sedang (Evaluasi Ulang)";
    }
  }

  // Extract age in months for age-specific MPASI/feeding guidance
  const age = extractFloat(row, ["usia", "umur", "bulan", "age", "usia_bulan", "umur_bulan"]);
  const recommendations = generateInterventionRecommendations(label, age, haz, waz, whz, cleanNotes);

  return {
    autoLabel: label,
    confidence,
    priority,
    saranNutrisi: recommendations.saranNutrisi,
    tindakanMedis: recommendations.tindakanMedis,
    jadwalMonitoring: recommendations.jadwalMonitoring,
    extractedHaz: haz,
    extractedWaz: waz,
    extractedWhz: whz
  };
}

export function generateInterventionRecommendations(
  autoLabel: string,
  ageMonths: number | null,
  haz: number | null,
  waz: number | null,
  whz: number | null,
  notes: string
): {
  saranNutrisi: string;
  tindakanMedis: string;
  jadwalMonitoring: string;
} {
  const isSevere = autoLabel.includes("Stunting Berat") || autoLabel.includes("Gizi Buruk") || (haz !== null && haz < -3.0);
  const isModerate = autoLabel.includes("Stunting Ringan") || autoLabel.includes("Gizi Kurang") || (haz !== null && haz < -2.0 && haz >= -3.0);
  const isOverweight = autoLabel.includes("Risiko Gizi Lebih");

  // 1. Dietary & Nutrition Recommendations
  let dietAdvice = "";

  // Age-specific base guidance
  let ageGuidance = "";
  if (ageMonths !== null) {
    if (ageMonths < 6) {
      ageGuidance = "Usia < 6 Bulan: Wajib ASI Eksklusif on-demand minimal 8-12 kali sehari semalam tanpa tambahan air putih atau makanan padat apapun.";
    } else if (ageMonths >= 6 && ageMonths <= 8) {
      ageGuidance = "Usia 6-8 Bulan: MPASI tekstur lumat kental (saring/puree), 2-3 kali makan utama (2-3 sdm bertahap hingga 1/2 mangkuk 250ml) + 1-2 selingan dan tetap ASI.";
    } else if (ageMonths >= 9 && ageMonths <= 11) {
      ageGuidance = "Usia 9-11 Bulan: MPASI tekstur cincang halus/kasar atau finger foods lunak, 3-4 kali makan utama (1/2 mangkuk 250ml) + 1-2 selingan padat gizi dan ASI.";
    } else if (ageMonths >= 12 && ageMonths <= 23) {
      ageGuidance = "Usia 12-23 Bulan: Makanan keluarga menu seimbang, 3-4 kali makan utama (3/4 mangkuk 250ml) + 1-2 selingan kaya protein, ASI diteruskan hingga 2 tahun.";
    } else {
      ageGuidance = "Usia 2-5 Tahun: Menu keluarga Isi Piringku 3 kali sehari porsi penuh + 2 selingan buah/kudapan berprotein hewani, hindari jajanan manis sebelum makan.";
    }
  }

  if (isSevere) {
    dietAdvice = "Prioritas Protein Hewani Ganda: Wajib berikan minimal 1 butir telur ayam setiap hari + ikan laut lokal (ikan kembung/lele/tongkol kaya omega-3 DHA dan asam amino esensial untuk lempeng pertumbuhan tulang). Tambahkan hati ayam/sapi 2x seminggu untuk mencukupi cadangan zat besi (mencegah anemia penghambat hormon pertumbuhan). Berikan lemak tambahan (1-2 sdt minyak kelapa, santan matang, atau margarin) pada setiap porsi makan untuk mendongkrak densitas kalori (energy booster). Berikan suplemen tabur gizi mikro (Taburia) dan suplemen zinc sesuai anjuran faskes.";
  } else if (isModerate) {
    dietAdvice = "Perkaya Kualitas MPASI/Pola Makan: Pastikan lauk berprotein hewani hadir di setiap waktu makan (telur, ikan kembung, daging ayam cincang, atau udang). Kombinasikan dengan protein nabati (tempe/tahu) dan sayuran hijau/labu kuning kaya vitamin A. Berikan Makanan Tambahan Pemulihan (PMT) berbasis pangan lokal selama 90 hari. Batasi ketat konsumsi biskuit manis, wafer, dan teh manis agar nafsu makan anak tidak tertekan.";
  } else if (isOverweight) {
    dietAdvice = "Pengaturan Porsi Seimbang: Perbanyak serat dari sayur dan buah potong segar, kurangi gorengan, tepung, dan minuman berkadar gula tinggi. Jangan batasi nutrisi esensial protein hewani (tetap berikan telur dan ikan tanpa digoreng tepung). Dorong aktivitas fisik aktif bermain minimal 60 menit per hari.";
  } else {
    dietAdvice = "Pertahankan Pola Gizi Seimbang Isi Piringku: 1/3 karbohidrat (nasi/umbi), 1/3 lauk pauk sumber protein hewani dan nabati berkualitas, 1/3 sayuran dan buah beraneka ragam. Pastikan anak sarapan bergizi setiap pagi dan mengonsumsi air minum matang yang higienis.";
  }

  const finalDietAdvice = ageGuidance ? `${ageGuidance} ${dietAdvice}` : dietAdvice;

  // 2. Medical, Hygiene & Sanitation Actions
  const actions: string[] = [];

  if (notes.includes("diare")) {
    actions.push("Tatalaksana Diare: Berikan tablet Zinc dispersible selama 10 hari berturut-turut tanpa putus (10 mg untuk usia <6 bulan, 20 mg untuk usia >=6 bulan) + Oralit setiap BAB cair untuk memperbaiki vili usus dan mencegah malabsorpsi gizi.");
  }

  if (notes.includes("batuk") || notes.includes("tb") || notes.includes("flek")) {
    actions.push("Investigasi Infeksi TB: Segera lakukan uji Mantoux / skoring TB anak di Puskesmas bila ada riwayat batuk kronis atau kontak erat serumah dengan orang dewasa penderita TB aktif.");
  }

  if (notes.includes("bgm") || notes.includes("2t") || notes.includes("tidak naik") || isSevere) {
    actions.push("Rujukan Medis Faskes: Rujuk ke dokter spesialis anak / Puskesmas untuk evaluasi red flags gagal tumbuh (growth faltering), skrining infeksi saluran kemih (ISK) asimtomatik, dan pertimbangan terapi Pangan Olahan untuk Keperluan Medis Khusus (PKMK).");
  }

  // General hygiene & sanitation
  actions.push("Sanitasi & Pencegahan: Berikan obat cacing (Albendazole) rutin tiap 6 bulan untuk anak usia >1 tahun. Rebus air minum keluarga hingga mendidih 100°C. Biasakan Cuci Tangan Pakai Sabun (CTPS) pada 5 waktu kritis (sebelum makan, sebelum mengolah MPASI, setelah BAB/membersihkan popok, dan sebelum menyusui).");

  const finalMedicalAction = actions.join(" ");

  // 3. Monitoring Schedule
  let schedule = "";
  if (isSevere) {
    schedule = "Pemantauan Mingguan: Penimbangan berat badan dan evaluasi asupan makanan setiap 1 minggu sekali di Puskesmas atau Faskes rujukan.";
  } else if (isModerate) {
    schedule = "Pemantauan Dua Mingguan: Penimbangan berat badan dan pengukuran panjang/tinggi badan setiap 2 minggu sekali di Posyandu/Puskesmas.";
  } else {
    schedule = "Pemantauan Rutin Bulanan: Penimbangan berat badan dan pengukuran panjang/tinggi badan setiap 1 bulan sekali pada Hari Buka Posyandu.";
  }

  return {
    saranNutrisi: finalDietAdvice,
    tindakanMedis: finalMedicalAction,
    jadwalMonitoring: schedule
  };
}

export function extractFloat(record: Record<string, any>, candidates: string[]): number | null {
  for (const [k, v] of Object.entries(record)) {
    const kClean = k.toLowerCase().replace(/[\s-]/g, "_");
    if (candidates.some(c => kClean === c || kClean.startsWith(c))) {
      if (v !== null && v !== undefined && v !== "") {
        const valStr = String(v).replace(",", ".").trim();
        const match = valStr.match(/[-+]?\d*\.?\d+/);
        if (match) {
          const num = parseFloat(match[0]);
          if (!isNaN(num)) return num;
        }
      }
    }
  }
  return null;
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
    const rawData = XLSX.utils.sheet_to_json<Record<string, any>>(ws, { defval: "" });
    if (!rawData || rawData.length === 0) continue;

    result.processedSheets++;
    result.totalRows += rawData.length;

    // Numerical feature extraction for CURE
    const allCols = Object.keys(rawData[0]);
    const numCols = allCols.filter(col => {
      const colLow = col.toLowerCase();
      const isAnthro = ["usia", "umur", "tinggi", "tb", "berat", "bb", "haz", "waz", "whz", "zscore", "lila", "imt"].some(t => colLow.includes(t));
      const isId = ["id", "nik", "no", "kode"].some(t => colLow.includes(t));
      return isAnthro && !isId;
    });

    const activeNumCols = numCols.length > 0 ? numCols : allCols.filter(c => {
      const val = rawData[0][c];
      return typeof val === "number" || (!isNaN(parseFloat(val)) && !c.toLowerCase().includes("id"));
    });

    const featureMatrix: number[][] = rawData.map(row => {
      if (activeNumCols.length === 0) return [1.0];
      return activeNumCols.map(col => {
        const parsed = extractFloat({ [col]: row[col] }, [col.toLowerCase()]);
        return parsed !== null ? parsed : 0;
      });
    });

    const { labels } = cure.fitPredict(featureMatrix);

    const sheetClusterDist: Record<string, number> = {};
    const sheetLabelDist: Record<string, number> = {};

    const enrichedRecords: ChildRecord[] = rawData.map((row, idx) => {
      const cid = labels[idx] ?? 0;
      const inference = predictIndoBERTLabel(row);

      const clusterKey = `Klaster ${cid}`;
      sheetClusterDist[clusterKey] = (sheetClusterDist[clusterKey] || 0) + 1;
      sheetLabelDist[inference.autoLabel] = (sheetLabelDist[inference.autoLabel] || 0) + 1;

      result.globalClusterDistribution[clusterKey] = (result.globalClusterDistribution[clusterKey] || 0) + 1;
      result.globalLabelDistribution[inference.autoLabel] = (result.globalLabelDistribution[inference.autoLabel] || 0) + 1;

      return {
        ...row,
        Cluster_ID: cid,
        Auto_Label: inference.autoLabel,
        Label_Confidence: Math.round(inference.confidence * 100) / 100,
        Clinical_Priority: inference.priority,
        Saran_Asupan_Nutrisi: inference.saranNutrisi,
        Tindakan_Medis_Sanitasi: inference.tindakanMedis,
        Jadwal_Monitoring: inference.jadwalMonitoring,
        Calculated_HAZ: inference.extractedHaz
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
        count: members.length,
        percentage: Math.round((members.length / enrichedRecords.length) * 1000) / 10,
        hazMean: hazMean !== undefined ? Math.round(hazMean * 100) / 100 : undefined
      });
    }

    const finalCols = [
      "Cluster_ID",
      "Auto_Label",
      "Clinical_Priority",
      "Saran_Asupan_Nutrisi",
      "Tindakan_Medis_Sanitasi",
      "Jadwal_Monitoring",
      "Label_Confidence",
      ...allCols
    ];

    result.sheets.push({
      sheetName,
      rowCount: enrichedRecords.length,
      columns: finalCols,
      featuresUsed: activeNumCols,
      records: enrichedRecords,
      clusterDistribution: sheetClusterDist,
      labelDistribution: sheetLabelDist,
      clusterProfiles
    });
  }

  return result;
}

// Generate benchmark multi-sheet workbook in memory
export function createBenchmarkWorkbook(): XLSX.WorkBook {
  const namesMale = ["Muhammad Rizky", "Alif Pratama", "Bima Arya", "Dimas Saputra", "Fikri Maulana", "Gilang Ramadhan", "Haikal Putra", "Ilham Fauzi", "Kenzo Raditya", "Raffa Putra"];
  const namesFemale = ["Aisyah Putri", "Anindita Zahra", "Cantika Dewi", "Dara Ayu", "Fadhila Nur", "Gita Lestari", "Hanum Salsabila", "Inara Kirana", "Keysha Anindya", "Naila Rahma"];
  const notesTemplates = [
    "Balita tampak kurus, riwayat diare berulang dalam 2 bulan terakhir, KMS pita kuning.",
    "Pertumbuhan optimal, anak sangat aktif, imunisasi dasar lengkap, ASI eksklusif 6 bulan.",
    "Anak sangat pendek untuk usianya, nafsu makan minim, asupan protein hewani sangat rendah, riwayat ISPA.",
    "Bawah Garis Merah (BGM) pada KMS posyandu, berat badan 2 bulan tidak naik berturut-turut (2T).",
    "Tinggi badan sesuai usia, motorik kasar baik, konsumsi makanan keluarga variatif telur dan ikan.",
    "Gejala perawakan pendek (stunting), sanitasi air sumur tercemar, riwayat berat lahir rendah (BBLR 2.1 kg).",
    "Lingkar lengan atas (LiLA) 11.2 cm (pita merah), asupan gizi makro sangat kurang, ibu bekerja.",
    "Pertumbuhan stabil di kurva hijau KMS, stimulasi tumbuh kembang baik oleh orang tua.",
    "Balita tampak lesu, nafsu makan berkurang, riwayat batuk kronis, dicurigai kontak TB keluarga.",
    "Gizi kurang dengan risiko gagal tumbuh, MPASI didominasi karbohidrat bubur tanpa variasi lauk."
  ];

  const sheet1Data = [];
  for (let i = 0; i < 28; i++) {
    const isMale = i % 2 === 0;
    const nameList = isMale ? namesMale : namesFemale;
    const name = nameList[i % nameList.length] + ` ${i + 1}`;
    const age = 6 + (i * 2) % 52;
    
    let haz: number;
    let waz: number;
    let whz: number;
    let note: string;

    if (i % 3 === 0) {
      // Stunted
      haz = -3.4 + (i % 5) * 0.2;
      waz = -2.8 + (i % 4) * 0.3;
      whz = -1.5 + (i % 3) * 0.4;
      note = notesTemplates[2];
    } else if (i % 3 === 1) {
      // Wasted
      haz = -1.8 + (i % 3) * 0.3;
      waz = -3.1 + (i % 3) * 0.2;
      whz = -2.9 + (i % 4) * 0.2;
      note = notesTemplates[3];
    } else {
      // Normal
      haz = 0.2 + (i % 4) * 0.3;
      waz = 0.1 + (i % 3) * 0.3;
      whz = 0.3 + (i % 3) * 0.2;
      note = notesTemplates[1];
    }

    const tb = Math.round((48.0 + (age * 0.95) + (haz * 3.2)) * 10) / 10;
    const bb = Math.round((3.2 + (age * 0.28) + (waz * 1.4)) * 10) / 10;

    sheet1Data.push({
      No_Registrasi: `REG-PKM-${1001 + i}`,
      Nama_Balita: name,
      Jenis_Kelamin: isMale ? "Laki-laki" : "Perempuan",
      Usia_Bulan: age,
      Berat_Badan_Kg: bb,
      Tinggi_Badan_Cm: tb,
      ZScore_TB_U: Math.round(haz * 100) / 100,
      ZScore_BB_U: Math.round(waz * 100) / 100,
      ZScore_BB_TB: Math.round(whz * 100) / 100,
      Nama_Puskesmas: "Puskesmas Cikarang Barat",
      Catatan_Pemeriksaan_Klinis: note
    });
  }

  const sheet2Data = [];
  for (let i = 0; i < 20; i++) {
    const isMale = i % 2 !== 0;
    const nameList = isMale ? namesMale : namesFemale;
    const name = nameList[i % nameList.length] + ` M-${i + 1}`;
    const age = 10 + (i * 2) % 45;
    const haz = -2.6 + (i % 6) * 0.6;
    const waz = -2.2 + (i % 5) * 0.5;
    const whz = -1.8 + (i % 4) * 0.5;
    const tb = Math.round((52.0 + (age * 0.88) + (haz * 3.0)) * 10) / 10;
    const bb = Math.round((4.0 + (age * 0.25) + (waz * 1.3)) * 10) / 10;

    sheet2Data.push({
      ID_Anak: `POS-MLT-${201 + i}`,
      Nama_Lengkap: name,
      Gender: isMale ? "L" : "P",
      Umur_Bulan: age,
      BB_Kg: bb,
      TB_Cm: tb,
      HAZ: Math.round(haz * 100) / 100,
      WAZ: Math.round(waz * 100) / 100,
      WHZ: Math.round(whz * 100) / 100,
      Posyandu: "Posyandu Melati 03",
      Keterangan_Bidan_Desa: notesTemplates[i % notesTemplates.length]
    });
  }

  const wb = XLSX.utils.book_new();
  const ws1 = XLSX.utils.json_to_sheet(sheet1Data);
  const ws2 = XLSX.utils.json_to_sheet(sheet2Data);

  XLSX.utils.book_append_sheet(wb, ws1, "Puskesmas_Kecamatan_Barat");
  XLSX.utils.book_append_sheet(wb, ws2, "Posyandu_Melati_Timur");

  return wb;
}

// Export pipeline result to formatted Excel workbook blob
export function exportResultToExcel(result: PipelineResult): ArrayBuffer {
  const wb = XLSX.utils.book_new();

  // 1. Summary Dashboard Sheet
  const summaryData = [
    { Metric: "Pipeline Engine", Value: "StuntSense CURE + IndoBERT" },
    { Metric: "Execution Timestamp", Value: result.executionTimestamp },
    { Metric: "Original File", Value: result.originalFilename },
    { Metric: "Total Records Processed", Value: result.totalRows },
    { Metric: "Total Sheets Processed", Value: result.processedSheets },
    { Metric: "CURE Clusters (k)", Value: result.cureParams.k },
    { Metric: "CURE Representatives (c)", Value: result.cureParams.c },
    { Metric: "CURE Shrinkage Factor (alpha)", Value: result.cureParams.alpha }
  ];
  const summaryWs = XLSX.utils.json_to_sheet(summaryData);
  XLSX.utils.book_append_sheet(wb, summaryWs, "Summary_Dashboard");

  // 2. Data Sheets
  for (const sheet of result.sheets) {
    const ws = XLSX.utils.json_to_sheet(sheet.records);
    XLSX.utils.book_append_sheet(wb, ws, sheet.sheetName.slice(0, 31));
  }

  const wbout = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  return wbout as ArrayBuffer;
}
