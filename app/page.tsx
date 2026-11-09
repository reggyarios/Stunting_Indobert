'use client';

import React, { useState, useRef, useMemo, useEffect, useSyncExternalStore } from 'react';
import * as XLSX from 'xlsx';
import {
  UploadCloud,
  Download,
  Search,
  X,
  FileSpreadsheet,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  ChevronDown,
  Settings2,
  Target,
  Scale,
  AlertTriangle
} from 'lucide-react';
import {
  PipelineResult,
  ChildRecord,
  AnalysisParams,
  AnalysisError,
  ServerResponse,
  MAX_SAMPLE_SIZE,
  PERMENKES_LABELS,
  RULE_LABEL_FIELD,
  RULE_LABEL_TITLE,
  columnRole,
  roleValue,
  hasExcelSignature,
  fromServerResponse,
  interpretSheet,
  referenceDateWarning,
  createBenchmarkWorkbook,
  exportResultToExcel
} from '@/lib/stunting-pipeline';

// Bila diisi (mis. https://<space>.hf.space), analisis dijalankan oleh backend Flask (CURE + IndoBERT-RCNN).
const API_URL = (process.env.NEXT_PUBLIC_API_URL || '').replace(/\/+$/, '');
const MAX_UPLOAD_MB = 64;
const MAX_K = 20;
const MAX_C = 20;
const todayLocal = () => new Date().toLocaleDateString('sv-SE'); // YYYY-MM-DD
const shortLabel = (s: string) => String(s ?? '-').split(' (')[0];

// Kolom yang sudah tampil di kolom khusus tabel
const DEDICATED_COLUMNS = new Set([
  'CURE_Cluster_ID', 'CURE_Cluster_Label', 'Cluster_Label_Confidence', 'Profil_Klaster',
  'IndoBERT_Record_Prediksi', 'IndoBERT_Record_Confidence', 'Catatan_Input_IndoBERT',
  'Prioritas_Intervensi', 'Saran_Asupan_Nutrisi', 'Tindakan_Medis_Sanitasi', 'Jadwal_Monitoring'
]);

async function checkExcelFile(file: File): Promise<string | null> {
  if (!/\.(xlsx|xls)$/i.test(file.name)) return `Berkas "${file.name}" bukan berkas Excel. Unggah berkas .xlsx atau .xls.`;
  if (file.size === 0) return `Berkas "${file.name}" kosong (0 byte).`;
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `Ukuran berkas melebihi batas ${MAX_UPLOAD_MB} MB.`;
  const head = new Uint8Array(await file.slice(0, 8).arrayBuffer());
  return hasExcelSignature(head)
    ? null
    : `Isi berkas "${file.name}" bukan workbook Excel. Berkas mungkin rusak, atau berupa CSV/PDF yang diganti ekstensinya.`;
}

function validateParams(p: AnalysisParams): string | null {
  if (!Number.isInteger(p.k) || p.k < 1 || p.k > MAX_K) return `Jumlah klaster (k) harus bilangan bulat 1–${MAX_K}.`;
  if (!Number.isInteger(p.c) || p.c < 1 || p.c > MAX_C) return `Titik perwakilan (c) harus bilangan bulat 1–${MAX_C}.`;
  if (!(p.alpha >= 0 && p.alpha <= 1)) return 'Faktor penyusutan (α) harus di antara 0 dan 1.';
  if (!Number.isInteger(p.sampleSize) || p.sampleSize < 2 || p.sampleSize > MAX_SAMPLE_SIZE) return `Ukuran sampel CURE harus bilangan bulat 2–${MAX_SAMPLE_SIZE}.`;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(p.referenceDate)) return 'Tanggal acuan usia wajib diisi (YYYY-MM-DD).';
  return null;
}

const pickRaw = (rec: ChildRecord, role: Parameters<typeof roleValue>[1]) =>
  Object.entries(rec).find(([k]) => columnRole(k) === role)?.[1];

export default function CureIndoBertStuntingPage() {
  // Hyperparameters (CURE)
  const [nClusters, setNClusters] = useState<number>(4);
  const [nRepresentatives, setNRepresentatives] = useState<number>(5);
  const [shrinkFactor, setShrinkFactor] = useState<number>(0.3);
  const [sampleSize, setSampleSize] = useState<number>(1500);
  const [referenceDateInput, setReferenceDate] = useState<string>('');
  // Default = hari ini di browser (bukan tanggal build saat prerender)
  const today = useSyncExternalStore(() => () => {}, todayLocal, () => '');
  const referenceDate = referenceDateInput || today;
  const [showAdvancedParams, setShowAdvancedParams] = useState<boolean>(false);
  const [showCrosstab, setShowCrosstab] = useState<boolean>(true);
  const [showExcluded, setShowExcluded] = useState<boolean>(false);

  // File & Processing State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [processingMessage, setProcessingMessage] = useState<string>('Menyiapkan pemrosesan data...');
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<{ message: string; details: string[] } | null>(null);
  const [serverStatus, setServerStatus] = useState<{ ok: boolean; hasCheckpoint: boolean } | null>(null);
  const [activeSheetIndex, setActiveSheetIndex] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterCluster, setFilterCluster] = useState<string>('ALL');
  const [filterLabel, setFilterLabel] = useState<string>('ALL');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 50;

  // Selected Record for Detail Modal
  const [detailRecord, setDetailRecord] = useState<ChildRecord | null>(null);

  // Berkas terakhir yang dianalisis, untuk kalkulasi ulang dengan parameter lain (mis. 4 vs 7 klaster)
  const [loadedFile, setLoadedFile] = useState<File | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cancelRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (!API_URL) return;
    fetch(`${API_URL}/api/health`)
      .then(r => r.json())
      .then(b => setServerStatus({ ok: true, hasCheckpoint: Boolean(b.has_trained_checkpoint) }))
      .catch(() => setServerStatus({ ok: false, hasCheckpoint: false }));
  }, []);

  useEffect(() => {
    if (!detailRecord) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setDetailRecord(null); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [detailRecord]);

  const analyzeInBrowser = async (file: File, params: AnalysisParams): Promise<PipelineResult> => {
    const buffer = await file.arrayBuffer();
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL('../lib/analysis.worker.ts', import.meta.url));
      const finish = () => { worker.terminate(); cancelRef.current = null; };
      cancelRef.current = () => { finish(); reject(new AnalysisError('Analisis dibatalkan.')); };
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === 'progress') {
          setProcessingProgress(m.pct);
          setProcessingMessage(m.message);
          return;
        }
        finish();
        if (m.type === 'done') resolve(m.result);
        else reject(new AnalysisError(m.message, m.details));
      };
      worker.onerror = () => { finish(); reject(new AnalysisError('Proses analisis di browser berhenti karena galat.')); };
      worker.postMessage({ buffer, filename: file.name, params }, [buffer]);
    });
  };

  const analyzeOnServer = async (file: File, params: AnalysisParams): Promise<PipelineResult> => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('n_clusters', String(params.k));
    fd.append('n_representatives', String(params.c));
    fd.append('shrink_factor', String(params.alpha));
    fd.append('sample_size', String(params.sampleSize));
    fd.append('reference_date', params.referenceDate);
    const controller = new AbortController();
    cancelRef.current = () => controller.abort();
    setProcessingProgress(40);
    setProcessingMessage('Mengunggah & menjalankan CURE + IndoBERT-RCNN di server (bisa beberapa menit untuk data besar)...');
    let res: Response;
    try {
      res = await fetch(`${API_URL}/api/analyze`, { method: 'POST', body: fd, signal: controller.signal });
    } catch {
      throw new AnalysisError(controller.signal.aborted ? 'Analisis dibatalkan.' : `Tidak dapat menghubungi server analisis (${API_URL}).`);
    }
    const body = await res.json().catch(() => null);
    if (!res.ok || !body || body.status !== 'success') {
      throw new AnalysisError(
        body?.message || `Server mengembalikan galat (HTTP ${res.status}).`,
        (body?.skipped_sheets || []).map((s: { sheet_name: string; reason: string }) => `Lembar ${s.sheet_name}: ${s.reason}`)
      );
    }
    setProcessingProgress(90);
    setProcessingMessage('Menyusun hasil...');
    return fromServerResponse(body as ServerResponse, file.name, params);
  };

  const runAnalysis = async (file: File, override: Partial<AnalysisParams> = {}) => {
    const params: AnalysisParams = {
      k: nClusters, c: nRepresentatives, alpha: shrinkFactor, sampleSize, referenceDate, ...override
    };
    const invalid = validateParams(params);
    if (invalid) {
      setError({ message: invalid, details: [] });
      return;
    }
    setError(null);
    setIsProcessing(true);
    setProcessingProgress(5);
    setProcessingMessage(`Membaca berkas ${file.name}...`);
    try {
      const result = API_URL ? await analyzeOnServer(file, params) : await analyzeInBrowser(file, params);
      setPipelineResult(result);
      setLoadedFile(file);
      setNClusters(params.k);
      setNRepresentatives(params.c);
      setShrinkFactor(params.alpha);
      setActiveSheetIndex(0);
      setFilterCluster('ALL');
      setFilterLabel('ALL');
      setSearchQuery('');
      setCurrentPage(1);
    } catch (err) {
      console.error(err);
      setError(err instanceof AnalysisError
        ? { message: err.message, details: err.details }
        : { message: 'Gagal memproses berkas.', details: [String(err)] });
    } finally {
      cancelRef.current = null;
      setIsProcessing(false);
    }
  };

  const acceptFile = async (file?: File) => {
    if (!file) return;
    const problem = await checkExcelFile(file);
    if (problem) {
      setSelectedFile(null);
      setError({ message: problem, details: [] });
      return;
    }
    setError(null);
    setSelectedFile(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    acceptFile(e.dataTransfer.files?.[0]);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    acceptFile(e.target.files?.[0]);
    e.target.value = '';
  };

  const benchmarkFile = () => {
    const buf = XLSX.write(createBenchmarkWorkbook(), { bookType: 'xlsx', type: 'array', compression: true });
    return new File([buf], 'Format_Contoh_Data_Balita.xlsx', {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
  };

  const downloadSampleBenchmark = () => {
    XLSX.writeFile(createBenchmarkWorkbook(), 'Format_Contoh_Data_Balita.xlsx', { compression: true });
  };

  const downloadEnrichedWorkbook = () => {
    if (!pipelineResult) return;
    const buffer = exportResultToExcel(pipelineResult);
    const blob = new Blob([buffer], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `Hasil_Analisis_${pipelineResult.originalFilename.replace(/\.xls$/i, '.xlsx')}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const currentSheet = pipelineResult?.sheets[activeSheetIndex];
  const labelField = pipelineResult?.labelField ?? RULE_LABEL_FIELD;
  const labelTitle = pipelineResult?.labelTitle ?? RULE_LABEL_TITLE;
  const showConfidence = Boolean(pipelineResult?.indobertActive);
  const totalExcluded = pipelineResult?.sheets.reduce((a, s) => a + s.excluded.length, 0) ?? 0;
  const dateWarnings = pipelineResult
    ? pipelineResult.sheets.map(s => referenceDateWarning(s, pipelineResult.cureParams.referenceDate)).filter((w): w is string => w !== null)
    : [];
  const trailingColumns = useMemo(
    () => currentSheet?.columns.filter(c => !DEDICATED_COLUMNS.has(c) && c !== labelField) ?? [],
    [currentSheet, labelField]
  );
  const interpretation = useMemo(
    () => (currentSheet && pipelineResult ? interpretSheet(currentSheet, pipelineResult) : []),
    [currentSheet, pipelineResult]
  );

  const modeBadge = (() => {
    const client = { text: 'Mode tanpa model: CURE + aturan Permenkes', ok: false };
    const serverNoModel = { text: 'Server tanpa checkpoint: CURE + aturan Permenkes', ok: false };
    const model = { text: 'Model IndoBERT-RCNN (server)', ok: true };
    if (pipelineResult) return pipelineResult.mode === 'client' ? client : pipelineResult.indobertActive ? model : serverNoModel;
    if (!API_URL) return client;
    if (!serverStatus) return { text: 'Memeriksa server analisis...', ok: false };
    if (!serverStatus.ok) return { text: 'Server analisis tidak terjangkau', ok: false };
    return serverStatus.hasCheckpoint ? model : serverNoModel;
  })();

  // Filtering records
  const filteredRecords = useMemo(() => {
    if (!currentSheet) return [];
    const q = searchQuery.toLowerCase();
    return currentSheet.records.filter(r => {
      const matchesSearch = q === '' || Object.values(r).some(val => String(val).toLowerCase().includes(q));
      const matchesCluster = filterCluster === 'ALL' || String(r.CURE_Cluster_ID) === filterCluster;
      const matchesLabel = filterLabel === 'ALL' || r[labelField] === filterLabel;
      return matchesSearch && matchesCluster && matchesLabel;
    });
  }, [currentSheet, searchQuery, filterCluster, filterLabel, labelField]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredRecords.length / pageSize));
  const paginatedRecords = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredRecords.slice(start, start + pageSize);
  }, [filteredRecords, currentPage, pageSize]);

  const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '-' : v.toFixed(d));

  return (
    <div className="min-h-screen flex flex-col bg-[#f9fbf9] text-[#223124] font-sans antialiased">

      {/* Top Header */}
      <header className="border-b border-[#e2eae2] bg-white/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 min-h-16 py-2 flex flex-wrap items-center justify-between gap-2">

          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#58725c]"></span>
            <div className="flex items-baseline gap-1.5">
              <span className="text-base font-bold tracking-tight text-[#1e2920]">
                CURE-IndoBERT
              </span>
              <span className="text-xs font-semibold text-[#58725c] tracking-normal">
                for Stunting
              </span>
            </div>
          </div>

          <div className="hidden lg:block text-xs text-[#5c6e5e]">
            Klasterisasi CURE &middot; Klasifikasi IndoBERT &middot; Standar Permenkes No 2/2020 &amp; WHO
          </div>

          {/* Actions */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={downloadSampleBenchmark}
              className="text-xs font-medium text-[#465b49] hover:text-[#253628] border border-[#d2ded2] hover:border-[#b4c7b4] bg-[#f4f7f4] rounded-md px-3.5 py-1.5 transition-colors whitespace-nowrap"
            >
              Unduh Format Contoh
            </button>
            {pipelineResult && (
              <button
                onClick={downloadEnrichedWorkbook}
                className="bg-[#58725c] hover:bg-[#49604d] text-white text-xs font-medium px-4 py-1.5 rounded-md transition-colors whitespace-nowrap flex items-center gap-1.5 shadow-sm"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Unduh Hasil Excel</span>
              </button>
            )}
          </div>

        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-10 space-y-8">

        {/* Hero Section */}
        <section className="space-y-2 max-w-3xl">
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-[#eef4ee] border border-[#d6e3d6] text-[11px] font-medium text-[#465c49]">
              <Sparkles className="w-3 h-3 text-[#58725c]" />
              <span>Platform Riset Klasterisasi &amp; Klasifikasi Antropometri Balita</span>
            </div>
            <div
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[11px] font-semibold ${
                modeBadge.ok ? 'bg-[#e7f3e9] border-[#bcd9c1] text-[#245b2d]' : 'bg-[#fcf6ec] border-[#e8d8bb] text-[#6e5326]'
              }`}
              title={API_URL ? `Backend: ${API_URL}` : 'NEXT_PUBLIC_API_URL tidak diset: analisis berjalan di browser tanpa model IndoBERT'}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${modeBadge.ok ? 'bg-[#388e3c]' : 'bg-[#b07d32]'}`}></span>
              <span>{modeBadge.text}</span>
            </div>
          </div>
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1b261d]">
            CURE-IndoBERT for Stunting Analytics
          </h1>
          <p className="text-xs sm:text-sm text-[#556958] leading-relaxed">
            Sistem analisis data pertumbuhan balita terintegrasi: Klasterisasi spasial CURE (*Clustering Using Representatives*) untuk pengelompokan pola antropometri dan model IndoBERT untuk interpretasi klinis serta perumusan rekomendasi intervensi gizi terpersonalisasi sesuai standar Kemenkes RI &amp; WHO.
          </p>
          {!API_URL && (
            <p className="text-[11px] text-[#7a6340] leading-relaxed">
              Mode tanpa model: klaster dihitung dengan algoritma CURE di browser; label per anak adalah {RULE_LABEL_TITLE}, bukan prediksi IndoBERT.
            </p>
          )}
        </section>

        {/* Error / validation feedback */}
        {error && (
          <div role="alert" className="max-w-3xl p-3.5 rounded-lg border border-[#ebd0d0] bg-[#fdf5f5] text-xs text-[#912d2d] space-y-1.5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2 font-semibold">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error.message}</span>
              </div>
              <button onClick={() => setError(null)} aria-label="Tutup pesan" className="p-0.5 hover:bg-white/60 rounded">
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            {error.details.length > 0 && (
              <ul className="list-disc list-inside space-y-0.5 pl-6 font-normal">
                {error.details.slice(0, 12).map((d, i) => <li key={i}>{d}</li>)}
              </ul>
            )}
          </div>
        )}

        {/* Upload & Configuration Card (When no result is loaded) */}
        {!pipelineResult && (
          <div className="bg-white border border-[#e2ebe2] rounded-xl p-5 sm:p-8 space-y-6 shadow-xs max-w-3xl">

            <div className="border-b border-[#edf2ed] pb-4 flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-sm font-semibold text-[#1e2920]">Unggah Berkas Antropometri Balita</h2>
                <p className="text-xs text-[#6e8070] mt-0.5">Unggah data Posyandu / e-PPGBM Puskesmas (.xlsx atau .xls)</p>
              </div>
              <span className="text-[11px] font-mono text-[#58725c] bg-[#eef4ee] px-2 py-0.5 rounded">
                Mendukung Multi-Sheet
              </span>
            </div>

            {/* Dropzone */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
                selectedFile || isDragging
                  ? 'border-[#58725c] bg-[#f4f8f4]'
                  : 'border-[#ccd9cc] hover:border-[#6e8a72] bg-[#fafcfa] hover:bg-[#f3f7f3]'
              } space-y-3`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
                className="hidden"
              />
              <div className="w-12 h-12 mx-auto rounded-full bg-[#ebf2eb] flex items-center justify-center text-[#556e59]">
                <UploadCloud className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <div className="text-sm font-medium text-[#243527]">
                  {selectedFile ? 'Berkas Telah Dipilih — Klik untuk Mengganti' : 'Pilih atau seret berkas Excel Anda ke sini'}
                </div>
                <div className="text-xs text-[#687b6a]">
                  Kolom wajib: JK, Berat, Tinggi, ZS TB/U, ZS BB/U, ZS BB/TB, serta Usia (bulan) atau Tgl Lahir. Kolom nama/NIK/alamat tidak ditampilkan.
                </div>
              </div>
            </div>

            {/* Selected File Details */}
            {selectedFile && (
              <div className="p-3.5 border border-[#cde0cd] bg-[#f1f7f1] rounded-lg flex items-center justify-between text-xs">
                <div className="flex items-center gap-2.5 truncate">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#58725c] shrink-0"></span>
                  <div className="truncate">
                    <div className="font-semibold text-[#1e2f20] truncate">{selectedFile.name}</div>
                    <div className="text-[#59705b] font-mono text-[11px]">
                      {(selectedFile.size / 1024).toFixed(1)} KB &middot; Siap diproses
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-[#6d826f] hover:text-[#9e3a3a] text-xs font-medium ml-3 shrink-0 px-2 py-1 hover:bg-white/60 rounded"
                >
                  Hapus
                </button>
              </div>
            )}

            {/* CURE Parameters - Collapsible Advanced Panel */}
            <div className="pt-2 border-t border-[#edf2ed]">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setShowAdvancedParams(!showAdvancedParams)}
                  className="inline-flex items-center gap-1.5 text-xs text-[#415344] hover:text-[#1e2920] font-medium group transition-colors"
                >
                  <Settings2 className="w-3.5 h-3.5 text-[#58725c] group-hover:rotate-45 transition-transform" />
                  <span>Pengaturan Parameter CURE &amp; Tanggal Acuan</span>
                  <ChevronDown className={`w-3.5 h-3.5 text-[#6a7d6c] transition-transform duration-200 ${showAdvancedParams ? 'rotate-180' : ''}`} />
                </button>
                <span className="font-mono text-[#58725c] bg-[#eef4ee] px-2 py-0.5 rounded text-[11px] self-start sm:self-auto">
                  Aktif: k={nClusters} &middot; c={nRepresentatives} &middot; &alpha;={shrinkFactor.toFixed(2)} &middot; acuan {referenceDate || '-'}
                </span>
              </div>

              {showAdvancedParams && (
                <div className="mt-3 space-y-3 p-3.5 rounded-lg bg-[#fafcfa] border border-[#e5ebe5] transition-all">
                  <div className="text-[11px] text-[#58725c] bg-[#eef4ee] p-2 rounded border border-[#dce8dc] leading-relaxed">
                    💡 <strong>Catatan Penggunaan:</strong> Nilai k, c, dan &alpha; bawaan sudah cukup untuk penggunaan lapangan; ubah hanya untuk pengujian penelitian. <strong>Tanggal acuan</strong> penting untuk data e-PPGBM: bila berkas tidak punya kolom usia atau tanggal ukur, usia dihitung sebagai tanggal acuan &minus; Tgl Lahir.
                  </div>

                  {/* Quick Preset Buttons */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2.5 bg-white rounded-md border border-[#dce8dc]">
                    <div className="text-xs text-[#2d3e30] font-medium">Pilihan Jumlah Klaster:</div>
                    <div className="flex flex-wrap items-center gap-1.5">
                      {[4, 7].map(k => (
                        <button
                          key={k}
                          type="button"
                          onClick={() => {
                            setNClusters(k);
                            setNRepresentatives(5);
                            setShrinkFactor(0.3);
                            if (loadedFile) runAnalysis(loadedFile, { k, c: 5, alpha: 0.3 });
                          }}
                          className={`text-xs px-2.5 py-1 rounded transition-all ${
                            nClusters === k
                              ? 'bg-[#58725c] text-white font-medium shadow-xs'
                              : 'bg-[#f4f7f4] text-[#3d5140] hover:bg-[#e6eee6] border border-[#d2ded2]'
                          }`}
                        >
                          {k === 4 ? '🎯 4 Klaster (Riset Fokus Stunting)' : '🏆 7 Klaster (= jumlah kategori acuan Permenkes)'}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                    {/* Parameter k */}
                    <div className="space-y-1.5 p-2.5 rounded-md bg-white border border-[#e2eae2]">
                      <div className="flex justify-between items-center text-xs">
                        <label htmlFor="param-k" className="font-medium text-[#2d3e30]">Jumlah Klaster (k)</label>
                        <span className="font-mono font-semibold text-[#58725c]">{nClusters} Klaster</span>
                      </div>
                      <input
                        id="param-k"
                        type="range"
                        min="1"
                        max={MAX_K}
                        value={nClusters}
                        onChange={(e) => setNClusters(parseInt(e.target.value))}
                        className="w-full accent-[#58725c] cursor-pointer"
                      />
                      <div className="text-[10px] text-[#6b7c6d] leading-snug">
                        {nClusters === 7
                          ? 'k disamakan dengan 7 kategori acuan Permenkes; nama klaster tetap ditentukan dari isi klaster.'
                          : `Mengelompokkan balita ke dalam ${nClusters} klaster CURE (rentang 1–${MAX_K}).`}
                      </div>
                    </div>

                    {/* Parameter c (Titik Perwakilan) */}
                    <div className="space-y-1.5 p-2.5 rounded-md bg-white border border-[#e2eae2]">
                      <div className="flex justify-between items-center text-xs">
                        <label htmlFor="param-c" className="font-medium text-[#2d3e30]">Titik Perwakilan (c)</label>
                        <span className="font-mono font-semibold text-[#58725c]">{nRepresentatives} Titik</span>
                      </div>
                      <input
                        id="param-c"
                        type="range"
                        min="1"
                        max={MAX_C}
                        value={nRepresentatives}
                        onChange={(e) => setNRepresentatives(parseInt(e.target.value))}
                        className="w-full accent-[#58725c] cursor-pointer"
                      />
                      <div className="text-[10px] text-[#6b7c6d] leading-snug">
                        Titik sampel terluar untuk menangkap bentuk sebaran non-lingkaran (rentang 1–{MAX_C}).
                      </div>
                    </div>

                    {/* Parameter alpha (Faktor Penyusutan) */}
                    <div className="space-y-1.5 p-2.5 rounded-md bg-white border border-[#e2eae2]">
                      <div className="flex justify-between items-center text-xs">
                        <label htmlFor="param-alpha" className="font-medium text-[#2d3e30]">Faktor Penyusutan (&alpha;)</label>
                        <span className="font-mono font-semibold text-[#58725c]">{shrinkFactor.toFixed(2)}</span>
                      </div>
                      <input
                        id="param-alpha"
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={shrinkFactor}
                        onChange={(e) => setShrinkFactor(parseFloat(e.target.value))}
                        className="w-full accent-[#58725c] cursor-pointer"
                      />
                      <div className="text-[10px] text-[#6b7c6d] leading-snug">
                        Menarik titik perwakilan ke centroid untuk meredam pencilan (rentang 0–1).
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="space-y-1.5 p-2.5 rounded-md bg-white border border-[#e2eae2]">
                      <label htmlFor="param-ref-date" className="block text-xs font-medium text-[#2d3e30]">Tanggal Acuan Usia (tanggal ukur)</label>
                      <input
                        id="param-ref-date"
                        type="date"
                        value={referenceDate}
                        onChange={(e) => setReferenceDate(e.target.value)}
                        className="w-full bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#223124] focus:outline-none focus:border-[#58725c]"
                      />
                      <div className="text-[10px] text-[#6b7c6d] leading-snug">
                        Dipakai bila tidak ada kolom usia / tanggal ukur. Isi dengan tanggal pengukuran (mis. bulan timbang).
                      </div>
                    </div>
                    <div className="space-y-1.5 p-2.5 rounded-md bg-white border border-[#e2eae2]">
                      <label htmlFor="param-sample" className="block text-xs font-medium text-[#2d3e30]">Ukuran Sampel CURE</label>
                      <input
                        id="param-sample"
                        type="number"
                        min={2}
                        max={MAX_SAMPLE_SIZE}
                        value={Number.isNaN(sampleSize) ? '' : sampleSize}
                        onChange={(e) => setSampleSize(e.target.value === '' ? NaN : Number(e.target.value))}
                        className="w-full bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#223124] focus:outline-none focus:border-[#58725c]"
                      />
                      <div className="text-[10px] text-[#6b7c6d] leading-snug">
                        Tahap hierarkis CURE dijalankan pada sampel acak (Guha dkk. 1998), lalu semua anak diberi label ke representatif terdekat (2–{MAX_SAMPLE_SIZE}).
                      </div>
                    </div>
                  </div>

                  {loadedFile && (
                    <div className="pt-2 flex justify-end">
                      <button
                        type="button"
                        onClick={() => runAnalysis(loadedFile)}
                        className="text-xs bg-[#58725c] hover:bg-[#49604d] text-white px-3 py-1.5 rounded-md font-medium transition-colors shadow-xs"
                      >
                        🔄 Terapkan &amp; Kalkulasi Ulang Klaster ({nClusters} Klaster)
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
              <button
                disabled={!selectedFile || isProcessing}
                onClick={() => selectedFile && runAnalysis(selectedFile)}
                className="w-full sm:w-auto flex-1 bg-[#58725c] hover:bg-[#49604d] disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-5 rounded-md text-xs transition-colors shadow-xs truncate"
              >
                {selectedFile ? `Mulai Analisis Berkas (${selectedFile.name})` : 'Pilih Berkas Excel Dahulu'}
              </button>
              <button
                onClick={() => runAnalysis(benchmarkFile())}
                disabled={isProcessing}
                className="w-full sm:w-auto border border-[#ccd9cc] hover:border-[#b4c7b4] text-[#3d5140] hover:text-[#1e2920] bg-white font-medium py-2 px-4 rounded-md text-xs transition-colors"
              >
                Gunakan Data Contoh
              </button>
            </div>

          </div>
        )}

        {/* Processing Modal */}
        {isProcessing && (
          <div className="fixed inset-0 z-50 bg-[#1e2920]/25 backdrop-blur-xs flex items-center justify-center p-6">
            <div className="max-w-sm w-full bg-white border border-[#dbe5db] rounded-xl p-6 space-y-4 shadow-lg text-center">
              <div className="w-8 h-8 mx-auto rounded-full bg-[#eef4ee] flex items-center justify-center text-[#58725c] animate-pulse">
                <FileSpreadsheet className="w-4 h-4" />
              </div>
              <div className="space-y-1">
                <div className="text-sm font-semibold text-[#1e2920]">Sedang Memproses Data</div>
                <div className="text-xs text-[#556957]">{processingMessage}</div>
              </div>
              <div className="w-full bg-[#ebf2eb] rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-[#58725c] h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${processingProgress}%` }}
                ></div>
              </div>
              <button
                type="button"
                onClick={() => cancelRef.current?.()}
                className="text-xs font-medium text-[#6d826f] hover:text-[#9e3a3a] border border-[#d2ded2] rounded-md px-3 py-1"
              >
                Batalkan
              </button>
            </div>
          </div>
        )}

        {/* RESULTS DASHBOARD */}
        {pipelineResult && currentSheet && (
          <div className="space-y-6">

            {/* Header Result Action */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#e2ebe2] pb-5">
              <div className="min-w-0">
                <div className="text-xs text-[#687c69] font-medium truncate">
                  Berkas: {pipelineResult.originalFilename}
                </div>
                <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-[#1b261d] mt-0.5">
                  Hasil Analisis Status Gizi & Rekomendasi
                </h2>
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => setPipelineResult(null)}
                  className="text-xs font-medium text-[#465b49] hover:text-[#1e2920] border border-[#d2ded2] hover:border-[#b4c7b4] bg-white px-3.5 py-1.5 rounded-md transition-colors whitespace-nowrap"
                >
                  Unggah Berkas Baru
                </button>
                <button
                  onClick={downloadEnrichedWorkbook}
                  className="bg-[#58725c] hover:bg-[#49604d] text-white text-xs font-medium px-4 py-1.5 rounded-md transition-colors flex items-center gap-1.5 shadow-sm whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>Unduh Excel Hasil</span>
                </button>
              </div>
            </div>

            {/* Metrics Summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="bg-white border border-[#e2ebe2] rounded-lg p-4 shadow-xs">
                <div className="text-xs text-[#667a68]">Total Data Anak</div>
                <div className="text-2xl font-semibold text-[#1e2920] font-mono mt-1 tabular-nums">
                  {pipelineResult.totalRows.toLocaleString('id-ID')}
                </div>
                <div className="text-[11px] text-[#7d907f] mt-0.5">Lolos validasi, seluruh lembar</div>
              </div>

              <div className="bg-white border border-[#e2ebe2] rounded-lg p-4 shadow-xs">
                <div className="text-xs text-[#667a68]">Jumlah Lembar / Tahun</div>
                <div className="text-2xl font-semibold text-[#1e2920] font-mono mt-1 tabular-nums">
                  {pipelineResult.processedSheets}
                </div>
                <div className="text-[11px] text-[#7d907f] mt-0.5">Data Posyandu & Puskesmas</div>
              </div>

              <div className="bg-white border border-[#e2ebe2] rounded-lg p-4 shadow-xs">
                <div className="text-xs text-[#667a68]">Parameter CURE</div>
                <div className="text-2xl font-semibold text-[#58725c] font-mono mt-1 tabular-nums">
                  k={pipelineResult.cureParams.k}
                </div>
                <div className="text-[11px] text-[#7d907f] font-mono mt-0.5">
                  c={pipelineResult.cureParams.c} &middot; &alpha;={pipelineResult.cureParams.alpha.toFixed(2)} &middot; n<sub>sampel</sub>={pipelineResult.cureParams.sampleSize}
                </div>
              </div>

              <div className="bg-white border border-[#e2ebe2] rounded-lg p-4 shadow-xs">
                <div className="text-xs text-[#667a68]">Pedoman Standar</div>
                <div className="text-2xl font-semibold text-[#1e2920] mt-1 text-base leading-tight pt-1">
                  Kemenkes RI
                </div>
                <div className="text-[11px] text-[#7d907f] mt-0.5">Permenkes No 2/2020</div>
              </div>
            </div>

            {/* Tanggal acuan kemungkinan salah: banyak balita keluar karena usia */}
            {dateWarnings.length > 0 && (
              <div role="alert" className="p-3.5 rounded-lg border border-[#ebd0d0] bg-[#fdf5f5] text-xs text-[#912d2d] space-y-1.5">
                <div className="flex items-center gap-2 font-semibold">
                  <AlertTriangle className="w-4 h-4 shrink-0" />
                  <span>Periksa tanggal acuan usia: hasil ini kemungkinan tidak valid</span>
                </div>
                {dateWarnings.map(w => <p key={w} className="pl-6 leading-relaxed">{w}</p>)}
              </div>
            )}

            {/* Excluded rows & skipped sheets */}
            {(totalExcluded > 0 || pipelineResult.skippedSheets.length > 0) && (
              <div className="p-3.5 rounded-lg border border-[#e8ded0] bg-[#fcf9f5] text-xs text-[#5c4a30] space-y-2">
                {totalExcluded > 0 && (
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span>
                      <strong>{totalExcluded.toLocaleString('id-ID')} baris dikeluarkan</strong> dari analisis: data tidak lengkap, WHO flag (|ZS TB/U|&gt;6, ZS BB/U &lt;−6 atau &gt;+5, |ZS BB/TB|&gt;5), atau berat/tinggi/usia tidak wajar. Lembar {currentSheet.sheetName}: {currentSheet.excluded.length.toLocaleString('id-ID')} baris.
                    </span>
                    {currentSheet.excluded.length > 0 && (
                      <button
                        type="button"
                        onClick={() => setShowExcluded(!showExcluded)}
                        className="text-xs text-[#5c4a30] border border-[#e0d0b5] hover:bg-white px-2.5 py-1 rounded-md flex items-center gap-1"
                      >
                        <span>{showExcluded ? 'Sembunyikan Rincian' : 'Lihat Rincian'}</span>
                        <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${showExcluded ? 'rotate-180' : ''}`} />
                      </button>
                    )}
                  </div>
                )}
                {pipelineResult.skippedSheets.map(s => (
                  <div key={s.sheetName}>Lembar <strong>{s.sheetName}</strong> tidak dianalisis: {s.reason}</div>
                ))}
                {showExcluded && currentSheet.excluded.length > 0 && (
                  <div className="max-h-64 overflow-y-auto border border-[#ecdfc9] rounded bg-white">
                    <table className="w-full text-left text-[11px]">
                      <thead className="bg-[#faf5ec] sticky top-0">
                        <tr>
                          <th className="px-3 py-1.5 font-semibold whitespace-nowrap">Baris Excel</th>
                          <th className="px-3 py-1.5 font-semibold">Alasan</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#f3ebdc]">
                        {currentSheet.excluded.slice(0, 500).map(e => (
                          <tr key={e.row}>
                            <td className="px-3 py-1 font-mono tabular-nums">{e.row}</td>
                            <td className="px-3 py-1">{e.reasons.join('; ')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {currentSheet.excluded.length > 500 && (
                      <div className="px-3 py-1.5 text-[11px] text-[#7a6340]">
                        &hellip; dan {(currentSheet.excluded.length - 500).toLocaleString('id-ID')} baris lain (lengkap di Excel hasil, lembar Baris_Dikeluarkan).
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Summary Distributions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

              {/* Cluster Distribution */}
              <div className="bg-white border border-[#e2ebe2] rounded-lg p-5 space-y-3 shadow-xs min-w-0">
                <div className="flex items-center justify-between gap-2 border-b border-[#edf2ed] pb-2.5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#3d5140]">
                    Pengelompokan Klaster CURE
                  </h3>
                  <span className="text-xs font-mono text-[#7d907f] whitespace-nowrap">
                    Lembar {currentSheet.sheetName}
                  </span>
                </div>
                <div className="space-y-2.5">
                  {currentSheet.clusterProfiles.map(cp => (
                    <div key={cp.clusterId} className="space-y-1">
                      <div className="flex items-start justify-between gap-3 text-xs">
                        <span className="text-[#2b3c2e] font-medium min-w-0" title={cp.name}>{cp.name}</span>
                        <span className="font-mono text-[#5f7361] tabular-nums shrink-0">
                          {cp.count.toLocaleString('id-ID')} anak ({cp.percentage}%)
                        </span>
                      </div>
                      {cp.semanticLabel && (
                        <div className="text-[11px] text-[#245b2d]">
                          Label IndoBERT-RCNN (konsensus): {cp.semanticLabel}
                          {cp.semanticConfidence !== null && cp.semanticConfidence !== undefined && ` · ${cp.semanticConfidence}`}
                        </div>
                      )}
                      <div className="w-full bg-[#ebf2eb] rounded-full h-1.5 overflow-hidden">
                        <div className="bg-[#58725c] h-1.5 rounded-full" style={{ width: `${cp.percentage}%` }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Status Distribution */}
              <div className="bg-white border border-[#e2ebe2] rounded-lg p-5 space-y-3 shadow-xs min-w-0">
                <div className="flex items-center justify-between gap-2 border-b border-[#edf2ed] pb-2.5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#3d5140]">
                    {labelTitle}
                  </h3>
                  <span className="text-xs font-mono text-[#7d907f] whitespace-nowrap">Lembar {currentSheet.sheetName}</span>
                </div>
                <div className="space-y-2.5">
                  {Object.entries(currentSheet.labelDistribution).sort((a, b) => b[1] - a[1]).map(([label, count]) => {
                    const pct = Math.round((count / currentSheet.rowCount) * 1000) / 10;
                    return (
                      <div key={label} className="space-y-1">
                        <div className="flex items-center justify-between gap-3 text-xs">
                          <span className="text-[#2b3c2e] font-medium truncate" title={label}>{label}</span>
                          <span className="font-mono text-[#5f7361] tabular-nums shrink-0">
                            {count.toLocaleString('id-ID')} ({pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-[#ebf2eb] rounded-full h-1.5 overflow-hidden">
                          <div className="bg-[#78967c] h-1.5 rounded-full" style={{ width: `${pct}%` }}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

            </div>

            {/* Research Validation: CURE vs Permenkes reference categories */}
            <div className="bg-white border border-[#d6e2d6] rounded-lg p-5 shadow-xs space-y-4">
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-[#edf2ed] pb-3.5">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-[#435e47] shrink-0" />
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-[#263728]">
                      Evaluasi Penelitian: Keselarasan Klaster CURE vs {RULE_LABEL_TITLE}
                    </h3>
                  </div>
                  <p className="text-[11px] text-[#607462]">
                    Purity dihitung terhadap kategori acuan Permenkes No. 2/2020 (ambang z-score), bukan terhadap prediksi model. Lembar {currentSheet.sheetName}.
                  </p>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {/* Instant Switch between 4 and 7 Clusters */}
                  {loadedFile && (
                    <div className="flex items-center gap-1.5 bg-[#f0f4f0] p-1 rounded-md border border-[#d6e2d6]">
                      <span className="text-[11px] font-medium text-[#465c49] px-1.5">Uji Klaster:</span>
                      {[4, 7].map(k => (
                        <button
                          key={k}
                          type="button"
                          onClick={() => runAnalysis(loadedFile, { k, c: pipelineResult.cureParams.c, alpha: pipelineResult.cureParams.alpha })}
                          className={`text-[11px] px-2.5 py-1 rounded transition-all ${
                            pipelineResult.cureParams.k === k
                              ? 'bg-[#58725c] text-white font-semibold shadow-xs'
                              : 'text-[#536b56] hover:text-[#253727]'
                          }`}
                        >
                          {k} Klaster
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="flex items-center gap-1.5 bg-[#eef5ee] border border-[#d0ded0] px-3 py-1.5 rounded-md">
                    <Scale className="w-3.5 h-3.5 text-[#3b533e]" />
                    <span className="text-xs text-[#3b533e] font-medium">Purity vs kategori Permenkes:</span>
                    <span className="font-mono font-bold text-xs text-[#2b402e]">
                      {currentSheet.purityScore === null ? '-' : `${currentSheet.purityScore.toFixed(2)}%`}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5 bg-[#f5f8f5] border border-[#d0ded0] px-3 py-1.5 rounded-md">
                    <span className="text-xs text-[#3b533e] font-medium">Silhouette:</span>
                    <span className="font-mono font-bold text-xs text-[#2b402e]">{fmt(currentSheet.silhouette, 4)}</span>
                  </div>

                  <button
                    type="button"
                    onClick={() => setShowCrosstab(!showCrosstab)}
                    className="text-xs text-[#4e6351] hover:text-[#263728] border border-[#d6e2d6] hover:bg-[#f7faf7] px-2.5 py-1.5 rounded-md flex items-center gap-1 transition-colors"
                  >
                    <span>{showCrosstab ? 'Sembunyikan Matriks' : 'Tampilkan Matriks'}</span>
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${showCrosstab ? 'rotate-180' : ''}`} />
                  </button>
                </div>
              </div>

              {showCrosstab && (() => {
                const crosstab = currentSheet.crosstab;
                const clusterKeys = currentSheet.clusterProfiles.map(p => p.name).filter(n => crosstab[n]);
                const present = new Set(clusterKeys.flatMap(ck => Object.keys(crosstab[ck])));
                const categories = Object.values(PERMENKES_LABELS).map(shortLabel).filter(c => present.has(c));

                return (
                  <div className="space-y-3.5">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="bg-[#f5f8f5] border-b border-[#e2ebe2] text-[#344736]">
                            <th className="py-2.5 px-3 font-semibold whitespace-nowrap">Klaster CURE (Unsupervised)</th>
                            {categories.map(label => (
                              <th key={label} className="py-2.5 px-3 font-semibold whitespace-nowrap text-center">
                                {label}
                              </th>
                            ))}
                            <th className="py-2.5 px-3 font-semibold whitespace-nowrap text-right">Total Anggota</th>
                            <th className="py-2.5 px-3 font-semibold whitespace-nowrap text-right">Kategori Mayoritas</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#edf2ed]">
                          {clusterKeys.map(ck => {
                            const rowCounts = crosstab[ck] || {};
                            const totalRow = Object.values(rowCounts).reduce((a, b) => a + b, 0);
                            let maxLabel = '-';
                            let maxVal = 0;
                            Object.entries(rowCounts).forEach(([lbl, cnt]) => {
                              if (cnt > maxVal) {
                                maxVal = cnt;
                                maxLabel = lbl;
                              }
                            });
                            const maxPct = totalRow > 0 ? Math.round((maxVal / totalRow) * 1000) / 10 : 0;

                            return (
                              <tr key={ck} className="hover:bg-[#fafcfa]">
                                <td className="py-2 px-3 font-medium text-[#213223] whitespace-nowrap max-w-[260px] truncate" title={ck}>
                                  {ck}
                                </td>
                                {categories.map(label => {
                                  const val = rowCounts[label] || 0;
                                  const isMax = val === maxVal && val > 0;
                                  return (
                                    <td
                                      key={label}
                                      className={`py-2 px-3 text-center tabular-nums whitespace-nowrap ${
                                        isMax ? 'bg-[#edf4ed] font-semibold text-[#283e2b]' : 'text-[#6b7f6d]'
                                      }`}
                                    >
                                      {val > 0 ? val.toLocaleString('id-ID') : '-'}
                                    </td>
                                  );
                                })}
                                <td className="py-2 px-3 text-right font-mono font-medium text-[#2d3e30] tabular-nums whitespace-nowrap">
                                  {totalRow.toLocaleString('id-ID')}
                                </td>
                                <td className="py-2 px-3 text-right text-[11px] text-[#4f6452] whitespace-nowrap">
                                  <span className="font-medium text-[#2d3e30]">{maxLabel}</span> ({maxPct}%)
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>

                    {/* Interpretation computed from the actual numbers */}
                    <div className="p-3 bg-[#f8faf8] border border-[#e2ebe2] rounded-md text-[11px] text-[#465c49] space-y-1.5 leading-relaxed">
                      <div className="font-semibold text-[#223525] flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-[#3b533e]" />
                        Interpretasi Temuan (dihitung otomatis dari hasil lembar {currentSheet.sheetName}):
                      </div>
                      <ul className="list-disc list-inside space-y-1.5 text-[#4e6451] pl-1">
                        {interpretation.map((line, i) => <li key={i}>{line}</li>)}
                      </ul>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Table Viewport */}
            <div className="bg-white border border-[#e2ebe2] rounded-lg overflow-hidden shadow-xs">

              {/* Sheet Switcher & Search Bar */}
              <div className="border-b border-[#edf2ed] bg-[#fafcfa] px-4 sm:px-6 py-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3">

                {/* Tabs */}
                <div className="flex items-center gap-1.5 overflow-x-auto">
                  {pipelineResult.sheets.map((sheet, idx) => (
                    <button
                      key={sheet.sheetName}
                      onClick={() => {
                        setActiveSheetIndex(idx);
                        setFilterCluster('ALL');
                        setFilterLabel('ALL');
                        setCurrentPage(1);
                      }}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                        activeSheetIndex === idx
                          ? 'bg-white text-[#1e2920] border border-[#cfdacd] shadow-xs'
                          : 'text-[#627764] hover:text-[#253628]'
                      }`}
                    >
                      Lembar {sheet.sheetName} ({sheet.rowCount.toLocaleString('id-ID')} anak)
                    </button>
                  ))}
                </div>

                {/* Filters and Search */}
                <div className="flex flex-wrap items-center gap-2.5">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-[#7d907f]" />
                    <input
                      type="text"
                      placeholder="Cari balita..."
                      aria-label="Cari balita"
                      value={searchQuery}
                      onChange={(e) => {
                        setSearchQuery(e.target.value);
                        setCurrentPage(1);
                      }}
                      className="bg-white border border-[#cfdacd] rounded-md pl-8 pr-3 py-1 text-xs text-[#223124] focus:outline-none focus:border-[#58725c] w-36 sm:w-44"
                    />
                  </div>

                  <select
                    value={filterCluster}
                    aria-label="Filter klaster"
                    onChange={(e) => {
                      setFilterCluster(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#324534] focus:outline-none focus:border-[#58725c] max-w-[16rem]"
                  >
                    <option value="ALL">Semua Klaster</option>
                    {currentSheet.clusterProfiles.map((cp) => (
                      <option key={cp.clusterId} value={String(cp.clusterId)}>
                        {cp.name} ({cp.count})
                      </option>
                    ))}
                  </select>

                  <select
                    value={filterLabel}
                    aria-label="Filter status gizi"
                    onChange={(e) => {
                      setFilterLabel(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#324534] focus:outline-none focus:border-[#58725c] max-w-[16rem]"
                  >
                    <option value="ALL">Semua: {labelTitle}</option>
                    {Object.keys(currentSheet.labelDistribution).map((lbl) => (
                      <option key={lbl} value={lbl}>
                        {shortLabel(lbl)} ({currentSheet.labelDistribution[lbl]})
                      </option>
                    ))}
                  </select>
                </div>

              </div>

              {/* Data Table */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs divide-y divide-[#edf2ed]">
                  <thead className="bg-[#f5f8f5] text-[#415344] font-medium">
                    <tr>
                      <th className="px-4 py-2.5 whitespace-nowrap">Klaster CURE</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">{labelTitle}</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Prioritas</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Saran Asupan Nutrisi</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Tindakan Medis & Sanitasi</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Jadwal Pantau</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Rincian</th>
                      {trailingColumns.map(col => (
                        <th key={col} className="px-4 py-2.5 whitespace-nowrap font-normal text-[#647766]">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#f2f6f2] bg-white">
                    {paginatedRecords.length === 0 ? (
                      <tr>
                        <td colSpan={7 + trailingColumns.length} className="px-4 py-8 text-center text-xs text-[#7d907f]">
                          Tidak ada data balita yang cocok dengan filter pencarian.
                        </td>
                      </tr>
                    ) : (
                      paginatedRecords.map((row) => (
                        <tr key={`${row.Baris_Excel}`} className="hover:bg-[#f8faf8] transition-colors">
                          <td className="px-4 py-2.5 whitespace-nowrap">
                            <div className="font-mono text-[#435e47] font-semibold">
                              {Number(row.CURE_Cluster_ID) >= 0 ? `Klaster ${row.CURE_Cluster_ID}` : 'Tidak terklaster'}
                            </div>
                            {row.Profil_Klaster && (
                              <div className="text-[10px] text-[#6d826f] max-w-[160px] truncate" title={row.Profil_Klaster}>
                                {String(row.Profil_Klaster).replace(/^Klaster \d+:\s*/, '')}
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-2.5 font-medium text-[#1e2920] max-w-[200px] truncate" title={row[labelField]}>
                            {row[labelField]}
                            {showConfidence && row.IndoBERT_Record_Confidence !== null && row.IndoBERT_Record_Confidence !== undefined && (
                              <span className="ml-1 font-mono text-[10px] text-[#6d826f]">({Number(row.IndoBERT_Record_Confidence).toFixed(2)})</span>
                            )}
                          </td>
                          <td className="px-4 py-2.5 whitespace-nowrap text-[#495f4c]">
                            {row.Prioritas_Intervensi}
                          </td>
                          <td className="px-4 py-2.5 text-[#405443] max-w-xs truncate" title={row.Saran_Asupan_Nutrisi}>
                            {row.Saran_Asupan_Nutrisi}
                          </td>
                          <td className="px-4 py-2.5 text-[#405443] max-w-xs truncate" title={row.Tindakan_Medis_Sanitasi}>
                            {row.Tindakan_Medis_Sanitasi}
                          </td>
                          <td className="px-4 py-2.5 text-[#627764] whitespace-nowrap text-[11px]">
                            {row.Jadwal_Monitoring}
                          </td>
                          <td className="px-4 py-2.5 whitespace-nowrap">
                            <button
                              onClick={() => setDetailRecord(row)}
                              className="text-[11px] font-medium text-[#38523c] hover:text-[#1e2920] bg-[#eef4ee] hover:bg-[#e2ebe2] border border-[#d2dfd2] px-2.5 py-1 rounded transition-colors whitespace-nowrap"
                            >
                              Lihat Saran
                            </button>
                          </td>
                          {trailingColumns.map(col => (
                            <td key={col} className="px-4 py-2.5 text-[#4f6452] whitespace-nowrap tabular-nums">
                              {row[col] !== undefined && row[col] !== null ? String(row[col]) : '-'}
                            </td>
                          ))}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>

              {/* Pagination Controls */}
              {filteredRecords.length > 0 && (
                <div className="border-t border-[#edf2ed] bg-[#fafcfa] px-4 sm:px-6 py-3 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#5f7361]">
                  <div>
                    Menampilkan {((currentPage - 1) * pageSize) + 1} &ndash; {Math.min(currentPage * pageSize, filteredRecords.length)} dari {filteredRecords.length.toLocaleString('id-ID')} balita
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="border border-[#cfdacd] bg-white hover:bg-[#f3f7f3] disabled:opacity-40 disabled:cursor-not-allowed rounded p-1 transition-colors"
                      title="Halaman Sebelumnya"
                      aria-label="Halaman Sebelumnya"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="font-mono px-2">
                      Halaman {currentPage} dari {totalPages}
                    </span>
                    <button
                      onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                      disabled={currentPage === totalPages}
                      className="border border-[#cfdacd] bg-white hover:bg-[#f3f7f3] disabled:opacity-40 disabled:cursor-not-allowed rounded p-1 transition-colors"
                      title="Halaman Berikutnya"
                      aria-label="Halaman Berikutnya"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}

            </div>

            {/* Recommendation Detail Modal: items-start + scroll di dalam modal (tidak terpotong di 1366x768 / ponsel) */}
            {detailRecord && (
              <div
                className="fixed inset-0 z-50 bg-[#1e2920]/40 backdrop-blur-xs flex items-start justify-center p-3 sm:p-6"
                onClick={(e) => { if (e.target === e.currentTarget) setDetailRecord(null); }}
              >
                <div
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="detail-title"
                  className="max-w-2xl w-full max-h-[calc(100dvh-1.5rem)] sm:max-h-[calc(100dvh-3rem)] overflow-y-auto bg-white border border-[#d8e3d8] rounded-xl p-5 sm:p-7 space-y-6 shadow-xl"
                >

                  {/* Modal Header */}
                  <div className="flex items-start justify-between gap-3 border-b border-[#edf2ed] pb-4">
                    <div className="space-y-1 min-w-0">
                      <div className="text-[11px] font-semibold text-[#58725c] tracking-wider uppercase">
                        Rekomendasi & Intervensi Gizi Individual
                      </div>
                      <h3 id="detail-title" className="text-lg font-bold text-[#1b261d]">
                        Data Balita &middot; Lembar {currentSheet.sheetName}, Baris Excel {detailRecord.Baris_Excel ?? '-'}
                      </h3>
                      <div className="text-xs text-[#627764] flex flex-wrap gap-x-2 gap-y-1">
                        {detailRecord.Usia_Bulan_Dihitung !== undefined && detailRecord.Usia_Bulan_Dihitung !== null && (
                          <span className="font-semibold text-[#344836]">Usia: {fmt(Number(detailRecord.Usia_Bulan_Dihitung), 1)} Bulan</span>
                        )}
                        {pickRaw(detailRecord, 'gender') !== undefined && <span>&middot; JK: {String(pickRaw(detailRecord, 'gender'))}</span>}
                        {roleValue(detailRecord, 'height') !== null && <span>&middot; TB: {roleValue(detailRecord, 'height')} cm</span>}
                        {roleValue(detailRecord, 'weight') !== null && <span>&middot; BB: {roleValue(detailRecord, 'weight')} kg</span>}
                        {(detailRecord.Pukesmas || detailRecord.Puskesmas) && <span>&middot; Puskesmas: {detailRecord.Pukesmas || detailRecord.Puskesmas}</span>}
                        {detailRecord.Kec && <span>&middot; Kec. {detailRecord.Kec}</span>}
                        {detailRecord['Desa/Kel'] && <span>&middot; Kel. {detailRecord['Desa/Kel']}</span>}
                      </div>
                    </div>

                    <button
                      onClick={() => setDetailRecord(null)}
                      aria-label="Tutup"
                      className="text-[#6d826f] hover:text-[#1e2920] p-1 border border-[#d2ded2] hover:border-[#b4c7b4] rounded-md transition-colors shrink-0"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Anthropometric Scores */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Klaster CURE:</span>
                      <span className="font-bold text-[#2e4231] font-mono">Klaster {detailRecord.CURE_Cluster_ID}</span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score TB/U (HAZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">{fmt(roleValue(detailRecord, 'haz'))} SD</span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score BB/U (WAZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">{fmt(roleValue(detailRecord, 'waz'))} SD</span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score BB/TB (WHZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">{fmt(roleValue(detailRecord, 'whz'))} SD</span>
                    </div>
                  </div>

                  {/* Diagnosis & Priority */}
                  <div className="p-3.5 border border-[#dbe6db] rounded-lg bg-[#f7faf7] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                    <div className="space-y-1.5">
                      <div>
                        <div className="text-[#647766]">{labelTitle}:</div>
                        <div className="font-semibold text-[#1e2920] mt-0.5">
                          {detailRecord[labelField]}
                          {showConfidence && detailRecord.IndoBERT_Record_Confidence != null && ` (keyakinan ${Number(detailRecord.IndoBERT_Record_Confidence).toFixed(2)})`}
                        </div>
                      </div>
                      {labelField !== RULE_LABEL_FIELD && (
                        <div>
                          <div className="text-[#647766]">{RULE_LABEL_TITLE}:</div>
                          <div className="font-medium text-[#1e2920] mt-0.5">{detailRecord[RULE_LABEL_FIELD]}</div>
                        </div>
                      )}
                      <div className="text-[#647766]">Profil klaster: <span className="text-[#1e2920]">{detailRecord.Profil_Klaster}</span></div>
                    </div>
                    <div className="sm:text-right">
                      <div className="text-[#647766]">Tingkat Prioritas:</div>
                      <div className="text-[#435e47] font-semibold mt-0.5">{detailRecord.Prioritas_Intervensi}</div>
                    </div>
                  </div>

                  {/* Food / Nutrition Advice */}
                  <div className="space-y-1.5">
                    <div className="text-xs font-semibold text-[#324534] flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-[#58725c]"></span>
                      <span>Saran Asupan Nutrisi & Porsi Menu Harian</span>
                    </div>
                    <div className="p-4 border border-[#e2ebe2] rounded-lg bg-[#fafcfa] text-xs text-[#2b3c2e] leading-relaxed">
                      <p>{detailRecord.Saran_Asupan_Nutrisi}</p>
                    </div>
                  </div>

                  {/* Medical & Sanitation Actions */}
                  <div className="space-y-1.5">
                    <div className="text-xs font-semibold text-[#324534] flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-[#7a957d]"></span>
                      <span>Tindakan Medis, Sanitasi & Pencegahan Infeksi</span>
                    </div>
                    <div className="p-4 border border-[#e2ebe2] rounded-lg bg-[#fafcfa] text-xs text-[#2b3c2e] leading-relaxed">
                      <p>{detailRecord.Tindakan_Medis_Sanitasi}</p>
                    </div>
                  </div>

                  {/* Schedule */}
                  <div className="p-3 border border-[#e2ebe2] rounded-lg bg-[#f7faf7] flex flex-wrap items-center justify-between gap-2 text-xs">
                    <span className="text-[#647766]">Jadwal Pemantauan:</span>
                    <span className="text-[#1e2920] font-medium">{detailRecord.Jadwal_Monitoring}</span>
                  </div>

                  <p className="text-[11px] text-[#7d907f]">
                    Panduan gizi umum berbasis kategori Permenkes No. 2/2020, bukan diagnosis atau resep medis perorangan.
                  </p>

                  {/* Modal Footer */}
                  <div className="pt-2 flex justify-end border-t border-[#edf2ed]">
                    <button
                      onClick={() => setDetailRecord(null)}
                      className="bg-[#58725c] hover:bg-[#49604d] text-white text-xs font-medium px-4 py-2 rounded-md transition-colors"
                    >
                      Tutup
                    </button>
                  </div>

                </div>
              </div>
            )}

          </div>
        )}

      </main>

      {/* Clean Footer */}
      <footer className="border-t border-[#e2eae2] bg-white py-5 text-center text-xs text-[#6e8270] mt-auto">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>CURE-IndoBERT for Stunting &middot; Sistem Analisis Klasterisasi &amp; Intervensi Gizi Balita</div>
          <div className="text-[11px] text-[#869988]">Standar Antropometri Kemenkes RI (Permenkes No 2/2020) &amp; WHO 2006</div>
        </div>
      </footer>

    </div>
  );
}
