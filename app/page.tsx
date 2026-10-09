'use client';

import React, { useState, useRef, useMemo } from 'react';
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
  Sparkles
} from 'lucide-react';
import {
  PipelineResult,
  ChildRecord,
  processWorkbook,
  createBenchmarkWorkbook,
  exportResultToExcel
} from '@/lib/stunting-pipeline';

export default function StuntSensePage() {
  // Hyperparameters (CURE)
  const [nClusters, setNClusters] = useState<number>(4);
  const [nRepresentatives, setNRepresentatives] = useState<number>(5);
  const [shrinkFactor, setShrinkFactor] = useState<number>(0.3);

  // File & Processing State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [processingMessage, setProcessingMessage] = useState<string>('Menyiapkan pemrosesan data...');
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [activeSheetIndex, setActiveSheetIndex] = useState<number>(0);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [filterCluster, setFilterCluster] = useState<string>('ALL');
  const [filterLabel, setFilterLabel] = useState<string>('ALL');
  const [currentPage, setCurrentPage] = useState<number>(1);
  const pageSize = 50;

  // Selected Record for Detail Modal
  const [detailRecord, setDetailRecord] = useState<ChildRecord | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
        setSelectedFile(file);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const runPipelineOnFile = async (file: File) => {
    setIsProcessing(true);
    setProcessingProgress(15);
    setProcessingMessage(`Membaca berkas ${file.name}...`);

    try {
      const arrayBuffer = await file.arrayBuffer();
      setProcessingProgress(45);
      setProcessingMessage('Menganalisis lembar kerja antropometri e-PPGBM / Puskesmas...');
      await new Promise(r => setTimeout(r, 150));

      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      setProcessingProgress(70);
      setProcessingMessage('Menjalankan klasterisasi spasial CURE dan perumusan rekomendasi gizi...');
      await new Promise(r => setTimeout(r, 150));

      const result = await processWorkbook(workbook, file.name, {
        k: nClusters,
        c: nRepresentatives,
        alpha: shrinkFactor
      });

      setProcessingProgress(100);
      setProcessingMessage('Selesai!');
      await new Promise(r => setTimeout(r, 150));

      setPipelineResult(result);
      setActiveSheetIndex(0);
      setCurrentPage(1);
    } catch (err) {
      console.error(err);
      alert('Gagal memproses file Excel. Pastikan file dalam format .xlsx atau .xls yang valid.');
    } finally {
      setIsProcessing(false);
    }
  };

  const runBenchmarkDemo = async () => {
    setIsProcessing(true);
    setProcessingProgress(25);
    setProcessingMessage('Membuat data benchmark antropometri balita...');

    try {
      await new Promise(r => setTimeout(r, 150));
      setProcessingProgress(60);
      const wb = createBenchmarkWorkbook();

      setProcessingProgress(85);
      setProcessingMessage('Mengelompokkan klaster CURE dan merumuskan intervensi...');
      await new Promise(r => setTimeout(r, 150));

      const result = await processWorkbook(wb, 'Format_Contoh_Data_Balita.xlsx', {
        k: nClusters,
        c: nRepresentatives,
        alpha: shrinkFactor
      });

      setProcessingProgress(100);
      await new Promise(r => setTimeout(r, 150));

      setPipelineResult(result);
      setActiveSheetIndex(0);
      setCurrentPage(1);
    } catch (err) {
      console.error(err);
      alert('Gagal memuat data contoh.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadSampleBenchmark = () => {
    const wb = createBenchmarkWorkbook();
    XLSX.writeFile(wb, 'Format_Contoh_Data_Balita.xlsx');
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
    a.download = `Hasil_Analisis_${pipelineResult.originalFilename}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const currentSheet = pipelineResult?.sheets[activeSheetIndex];

  // Filtering records
  const filteredRecords = useMemo(() => {
    if (!currentSheet) return [];
    return currentSheet.records.filter(r => {
      const matchesSearch = searchQuery === '' || Object.values(r).some(val =>
        String(val).toLowerCase().includes(searchQuery.toLowerCase())
      );
      const matchesCluster = filterCluster === 'ALL' || String(r.Cluster_ID) === filterCluster;
      const matchesLabel = filterLabel === 'ALL' || r.Auto_Label === filterLabel;
      return matchesSearch && matchesCluster && matchesLabel;
    });
  }, [currentSheet, searchQuery, filterCluster, filterLabel]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredRecords.length / pageSize));
  const paginatedRecords = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredRecords.slice(start, start + pageSize);
  }, [filteredRecords, currentPage, pageSize]);

  return (
    <div className="min-h-screen flex flex-col bg-[#f9fbf9] text-[#223124] font-sans antialiased">
      
      {/* Top Header */}
      <header className="border-b border-[#e2eae2] bg-white/90 backdrop-blur sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          
          {/* Brand */}
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-[#58725c]"></span>
            <span className="text-base font-semibold tracking-tight text-[#1e2920]">
              StuntSense
            </span>
          </div>

          <div className="hidden sm:block text-xs text-[#5c6e5e]">
            Sistem Pemantauan Pertumbuhan & Gizi Balita &middot; Permenkes No 2/2020
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
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
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-10 space-y-8">
        
        {/* Hero Section */}
        <section className="space-y-2 max-w-2xl">
          <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[#1b261d]">
            Analisis Data & Rekomendasi Gizi Balita
          </h1>
          <p className="text-xs sm:text-sm text-[#556958] leading-relaxed">
            Pengelompokan status stunting menggunakan algoritma CURE (*Clustering Using Representatives*) dan perumusan menu nutrisi serta tindakan medis individual sesuai standar antropometri WHO dan Kemenkes RI.
          </p>
        </section>

        {/* Upload & Configuration Card (When no result is loaded) */}
        {!pipelineResult && (
          <div className="bg-white border border-[#e2ebe2] rounded-xl p-6 sm:p-8 space-y-6 shadow-xs max-w-3xl">
            
            <div className="border-b border-[#edf2ed] pb-4 flex items-center justify-between">
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
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-all cursor-pointer ${
                selectedFile
                  ? 'border-[#58725c] bg-[#f4f8f4]'
                  : 'border-[#ccd9cc] hover:border-[#6e8a72] bg-[#fafcfa] hover:bg-[#f3f7f3]'
              } space-y-3`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept=".xlsx,.xls"
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
                  Mendukung kolom Nama, JK, Tgl Lahir/Usia, BB, TB, serta Z-Score (ZS BB/U, ZS TB/U, ZS BB/TB)
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

            {/* CURE Parameters with Explanations */}
            <div className="pt-2 border-t border-[#edf2ed] space-y-3.5">
              <div className="flex items-center justify-between text-xs text-[#415344]">
                <div>
                  <span className="font-semibold text-[#1e2920]">Konfigurasi Algoritma CURE</span>
                  <span className="text-[11px] text-[#6a7d6c] ml-2 hidden sm:inline">(Clustering Using Representatives)</span>
                </div>
                <span className="font-mono text-[#58725c] bg-[#eef4ee] px-2 py-0.5 rounded text-[11px]">
                  k={nClusters} &middot; c={nRepresentatives} &middot; &alpha;={shrinkFactor}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Parameter k */}
                <div className="space-y-1.5 p-3 rounded-lg bg-[#fafcfa] border border-[#e5ebe5]">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-[#2d3e30]">Jumlah Klaster (k)</span>
                    <span className="font-mono font-semibold text-[#58725c]">{nClusters}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="6"
                    value={nClusters}
                    onChange={(e) => setNClusters(parseInt(e.target.value))}
                    className="w-full accent-[#58725c] cursor-pointer"
                  />
                  <div className="text-[10px] text-[#6b7c6d] leading-snug">
                    Jumlah kelompok kategori stunting yang dibentuk (WHO / Kemenkes).
                  </div>
                </div>

                {/* Parameter c (Titik Perwakilan) */}
                <div className="space-y-1.5 p-3 rounded-lg bg-[#fafcfa] border border-[#e5ebe5]">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-[#2d3e30]">Titik Perwakilan (c)</span>
                    <span className="font-mono font-semibold text-[#58725c]">{nRepresentatives}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="8"
                    value={nRepresentatives}
                    onChange={(e) => setNRepresentatives(parseInt(e.target.value))}
                    className="w-full accent-[#58725c] cursor-pointer"
                  />
                  <div className="text-[10px] text-[#6b7c6d] leading-snug">
                    Jumlah sampel per klaster untuk memodelkan batas sebaran non-lingkaran.
                  </div>
                </div>

                {/* Parameter alpha (Faktor Penyusutan) */}
                <div className="space-y-1.5 p-3 rounded-lg bg-[#fafcfa] border border-[#e5ebe5]">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-medium text-[#2d3e30]">Faktor Penyusutan (&alpha;)</span>
                    <span className="font-mono font-semibold text-[#58725c]">{shrinkFactor}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.6"
                    step="0.05"
                    value={shrinkFactor}
                    onChange={(e) => setShrinkFactor(parseFloat(e.target.value))}
                    className="w-full accent-[#58725c] cursor-pointer"
                  />
                  <div className="text-[10px] text-[#6b7c6d] leading-snug">
                    Menarik titik perwakilan ke centroid guna meredam pengaruh data outlier.
                  </div>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
              <button
                disabled={!selectedFile || isProcessing}
                onClick={() => selectedFile && runPipelineOnFile(selectedFile)}
                className="w-full sm:w-auto flex-1 bg-[#58725c] hover:bg-[#49604d] disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2 px-5 rounded-md text-xs transition-colors shadow-xs"
              >
                {selectedFile ? `Mulai Analisis Berkas (${selectedFile.name})` : 'Pilih Berkas Excel Dahulu'}
              </button>
              <button
                onClick={runBenchmarkDemo}
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
            </div>
          </div>
        )}

        {/* RESULTS DASHBOARD */}
        {pipelineResult && (
          <div className="space-y-6">
            
            {/* Header Result Action */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#e2ebe2] pb-5">
              <div>
                <div className="text-xs text-[#687c69] font-medium">
                  Berkas: {pipelineResult.originalFilename}
                </div>
                <h2 className="text-xl sm:text-2xl font-semibold tracking-tight text-[#1b261d] mt-0.5">
                  Hasil Analisis Status Gizi & Rekomendasi
                </h2>
              </div>
              
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setPipelineResult(null)}
                  className="text-xs font-medium text-[#465b49] hover:text-[#1e2920] border border-[#d2ded2] hover:border-[#b4c7b4] bg-white px-3.5 py-1.5 rounded-md transition-colors"
                >
                  Unggah Berkas Baru
                </button>
                <button
                  onClick={downloadEnrichedWorkbook}
                  className="bg-[#58725c] hover:bg-[#49604d] text-white text-xs font-medium px-4 py-1.5 rounded-md transition-colors flex items-center gap-1.5 shadow-sm"
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
                <div className="text-[11px] text-[#7d907f] mt-0.5">Seluruh lembar kerja</div>
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
                  c={pipelineResult.cureParams.c} &middot; &alpha;={pipelineResult.cureParams.alpha}
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

            {/* Summary Distributions */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Cluster Distribution */}
              <div className="bg-white border border-[#e2ebe2] rounded-lg p-5 space-y-3 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#edf2ed] pb-2.5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#3d5140]">
                    Pengelompokan Klaster CURE
                  </h3>
                  <span className="text-xs font-mono text-[#7d907f]">
                    k={pipelineResult.cureParams.k}
                  </span>
                </div>
                <div className="space-y-2.5">
                  {Object.entries(pipelineResult.globalClusterDistribution).map(([cid, count]) => {
                    const pct = Math.round((count / pipelineResult.totalRows) * 1000) / 10;
                    return (
                      <div key={cid} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-[#2b3c2e] font-medium">{cid}</span>
                          <span className="font-mono text-[#5f7361] tabular-nums">
                            {count.toLocaleString('id-ID')} anak ({pct}%)
                          </span>
                        </div>
                        <div className="w-full bg-[#ebf2eb] rounded-full h-1.5 overflow-hidden">
                          <div className="bg-[#58725c] h-1.5 rounded-full" style={{ width: `${pct}%` }}></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Status Distribution */}
              <div className="bg-white border border-[#e2ebe2] rounded-lg p-5 space-y-3 shadow-xs">
                <div className="flex items-center justify-between border-b border-[#edf2ed] pb-2.5">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-[#3d5140]">
                    Kategori Status Gizi (WHO & Kemenkes)
                  </h3>
                  <span className="text-xs font-mono text-[#7d907f]">Permenkes RI</span>
                </div>
                <div className="space-y-2.5">
                  {Object.entries(pipelineResult.globalLabelDistribution).map(([label, count]) => {
                    const pct = Math.round((count / pipelineResult.totalRows) * 1000) / 10;
                    return (
                      <div key={label} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-[#2b3c2e] font-medium truncate max-w-[280px]" title={label}>{label}</span>
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

            {/* Table Viewport */}
            <div className="bg-white border border-[#e2ebe2] rounded-lg overflow-hidden shadow-xs">
              
              {/* Sheet Switcher & Search Bar */}
              <div className="border-b border-[#edf2ed] bg-[#fafcfa] px-6 py-3.5 flex flex-col md:flex-row md:items-center justify-between gap-3">
                
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
                    onChange={(e) => {
                      setFilterCluster(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#324534] focus:outline-none focus:border-[#58725c]"
                  >
                    <option value="ALL">Semua Klaster</option>
                    {currentSheet?.clusterProfiles?.map((cp) => (
                      <option key={cp.clusterId} value={String(cp.clusterId)}>
                        {cp.clusterName} ({cp.count})
                      </option>
                    )) || Array.from({ length: pipelineResult.cureParams.k }).map((_, i) => (
                      <option key={i} value={String(i)}>Klaster {i}</option>
                    ))}
                  </select>

                  <select
                    value={filterLabel}
                    onChange={(e) => {
                      setFilterLabel(e.target.value);
                      setCurrentPage(1);
                    }}
                    className="bg-white border border-[#cfdacd] rounded-md px-2.5 py-1 text-xs text-[#324534] focus:outline-none focus:border-[#58725c]"
                  >
                    <option value="ALL">Semua Status Gizi</option>
                    {currentSheet && Object.keys(currentSheet.labelDistribution).map((lbl) => (
                      <option key={lbl} value={lbl}>
                        {lbl.split('(')[0].trim()} ({currentSheet.labelDistribution[lbl]})
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
                      <th className="px-4 py-2.5 whitespace-nowrap">Status Gizi</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Prioritas</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Saran Asupan Nutrisi</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Tindakan Medis & Sanitasi</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Jadwal Pantau</th>
                      <th className="px-4 py-2.5 whitespace-nowrap">Rincian</th>
                      {currentSheet?.columns
                        .filter(col => ![
                          'Cluster_ID', 'Cluster_Name', 'Auto_Label', 'Label_Confidence', 'Clinical_Priority',
                          'Saran_Asupan_Nutrisi', 'Tindakan_Medis_Sanitasi', 'Jadwal_Monitoring'
                        ].includes(col))
                        .map(col => (
                          <th key={col} className="px-4 py-2.5 whitespace-nowrap font-normal text-[#647766]">{col}</th>
                        ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#f2f6f2] bg-white">
                    {paginatedRecords.length === 0 ? (
                      <tr>
                        <td colSpan={15} className="px-4 py-8 text-center text-xs text-[#7d907f]">
                          Tidak ada data balita yang cocok dengan filter pencarian.
                        </td>
                      </tr>
                    ) : (
                      paginatedRecords.map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-[#f8faf8] transition-colors">
                          <td className="px-4 py-2.5 whitespace-nowrap">
                            <div className="font-mono text-[#435e47] font-semibold">Klaster {row.Cluster_ID}</div>
                            {row.Cluster_Name && (
                              <div className="text-[10px] text-[#6d826f] max-w-[130px] truncate" title={row.Cluster_Name}>
                                {row.Cluster_Name.replace(/^Klaster \d+\s*/, '').replace(/[()]/g, '')}
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-2.5 font-medium text-[#1e2920] max-w-[170px] truncate" title={row.Auto_Label}>
                            {row.Auto_Label}
                          </td>
                          <td className="px-4 py-2.5 whitespace-nowrap text-[#495f4c]">
                            {row.Clinical_Priority}
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
                          {currentSheet?.columns
                            .filter(col => ![
                              'Cluster_ID', 'Cluster_Name', 'Auto_Label', 'Label_Confidence', 'Clinical_Priority',
                              'Saran_Asupan_Nutrisi', 'Tindakan_Medis_Sanitasi', 'Jadwal_Monitoring'
                            ].includes(col))
                            .map(col => (
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
                <div className="border-t border-[#edf2ed] bg-[#fafcfa] px-6 py-3 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-[#5f7361]">
                  <div>
                    Menampilkan {((currentPage - 1) * pageSize) + 1} &ndash; {Math.min(currentPage * pageSize, filteredRecords.length)} dari {filteredRecords.length.toLocaleString('id-ID')} balita
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                      disabled={currentPage === 1}
                      className="border border-[#cfdacd] bg-white hover:bg-[#f3f7f3] disabled:opacity-40 disabled:cursor-not-allowed rounded p-1 transition-colors"
                      title="Halaman Sebelumnya"
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
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              )}

            </div>

            {/* Recommendation Detail Modal */}
            {detailRecord && (
              <div className="fixed inset-0 z-50 bg-[#1e2920]/40 backdrop-blur-xs flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
                <div className="max-w-2xl w-full bg-white border border-[#d8e3d8] rounded-xl p-6 sm:p-7 space-y-6 shadow-xl my-8">
                  
                  {/* Modal Header */}
                  <div className="flex items-start justify-between border-b border-[#edf2ed] pb-4">
                    <div className="space-y-1">
                      <div className="text-[11px] font-semibold text-[#58725c] tracking-wider uppercase">
                        Rekomendasi & Intervensi Gizi Individual
                      </div>
                      <h3 className="text-lg font-bold text-[#1b261d]">
                        {detailRecord.Nama || detailRecord.Nama_Balita || detailRecord.Nama_Lengkap || detailRecord.Nama_Anak || 'Data Balita'}
                      </h3>
                      <div className="text-xs text-[#627764] flex flex-wrap gap-2">
                        {detailRecord.Usia_Bulan !== undefined && detailRecord.Usia_Bulan !== null && (
                          <span className="font-semibold text-[#344836]">Usia: {detailRecord.Usia_Bulan} Bulan</span>
                        )}
                        {detailRecord.JK && (
                          <span>&middot; JK: {detailRecord.JK === 'L' ? 'Laki-laki (L)' : detailRecord.JK === 'P' ? 'Perempuan (P)' : detailRecord.JK}</span>
                        )}
                        {(detailRecord.Tinggi || detailRecord.TB) && <span>&middot; TB: {detailRecord.Tinggi || detailRecord.TB} cm</span>}
                        {(detailRecord.Berat || detailRecord.BB) && <span>&middot; BB: {detailRecord.Berat || detailRecord.BB} kg</span>}
                        {(detailRecord.Pukesmas || detailRecord.Puskesmas) && <span>&middot; Puskesmas: {detailRecord.Pukesmas || detailRecord.Puskesmas}</span>}
                        {detailRecord.Kec && <span>&middot; Kec. {detailRecord.Kec}</span>}
                        {detailRecord['Desa/Kel'] && <span>&middot; Kel. {detailRecord['Desa/Kel']}</span>}
                      </div>
                    </div>

                    <button
                      onClick={() => setDetailRecord(null)}
                      className="text-[#6d826f] hover:text-[#1e2920] p-1 border border-[#d2ded2] hover:border-[#b4c7b4] rounded-md transition-colors"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Anthropometric Scores */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Klaster CURE:</span>
                      <span className="font-bold text-[#2e4231] font-mono">Klaster {detailRecord.Cluster_ID}</span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score TB/U (HAZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">
                        {detailRecord.Calculated_HAZ !== undefined && detailRecord.Calculated_HAZ !== null ? `${detailRecord.Calculated_HAZ} SD` : (detailRecord['ZS TB/U'] ? `${detailRecord['ZS TB/U']} SD` : '-')}
                      </span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score BB/U (WAZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">
                        {detailRecord.Calculated_WAZ !== undefined && detailRecord.Calculated_WAZ !== null ? `${detailRecord.Calculated_WAZ} SD` : (detailRecord['ZS BB/U'] ? `${detailRecord['ZS BB/U']} SD` : '-')}
                      </span>
                    </div>
                    <div className="bg-[#f7faf7] border border-[#e2ece2] rounded p-2.5 text-xs">
                      <span className="text-[#667a68] block text-[11px]">Z-Score BB/TB (WHZ):</span>
                      <span className="font-bold text-[#2e4231] font-mono">
                        {detailRecord.Calculated_WHZ !== undefined && detailRecord.Calculated_WHZ !== null ? `${detailRecord.Calculated_WHZ} SD` : (detailRecord['ZS BB/TB'] ? `${detailRecord['ZS BB/TB']} SD` : '-')}
                      </span>
                    </div>
                  </div>

                  {/* Diagnosis & Priority */}
                  <div className="p-3.5 border border-[#dbe6db] rounded-lg bg-[#f7faf7] flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                    <div>
                      <div className="text-[#647766]">Status Gizi Teridentifikasi:</div>
                      <div className="font-semibold text-[#1e2920] mt-0.5">{detailRecord.Auto_Label}</div>
                    </div>
                    <div className="sm:text-right">
                      <div className="text-[#647766]">Tingkat Prioritas:</div>
                      <div className="text-[#435e47] font-semibold mt-0.5">{detailRecord.Clinical_Priority}</div>
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
                  <div className="p-3 border border-[#e2ebe2] rounded-lg bg-[#f7faf7] flex items-center justify-between text-xs">
                    <span className="text-[#647766]">Jadwal Pemantauan:</span>
                    <span className="text-[#1e2920] font-medium">{detailRecord.Jadwal_Monitoring}</span>
                  </div>

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
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>StuntSense &middot; Sistem Pemantauan Pertumbuhan & Gizi Balita</div>
          <div className="text-[11px] text-[#869988]">Standar Antropometri Kemenkes RI (Permenkes No 2/2020)</div>
        </div>
      </footer>

    </div>
  );
}
