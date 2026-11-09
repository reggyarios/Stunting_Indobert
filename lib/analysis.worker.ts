// Analisis mode browser di Web Worker agar UI tidak membeku (CURE + aturan Permenkes).
import { analyzeWorkbook, AnalysisError, type AnalysisParams } from './stunting-pipeline';

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<{ buffer: ArrayBuffer; filename: string; params: AnalysisParams }>) => void) | null;
  postMessage: (msg: unknown) => void;
};

ctx.onmessage = (e) => {
  const { buffer, filename, params } = e.data;
  try {
    const result = analyzeWorkbook(buffer, filename, params, (pct, message) =>
      ctx.postMessage({ type: 'progress', pct: Math.round(pct), message }));
    ctx.postMessage({ type: 'done', result });
  } catch (err) {
    const known = err instanceof AnalysisError;
    ctx.postMessage({
      type: 'error',
      message: known ? err.message : 'Gagal menganalisis berkas di browser.',
      details: known ? err.details : [String(err)],
    });
  }
};
