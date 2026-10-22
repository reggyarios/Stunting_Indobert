import type {Metadata} from 'next';
import './globals.css'; // Global styles

export const metadata: Metadata = {
  title: 'CURE-IndoBERT for Stunting - Klasterisasi & Klasifikasi Pertumbuhan Balita',
  description: 'Platform riset dan implementasi sistem klasterisasi CURE dan klasifikasi IndoBERT untuk analisis stunting balita serta rekomendasi intervensi gizi berbasis standar Kemenkes RI & WHO.',
  openGraph: {
    title: 'CURE-IndoBERT for Stunting - Klasterisasi & Klasifikasi Pertumbuhan Balita',
    description: 'Platform riset dan implementasi sistem klasterisasi CURE dan klasifikasi IndoBERT untuk analisis stunting balita serta rekomendasi intervensi gizi berbasis standar Kemenkes RI & WHO.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CURE-IndoBERT for Stunting - Klasterisasi & Klasifikasi Pertumbuhan Balita',
    description: 'Platform riset dan implementasi sistem klasterisasi CURE dan klasifikasi IndoBERT untuk analisis stunting balita serta rekomendasi intervensi gizi berbasis standar Kemenkes RI & WHO.',
  },
};

export default function RootLayout({children}: {children: React.ReactNode}) {
  return (
    <html lang="id" className="h-full bg-[#fbfdfb] text-slate-800 antialiased">
      <body className="min-h-full font-sans selection:bg-[#dbe6dc] selection:text-[#253628]" suppressHydrationWarning>
        {children}
      </body>
    </html>
  );
}
