import type {Metadata} from 'next';
import './globals.css'; // Global styles

export const metadata: Metadata = {
  title: 'StuntSense CURE & IndoBERT Pipeline',
  description: 'Production platform for automated text and data labeling combined with CURE clustering on child stunting datasets sourced from Excel files.',
  openGraph: {
    title: 'StuntSense CURE & IndoBERT Pipeline',
    description: 'Production platform for automated text and data labeling combined with CURE clustering on child stunting datasets sourced from Excel files.',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'StuntSense CURE & IndoBERT Pipeline',
    description: 'Production platform for automated text and data labeling combined with CURE clustering on child stunting datasets sourced from Excel files.',
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
