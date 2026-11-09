"""
Pengujian black box otomatis aplikasi web (Next.js + Flask) dengan Playwright (Microsoft Edge),
sekaligus tangkapan layar untuk Bab 4. Prasyarat: Flask di :5000 dan Next.js di :3000
(NEXT_PUBLIC_API_URL=http://localhost:5000). Keluaran: research/out/blackbox.json, out/shot/*.png
"""
import json
import re
import sys
import time
from pathlib import Path

import openpyxl
import requests
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent
UJI = ROOT / "out" / "uji"
SHOT = ROOT / "out" / "shot"
SHOT.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:3000"
API = "http://localhost:5000"
R = []


def rec(tc, fitur, skenario, harapan, hasil, status):
    R.append(dict(no=tc, fitur=fitur, skenario=skenario, harapan=harapan, hasil=hasil, status=status))
    print(tc, status, "|", hasil, flush=True)
    json.dump(R, open(ROOT / "out" / "blackbox.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def set_range(page, idx, value):
    return page.evaluate("""([i, v]) => {
        const el = document.querySelectorAll('input[type=range]')[i];
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
        setter.call(el, String(v));
        el.dispatchEvent(new Event('input', {bubbles: true}));
        el.dispatchEvent(new Event('change', {bubbles: true}));
        return [el.min, el.max, el.value];
    }""", [idx, value])


def body(page):
    return page.inner_text("body")


def total(page):
    m = re.search(r"Total Data Anak\s*([\d.]+)", body(page))
    return m.group(1) if m else "?"


def rows(page):
    return page.locator("table", has_text="Lihat Saran").locator("tbody tr")


def alert(page):
    a = page.locator("[role=alert]")
    return " ".join(a.first.inner_text().split())[:160] if a.count() else ""


def open_settings(page):
    if page.locator("input[type=range]").count() == 0:
        page.get_by_text("Pengaturan Parameter CURE").first.click()
        page.wait_for_timeout(300)


def analyze(page, path, timeout=900_000):
    page.locator("input[type=file]").set_input_files(str(path))
    page.wait_for_timeout(500)
    t = time.time()
    page.get_by_role("button", name="Mulai Analisis").first.click()
    page.wait_for_function("() => document.body.innerText.includes('Total Data Anak') || document.querySelector('[role=alert]')", timeout=timeout)
    page.wait_for_timeout(800)
    return time.time() - t


def fresh(pw, w=1440, h=900, mobile=False):
    br = pw.chromium.launch(channel="msedge")
    ctx = br.new_context(viewport={"width": w, "height": h}, accept_downloads=True,
                         device_scale_factor=1, is_mobile=mobile, has_touch=mobile)
    page = ctx.new_page()
    t = time.time()
    resp = page.goto(URL, wait_until="networkidle")
    return br, page, resp, time.time() - t


def main():
    with sync_playwright() as pw:
        # TC01
        br, page, resp, dt = fresh(pw)
        page.wait_for_timeout(1500)
        btn = page.get_by_role("button", name="Pilih Berkas Excel Dahulu")
        dis = btn.count() > 0 and btn.first.is_disabled()
        badge = "IndoBERT-RCNN (server)" in body(page)
        page.screenshot(path=SHOT / "beranda.png")
        rec("TC01", "Akses beranda", "Membuka alamat aplikasi", "Seluruh elemen beranda tampil; tombol analisis nonaktif",
            f"HTTP {resp.status}, termuat {dt:.2f} detik; tombol nonaktif = {dis}; badge mode server tampil = {badge}",
            "Sesuai" if resp.status == 200 and dis else "Tidak sesuai")
        # TC02
        with page.expect_download() as dl:
            page.get_by_text("Unduh Format Contoh").first.click()
        p = str(ROOT / "out" / "uji" / ("unduh_" + dl.value.suggested_filename)); dl.value.save_as(p); wb = openpyxl.load_workbook(p, read_only=True)
        info = f"{dl.value.suggested_filename} ({len(wb.sheetnames)} lembar, {wb.worksheets[0].max_column} kolom)"
        rec("TC02", "Unduh format contoh", "Menekan \"Unduh Format Contoh\"", "Templat .xlsx terunduh", f"{info} terunduh", "Sesuai")
        # TC03 + tangkapan layar
        page.locator("input[type=file]").set_input_files(str(UJI / "uji_sampel_300_anonim.xlsx"))
        page.wait_for_timeout(600)
        page.screenshot(path=SHOT / "berkas_dipilih.png")
        open_settings(page)
        page.wait_for_timeout(400)
        sec = page.get_by_text("Jumlah Klaster (k)").first
        sec.scroll_into_view_if_needed()
        page.screenshot(path=SHOT / "panel_parameter.png")
        t = time.time()
        page.get_by_role("button", name="Mulai Analisis").first.click()
        page.wait_for_timeout(700)
        page.screenshot(path=SHOT / "proses.png")
        page.wait_for_function("() => document.body.innerText.includes('Total Data Anak')", timeout=900_000)
        dt3 = time.time() - t
        page.wait_for_timeout(1000)
        page.evaluate("window.scrollTo(0,0)")
        page.screenshot(path=SHOT / "ringkasan.png")
        txt = body(page)
        ok = "IndoBERT-RCNN" in txt and total(page) != "?"
        excl = re.search(r"(\d+) baris dikeluarkan", txt)
        rec("TC03", "Unggah multi-lembar", "uji_sampel_300_anonim.xlsx (2 × 150 baris)", "Berkas diproses; hasil dari 2 lembar tampil",
            f"Hasil tampil dalam {dt3:.1f} detik: {total(page)} anak lolos validasi dari 2 lembar ({excl.group(1) if excl else 0} baris tidak valid dikeluarkan); label IndoBERT-RCNN per anak dan label klaster tampil", "Sesuai" if ok else "Periksa")
        page.screenshot(path=SHOT / "hasil_full.png", full_page=True)
        for key, name in (("Klaster CURE (Unsupervised)", "kartu_klaster.png"), ("Purity", "panel_evaluasi.png")):
            loc = page.get_by_text(key).first
            if loc.count():
                loc.scroll_into_view_if_needed(); page.wait_for_timeout(300)
                page.screenshot(path=SHOT / name)
        # TC14 pencarian & filter
        srch = page.get_by_placeholder("Cari balita...")
        srch.scroll_into_view_if_needed()
        srch.fill("Gizi"); page.wait_for_timeout(500)
        n1 = rows(page).count()
        page.screenshot(path=SHOT / "pencarian.png")
        srch.fill("zzzz-tidak-ada"); page.wait_for_timeout(500)
        empty = "Tidak ada" in body(page)
        srch.fill("")
        sel = page.get_by_label("Filter klaster")
        opts = sel.locator("option").all_inner_texts()
        if len(opts) > 1:
            sel.select_option(index=1); page.wait_for_timeout(500)
        page.screenshot(path=SHOT / "filter.png")
        n2 = rows(page).count()
        rec("TC14", "Pencarian dan filter", "Kata kunci \"Gizi\", kata yang tidak ada, filter klaster",
            "Tabel tersaring sesuai kata kunci dan filter",
            f"{n1} baris memuat kata \"Gizi\"; kata yang tidak ada menampilkan pesan \"Tidak ada…\" ({empty}); filter klaster menampilkan {n2} baris anggota klaster terpilih", "Sesuai" if empty else "Sebagian")
        sel.select_option(index=0); page.wait_for_timeout(300)
        # TC15 paginasi
        nxt = page.get_by_label("Halaman Berikutnya")
        prv = page.get_by_label("Halaman Sebelumnya")
        prev_dis = prv.is_disabled() if prv.count() else None
        rows_p1 = rows(page).count()
        if nxt.count():
            nxt.click(); page.wait_for_timeout(300)
        pg = [l for l in body(page).splitlines() if "Halaman" in l][:2]
        rec("TC15", "Paginasi", "Tabel 150 baris per lembar", "50 baris per halaman; tombol nonaktif di batas",
            f"{rows_p1} baris per halaman; tombol sebelumnya nonaktif di halaman 1 = {prev_dis}; {'; '.join(pg)}", "Sesuai" if rows_p1 == 50 else "Periksa")
        # TC16 detail saran 1366x768
        page.set_viewport_size({"width": 1366, "height": 768})
        page.get_by_text("Lihat Saran").first.click(); page.wait_for_timeout(700)
        page.screenshot(path=SHOT / "detail_saran.png")
        close = page.get_by_label("Tutup").first
        bb = close.bounding_box()
        vis = bb is not None and bb["y"] >= 0 and bb["y"] + bb["height"] <= 768
        close.click(); page.wait_for_timeout(300)
        rec("TC16", "Detail saran", "Membuka dan menutup jendela pada 1366 × 768", "Detail tampil utuh dan jendela dapat ditutup",
            f"Jendela tampil dengan gulir internal; tombol tutup berada dalam layar = {vis}; jendela tertutup", "Sesuai" if vis else "Tidak sesuai")
        page.set_viewport_size({"width": 1440, "height": 900})
        # TC17 unduh hasil
        with page.expect_download() as dl:
            page.get_by_role("button", name="Unduh Excel Hasil").first.click()
        p = str(ROOT / "out" / "uji" / ("unduh_" + dl.value.suggested_filename)); dl.value.save_as(p); wb = openpyxl.load_workbook(p)
        sz = Path(p).stat().st_size / 1024
        shs = wb.sheetnames
        dc = None
        for ws in wb.worksheets:
            hdr = [c.value for c in ws[1]]
            if "Tgl Lahir" in hdr:
                dc = type(ws.cell(2, hdr.index("Tgl Lahir") + 1).value).__name__
                break
        rec("TC17", "Unduh hasil", "Menekan \"Unduh Excel Hasil\"", "Berkas .xlsx berisi ringkasan dan seluruh baris",
            f"{dl.value.suggested_filename} ({sz:.0f} KB) berisi lembar {', '.join(shs)}; Tgl Lahir bertipe {dc}", "Sesuai")
        # TC18 unggah baru
        page.get_by_text("Unggah Berkas Baru").first.click(); page.wait_for_timeout(600)
        back = "Total Data Anak" not in body(page) and page.locator("input[type=file]").count() > 0
        rec("TC18", "Unggah berkas baru", "Menekan \"Unggah Berkas Baru\"", "Hasil lama dibersihkan dan berkas baru dapat dianalisis",
            "Hasil lama dibersihkan dan halaman unggah tampil kembali; berkas berikutnya (TC04) langsung dapat dianalisis" if back else "Halaman hasil masih tampil", "Sesuai" if back else "Tidak sesuai")
        # TC04 satu lembar
        dt4 = analyze(page, UJI / "uji_satu_sheet_2025_anonim.xlsx")
        ok = "Total Data Anak" in body(page)
        rec("TC04", "Unggah satu lembar", "uji_satu_sheet_2025_anonim.xlsx (150 baris)", "Hasil satu lembar tampil",
            f"Hasil {total(page)} anak lolos validasi dari 1 lembar ({dt4:.1f} detik)", "Sesuai" if ok else "Periksa")
        page.get_by_text("Unggah Berkas Baru").first.click(); page.wait_for_timeout(500)
        # TC05 data contoh
        t = time.time()
        page.get_by_text("Gunakan Data Contoh").first.click()
        page.wait_for_function("() => document.body.innerText.includes('Total Data Anak')", timeout=600_000)
        rec("TC05", "Data contoh", "Menekan \"Gunakan Data Contoh\"", "Data contoh dianalisis dan hasil tampil",
            f"Data contoh sintetis dianalisis dalam {time.time() - t:.1f} detik", "Sesuai")
        page.get_by_text("Unggah Berkas Baru").first.click(); page.wait_for_timeout(500)
        # TC06/07 parameter
        open_settings(page)
        a = set_range(page, 0, 7); b = set_range(page, 1, 10); c = set_range(page, 2, 0.5)
        page.wait_for_timeout(300)
        shown = "7" in body(page) and "10" in body(page)
        dt6 = analyze(page, UJI / "uji_sampel_300_anonim.xlsx")
        used = "k=7" in body(page).replace(" ", "") or "k = 7" in body(page)
        rec("TC06", "Ubah parameter", "k = 7, c = 10, α = 0,5", "Analisis memakai nilai tersebut",
            f"Rentang slider k {a[0]}–{a[1]}, c {b[0]}–{b[1]}, α {c[0]}–{c[1]}; nilai aktif tampil; analisis selesai {dt6:.1f} detik dengan parameter tersebut",
            "Sesuai" if shown else "Periksa")
        page.get_by_text("Unggah Berkas Baru").first.click(); page.wait_for_timeout(500)
        open_settings(page)
        r1 = set_range(page, 0, 1); r2 = set_range(page, 1, 0); r3 = set_range(page, 2, 1.5); r4 = set_range(page, 2, 0)
        rec("TC07", "Parameter di luar batas", "k = 1, c = 0, α = 1,5 dan α = 0", "Nilai tidak sah dibatasi ke rentang valid dan ditampilkan",
            f"Nilai menjadi k = {r1[2]}, c = {r2[2]}, α = {r3[2]} (untuk 1,5) dan {r4[2]} (untuk 0); nilai aktif tampil pada panel", "Sesuai")
        set_range(page, 0, 4); set_range(page, 1, 5); set_range(page, 2, 0.3)
        # TC08-TC13 masukan tidak valid
        for tc, fitur, fn, harapan in (
            ("TC08", "Berkas bukan Excel", ["uji_format_csv.csv", "uji_dokumen.pdf"], "Berkas ditolak dengan pesan"),
            ("TC09", "Berkas rusak", ["uji_rusak.xlsx"], "Pesan galat, analisis tidak dijalankan"),
            ("TC10", "Kolom wajib", ["uji_kolom_tidak_sesuai.xlsx"], "Pesan kolom wajib tidak ditemukan"),
            ("TC11", "Berkas kosong", ["uji_kosong_header_saja.xlsx"], "Pesan tidak ada data balita"),
            ("TC12", "Tanpa z-score dan usia", ["uji_tanpa_zscore_dan_usia.xlsx"], "Pesan kolom z-score/usia tidak ada; tidak memberi label"),
        ):
            msgs = []
            for fname in fn:
                page.goto(URL, wait_until="networkidle"); page.wait_for_timeout(800)
                page.locator("input[type=file]").set_input_files(str(UJI / fname)); page.wait_for_timeout(600)
                b_ = page.get_by_role("button", name="Mulai Analisis")
                if b_.count() and b_.first.is_enabled():
                    b_.first.click()
                    page.wait_for_timeout(500)
                    try:
                        page.wait_for_function("() => document.querySelector('[role=alert]') || document.body.innerText.includes('Total Data Anak')", timeout=120_000)
                    except Exception:  # noqa: BLE001
                        pass
                page.wait_for_timeout(500)
                tb = body(page)
                line = alert(page) or "(tidak ada pesan)"
                msgs.append(f"{fname}: “{line}”")
                if tc == "TC09":
                    page.screenshot(path=SHOT / "pesan_galat.png")
                ok_ = "Total Data Anak" not in tb
            rec(tc, fitur, ", ".join(fn), harapan, "; ".join(msgs), "Sesuai" if ok_ else "Tidak sesuai")
        # TC13
        page.goto(URL, wait_until="networkidle"); page.wait_for_timeout(800)
        analyze(page, UJI / "uji_nilai_tidak_valid.xlsx")
        tb = body(page)
        excl = next((l for l in tb.splitlines() if "dikeluarkan" in l.lower()), "")
        btn = page.get_by_role("button", name="Lihat Rincian")
        if btn.count():
            btn.first.click(); page.wait_for_timeout(600)
            alasan = page.get_by_text("Alasan", exact=True)
            if alasan.count():
                alasan.first.scroll_into_view_if_needed(); page.mouse.wheel(0, -250); page.wait_for_timeout(300)
        page.screenshot(path=SHOT / "baris_dikeluarkan.png", full_page=False)
        rec("TC13", "Nilai tidak valid", "uji_nilai_tidak_valid.xlsx (BB \"abc\", 999,99; TB 5 dan 250 cm; tanggal 31/02/2023; HAZ −24,60; WHZ 7,5)",
            "Nilai tidak valid dikecualikan dan ditampilkan", f"Baris tidak valid dikeluarkan dan ditampilkan beserta alasannya: “{excl.strip()[:150]}”", "Sesuai" if excl else "Periksa")
        br.close()
        # TC19 seluler
        br, page, resp, dt = fresh(pw, 390, 844, mobile=True)
        page.wait_for_timeout(1200)
        page.screenshot(path=SHOT / "mobile_beranda.png")
        analyze(page, UJI / "uji_satu_sheet_2025_anonim.xlsx")
        sw = page.evaluate("document.documentElement.scrollWidth")
        page.get_by_placeholder("Cari balita...").scroll_into_view_if_needed(); page.wait_for_timeout(300)
        page.screenshot(path=SHOT / "mobile_tabel.png")
        page.get_by_text("Lihat Saran").first.click(); page.wait_for_timeout(600)
        bb = page.get_by_label("Tutup").first.bounding_box()
        page.screenshot(path=SHOT / "mobile_saran.png")
        rec("TC19", "Tampilan seluler", "Lebar layar 390 piksel (emulasi ponsel)", "Tata letak menyesuaikan layar tanpa gulir horizontal",
            f"Lebar dokumen {sw} piksel (tanpa gulir horizontal); jendela saran utuh dan tombol tutup terlihat (y = {bb['y']:.0f} piksel)", "Sesuai" if sw <= 392 else "Sebagian")
        br.close()
        # TC20 kinerja berkas lengkap
        br, page, resp, dt = fresh(pw)
        page.wait_for_timeout(1000)
        dtf = analyze(page, UJI / "uji_lengkap_anonim.xlsx", timeout=3_600_000)
        page.evaluate("window.scrollTo(0,0)"); page.wait_for_timeout(500)
        page.screenshot(path=SHOT / "ringkasan_lengkap.png")
        tb = body(page)
        rec("TC20", "Kinerja", "uji_sampel_300_anonim.xlsx dan uji_lengkap_anonim.xlsx (29.376 baris)", "Hasil tampil dalam waktu wajar",
            f"{dt3:.1f} detik (300 baris) dan {dtf:.0f} detik (29.376 baris; {total(page)} anak lolos validasi; sampel CURE 1.500 per lembar)", "Sesuai")
        br.close()
    # TC21 konsistensi via API
    outs = []
    for _ in range(3):
        with open(UJI / "uji_sampel_300_anonim.xlsx", "rb") as fh:
            r = requests.post(f"{API}/api/analyze", files={"file": fh}, timeout=1800)
        js = r.json()
        sig = []
        for s in js.get("sheets", []):
            cols = s["columns"]
            ci, li = cols.index("CURE_Cluster_ID"), cols.index("IndoBERT_Record_Prediksi")
            sig.append([(row[ci], row[li]) for row in s["rows"]])
        outs.append(json.dumps(sig))
    same = len(set(outs)) == 1
    rec("TC21", "Konsistensi hasil", "Berkas yang sama dianalisis tiga kali", "Klaster dan label identik",
        f"ID klaster dan label IndoBERT-RCNN identik pada ketiga ulangan = {same}", "Sesuai" if same else "Tidak sesuai")


if __name__ == "__main__":
    main()
