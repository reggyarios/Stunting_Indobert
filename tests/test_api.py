"""Validasi /api/analyze dengan Flask test client (tanpa checkpoint)."""

import io
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import app as webapp  # noqa: E402


def _xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _children(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "No": range(1, n + 1),
        "Nama": [f"Anak {i}" for i in range(n)],
        "JK": ["L", "P"] * (n // 2),
        "Tgl Lahir": pd.date_range("2022-01-01", periods=n, freq="20D"),
        "Berat": [9.0 + 0.3 * i for i in range(n)],
        "Tinggi": [75.0 + 0.8 * i for i in range(n)],
        "ZS BB/U": [-2.5 + 0.35 * i for i in range(n)],
        "ZS TB/U": [-3.2 + 0.4 * i for i in range(n)],
        "ZS BB/TB": [-1.5 + 0.25 * i for i in range(n)],
    })


@mock.patch.object(webapp, "get_classifier", lambda: None)
class TestAnalyzeApiValidation(unittest.TestCase):
    def setUp(self):
        self.client = webapp.app.test_client()

    def post(self, data: bytes, name: str = "data.xlsx", **form):
        return self.client.post("/api/analyze", data={"file": (io.BytesIO(data), name), **form},
                                content_type="multipart/form-data")

    def test_corrupt_or_random_binary_rejected(self):
        for blob in (os.urandom(4096), b"PK\x03\x04" + os.urandom(4096)):
            r = self.post(blob)
            self.assertEqual(r.status_code, 400)
            self.assertIn("Excel", r.get_json()["message"])

    def test_non_excel_extension_rejected(self):
        r = self.post(b"JK,Berat\nL,10\n", name="data.csv")
        self.assertEqual(r.status_code, 400)

    def test_missing_required_columns_rejected(self):
        r = self.post(_xlsx(_children().drop(columns=["ZS TB/U", "Berat"])))
        self.assertEqual(r.status_code, 400)
        body = r.get_json()
        self.assertIn("Kolom wajib tidak ditemukan", body["skipped_sheets"][0]["reason"])
        self.assertIn("ZS TB/U", body["skipped_sheets"][0]["reason"])

    def test_header_only_workbook_has_no_children(self):
        r = self.post(_xlsx(_children().iloc[0:0]))
        self.assertEqual(r.status_code, 400)
        self.assertIn("Tidak ada data balita", r.get_json()["message"])

    def test_valid_upload_excludes_flagged_rows_and_drops_names(self):
        df = _children()
        df.loc[3, "ZS TB/U"] = -24.6   # WHO flag
        df.loc[5, "Berat"] = 115.9     # berat tidak wajar
        r = self.post(_xlsx(df), reference_date="2024-08-01", sample_size="50")
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True)[:500])
        body = r.get_json()
        sheet = body["sheets"][0]
        self.assertEqual([e["row"] for e in sheet["excluded_rows"]], [5, 7])  # baris Excel (header = baris 1)
        self.assertEqual(len(sheet["rows"]), 10)
        self.assertNotIn("Nama", sheet["columns"])
        self.assertFalse(body["indobert_active"])
        pred = sheet["rows"][0][sheet["columns"].index("IndoBERT_Record_Prediksi")]
        self.assertIn("Belum Dilatih", pred)
        self.assertEqual(r.headers.get("Access-Control-Allow-Origin"), "*")


if __name__ == "__main__":
    unittest.main()
