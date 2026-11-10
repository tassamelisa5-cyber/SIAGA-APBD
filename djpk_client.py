"""
djpk_client.py
Modul pengambil data APBD dari Portal Data SIKD DJPK Kemenkeu
(https://djpk.kemenkeu.go.id/portal/data/apbd).

Sumber yang dipakai (ditemukan dari source HTML halaman portal):
  1. /portal/provinsi/{tahun}                -> JSON daftar provinsi
  2. /portal/pemda/{kode_prov}/{tahun}       -> JSON daftar pemda dalam provinsi
  3. /portal/csv_apbd?type=apbd&periode=..&tahun=..&provinsi=..&pemda=..
                                             -> file Excel XML (angka penuh, tidak dibulatkan)
  4. /portal/data/apbd?periode=..            -> halaman HTML (cadangan jika no.3 gagal;
                                                angka dibulatkan ke miliar 2 desimal)

Catatan:
  - Ini bukan API resmi yang terdokumentasi; jika DJPK mengubah situsnya,
    fungsi di sini perlu disesuaikan.
  - Gunakan secukupnya (ada cache di app.py) agar tidak membebani server DJPK.
"""

import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

BASE = "https://djpk.kemenkeu.go.id/portal"
HEADERS = {"User-Agent": "Mozilla/5.0 (tugas-kuliah-teknologi-audit; APBD-agent)"}
TIMEOUT = 30

# Struktur akun APBD (untuk mengembalikan level induk-anak yang hilang di file ekspor)
LEVEL1 = {"Pendapatan Daerah", "Belanja Daerah", "Pembiayaan Daerah"}
LEVEL2 = {
    "PAD", "TKDD", "Pendapatan Lainnya",
    "Belanja Pegawai", "Belanja Barang dan Jasa", "Belanja Modal", "Belanja Lainnya",
    "Penerimaan Pembiayaan Daerah", "Pengeluaran Pembiayaan Daerah",
}


# ---------------------------------------------------------------------------
# 1. Daftar wilayah
# ---------------------------------------------------------------------------
def list_provinsi(tahun: int) -> dict:
    """Kembalikan {kode: nama} provinsi untuk tahun tertentu."""
    r = requests.get(f"{BASE}/provinsi/{tahun}", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def list_pemda(kode_prov: str, tahun: int) -> dict:
    """Kembalikan {kode: nama} pemda dalam satu provinsi (00 = pemprov, -- = semua)."""
    r = requests.get(f"{BASE}/pemda/{kode_prov}/{tahun}", headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 2. Parser
# ---------------------------------------------------------------------------
def _beri_level(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah kolom Level (1/2/3) dan Kelompok (Pendapatan/Belanja/Pembiayaan)."""
    levels, kelompok, induk2 = [], [], []
    klp, prev_name, prev_lvl, cur_l2 = None, None, None, None
    for name in df["Akun"]:
        if name in LEVEL1:
            lvl, klp, cur_l2 = 1, name.split()[0], None
        elif name in LEVEL2 and not (name == prev_name and prev_lvl == 2):
            lvl, cur_l2 = 2, name
        else:
            lvl = 3  # termasuk baris duplikat, mis. "Belanja Pegawai" kedua
        levels.append(lvl)
        kelompok.append(klp)
        induk2.append(cur_l2 if lvl == 3 else None)
        prev_name, prev_lvl = name, lvl
    df["Level"] = levels
    df["Kelompok"] = kelompok
    df["Induk"] = induk2
    return df


def parse_excel_xml(content: bytes | str) -> pd.DataFrame:
    """Parse file ekspor DJPK (format Excel 2003 XML / SpreadsheetML)."""
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    ns = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}
    root = ET.fromstring(content)
    rows = []
    for row in root.iter(f"{{{ns['ss']}}}Row"):
        cells = [(d.text or "").strip() for d in row.iter(f"{{{ns['ss']}}}Data")]
        rows.append(cells)
    if not rows or rows[0][:3] != ["Akun", "Anggaran", "Realisasi"]:
        raise ValueError("Format file tidak dikenali sebagai ekspor APBD DJPK.")
    df = pd.DataFrame(rows[1:], columns=["Akun", "Anggaran", "Realisasi", "Persen_DJPK"][: len(rows[0])])
    for c in ["Anggaran", "Realisasi", "Persen_DJPK"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return _beri_level(df)


def _miliar_ke_rupiah(teks: str) -> float:
    """'3.570,04 M' -> 3_570_040_000_000.0  (M = miliar pada portal DJPK)."""
    t = teks.replace("M", "").replace(".", "").replace(",", ".").strip()
    return float(t) * 1_000_000_000 if t else float("nan")


def parse_html(content: bytes | str) -> pd.DataFrame:
    """Parse halaman HTML portal (cadangan). Angka hanya presisi miliar 2 desimal."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    tbody = soup.select_one("#tbl_data table tbody")
    if tbody is None:
        raise ValueError("Tabel APBD tidak ditemukan di halaman.")
    rows = []
    for tr in tbody.find_all("tr"):
        td = tr.find_all("td")
        if len(td) < 5:
            continue
        rows.append([
            td[1].get_text(strip=True),
            _miliar_ke_rupiah(td[2].get_text(strip=True)),
            _miliar_ke_rupiah(td[3].get_text(strip=True)),
            pd.to_numeric(td[4].get_text(strip=True), errors="coerce"),
        ])
    df = pd.DataFrame(rows, columns=["Akun", "Anggaran", "Realisasi", "Persen_DJPK"])
    return _beri_level(df)


def parse_keterangan_html(content: bytes | str) -> str:
    """Ambil teks keterangan (jenis APBD, batas realisasi, tanggal data diterima)."""
    m = re.search(r'id="keterangan"[^>]*>(.*?)</label>', content if isinstance(content, str)
                  else content.decode("utf-8", "replace"), re.S)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


# ---------------------------------------------------------------------------
# 3. Pengambil data utama
# ---------------------------------------------------------------------------
def fetch_apbd(tahun: int, periode: int, kode_prov: str, kode_pemda: str) -> pd.DataFrame:
    """
    Ambil postur APBD satu pemda.
      periode   : 1-12 (bulan realisasi)
      kode_prov : '04' = Riau
      kode_pemda: '00' = Pemprov, '01' = Kab. Bengkalis, '--' = gabungan semua pemda
    Mencoba ekspor Excel dulu (angka penuh), lalu HTML jika gagal.
    """
    params = dict(periode=periode, tahun=tahun, provinsi=kode_prov, pemda=kode_pemda)
    try:
        r = requests.get(f"{BASE}/csv_apbd", params={"type": "apbd", **params},
                         headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        df = parse_excel_xml(r.content)
        df.attrs["sumber"] = "ekspor Excel DJPK"
    except Exception:
        r = requests.get(f"{BASE}/data/apbd", params=params, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        df = parse_html(r.text)
        df.attrs["sumber"] = "halaman HTML DJPK (dibulatkan ke miliar)"
        df.attrs["keterangan"] = parse_keterangan_html(r.text)
    if df.empty:
        raise ValueError("Data kosong — mungkin pemda belum menyampaikan data untuk periode ini.")
    df.attrs.update(params)
    return df


# ---------------------------------------------------------------------------
# Uji cepat: jalankan  `python djpk_client.py`  di laptop (perlu internet Indonesia)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Daftar pemda Riau 2026:", list_pemda("04", 2026))
    df = fetch_apbd(2026, 11, "04", "01")
    print("Sumber:", df.attrs.get("sumber"))
    print(df.to_string())
