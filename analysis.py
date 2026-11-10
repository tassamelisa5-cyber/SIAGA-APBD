"""
analysis.py
Logika analisis SIAGA-APBD (dikerjakan Python, BUKAN oleh AI, agar angka pasti benar).

Isi:
  - analisis_pemda()   : prosedur analitis satu pemda + skor risiko per akun
  - tabel_peringkat()  : membandingkan banyak pemda/provinsi pada akun kunci
  - ringkasan_untuk_ai(): menyusun teks ringkas yang dikirim ke AI (Groq)
"""

import numpy as np
import pandas as pd

AKUN_KUNCI = ["Pendapatan Daerah", "PAD", "Belanja Daerah", "Belanja Barang dan Jasa", "Belanja Modal"]


# ---------------------------------------------------------------------------
# Utilitas
# ---------------------------------------------------------------------------
def _kosong(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def angka(x, desimal=2) -> str:
    """Format angka gaya Indonesia: 1234.5 -> '1.234,50'."""
    if _kosong(x):
        return "–"
    return f"{x:,.{desimal}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def rp(x) -> str:
    """Format rupiah ringkas dalam miliar: 1.234.567.890.000 -> 'Rp1.234,57 miliar'."""
    return "–" if _kosong(x) else f"Rp{angka(x / 1e9)} miliar"


def pct(x) -> str:
    return "–" if _kosong(x) else f"{angka(x, 1)}%"


def selisih_kata(gap) -> str:
    """-41.3 -> 'tertinggal 41,3 poin'; 5 -> 'melampaui 5,0 poin'."""
    if _kosong(gap):
        return "–"
    if abs(gap) < 0.05:
        return "sama"
    return f"{'tertinggal' if gap < 0 else 'melampaui'} {angka(abs(gap), 1)} poin"


def bersihkan(df: pd.DataFrame) -> pd.DataFrame:
    """Buang baris duplikat (anak yang sama persis dengan induknya) & hitung ulang %."""
    d = df[~((df["Level"] == 3) & (df["Akun"] == df["Induk"]))].copy()
    # Anak tunggal yang angkanya identik dengan induk (mis. TKDD = Transfer Pemerintah Pusat) -> buang
    induk = d[d["Level"] == 2].set_index("Akun")[["Anggaran", "Realisasi"]]
    anak = d[d["Level"] == 3]
    tunggal = anak.groupby("Induk").filter(lambda g: len(g) == 1)
    buang = [i for i, r in tunggal.iterrows()
             if r["Induk"] in induk.index
             and np.isclose(r["Anggaran"], induk.loc[r["Induk"], "Anggaran"])
             and np.isclose(r["Realisasi"], induk.loc[r["Induk"], "Realisasi"])]
    d = d.drop(index=buang)
    d["Real%"] = np.where(d["Anggaran"] > 0, d["Realisasi"] / d["Anggaran"] * 100, np.nan)
    d["Sisa"] = d["Anggaran"] - d["Realisasi"]
    return d.reset_index(drop=True)


def _ambil_pct(df: pd.DataFrame | None, nama_kolom: str) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["Akun", nama_kolom])
    d = bersihkan(df)
    return d[["Akun", "Real%"]].rename(columns={"Real%": nama_kolom})


# ---------------------------------------------------------------------------
# 1. Analisis satu pemda
# ---------------------------------------------------------------------------
def analisis_pemda(df_now: pd.DataFrame, periode: int,
                   df_tahun_lalu: pd.DataFrame | None = None,
                   df_pembanding: pd.DataFrame | None = None,
                   ambang_waktu: float = 15.0) -> pd.DataFrame:
    """
    Prosedur analitis per akun:
      - Waktu berjalan (periode/12)          -> apakah realisasi tertinggal dari waktu?
      - Periode sama tahun lalu              -> apakah lebih lambat dari kebiasaan?
      - Pembanding (gabungan se-provinsi)    -> apakah lebih lambat dari daerah lain?
    Menghasilkan skor & tingkat risiko beserta alasannya.
    """
    waktu = periode / 12 * 100
    d = bersihkan(df_now)
    d = d.merge(_ambil_pct(df_tahun_lalu, "Real%_TahunLalu"), on="Akun", how="left")
    d = d.merge(_ambil_pct(df_pembanding, "Real%_Pembanding"), on="Akun", how="left")
    d["Waktu%"] = waktu
    d["Gap_Waktu"] = d["Real%"] - waktu
    d["Gap_TahunLalu"] = d["Real%"] - d["Real%_TahunLalu"]
    d["Gap_Pembanding"] = d["Real%"] - d["Real%_Pembanding"]

    # Porsi anggaran terhadap total kelompoknya (materialitas)
    total = d[d["Level"] == 1].set_index("Kelompok")["Anggaran"].to_dict()
    d["Porsi%"] = d.apply(lambda r: r["Anggaran"] / total.get(r["Kelompok"], np.nan) * 100
                          if total.get(r["Kelompok"]) else np.nan, axis=1)

    skor, risiko, alasan = [], [], []
    for _, r in d.iterrows():
        s, why = 0, []
        tanpa_pagu = (r["Anggaran"] == 0) and (r["Realisasi"] > 0)
        if tanpa_pagu:
            s, why = 5, ["ada realisasi padahal pagu anggaran Rp0"]
        elif r["Kelompok"] in ("Pendapatan", "Belanja") and r["Anggaran"] > 0:
            g = r["Gap_Waktu"]
            if g <= -2 * ambang_waktu:
                s += 2; why.append(f"tertinggal {abs(g):.0f} poin dari target waktu")
            elif g <= -ambang_waktu:
                s += 1; why.append(f"tertinggal {abs(g):.0f} poin dari target waktu")
            if pd.notna(r["Gap_Pembanding"]) and r["Gap_Pembanding"] <= -15:
                s += 1; why.append(f"{abs(r['Gap_Pembanding']):.0f} poin di bawah rata-rata se-provinsi")
            if pd.notna(r["Gap_TahunLalu"]) and r["Gap_TahunLalu"] <= -15:
                s += 1; why.append(f"{abs(r['Gap_TahunLalu']):.0f} poin di bawah capaian tahun lalu")
            if r["Kelompok"] == "Belanja" and r["Real%"] > 100:
                s += 2; why.append("realisasi melebihi pagu")
            if s > 0 and pd.notna(r["Porsi%"]) and r["Porsi%"] >= 10 and r["Level"] > 1:
                s += 1; why.append(f"nilainya besar ({r['Porsi%']:.0f}% dari total {r['Kelompok'].lower()})")
        skor.append(s)
        risiko.append("Tinggi" if s >= 3 else "Sedang" if s == 2 else "Rendah")
        alasan.append("; ".join(why))
    d["Skor"], d["Risiko"], d["Alasan"] = skor, risiko, alasan
    return d


# ---------------------------------------------------------------------------
# 2. Peringkat antar pemda / antar provinsi
# ---------------------------------------------------------------------------
def tabel_peringkat(data: dict[str, pd.DataFrame], periode: int) -> pd.DataFrame:
    """data = {nama_wilayah: df}. Hasil: % realisasi akun kunci per wilayah."""
    baris = []
    for nama, df in data.items():
        d = bersihkan(df).set_index("Akun")
        row = {"Wilayah": nama}
        for a in AKUN_KUNCI:
            row[a] = d.loc[a, "Real%"] if a in d.index else np.nan
        row["Anggaran Belanja"] = d.loc["Belanja Daerah", "Anggaran"] if "Belanja Daerah" in d.index else np.nan
        row["Sisa Belanja"] = d.loc["Belanja Daerah", "Sisa"] if "Belanja Daerah" in d.index else np.nan
        baris.append(row)
    t = pd.DataFrame(baris)
    if t.empty:
        return t
    t["Gap Belanja vs Waktu"] = t["Belanja Daerah"] - periode / 12 * 100
    return t.sort_values("Belanja Daerah").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Ringkasan teks untuk AI
# ---------------------------------------------------------------------------
def ringkasan_untuk_ai(nama: str, tahun: int, periode: int, hasil: pd.DataFrame | None = None,
                       peringkat: pd.DataFrame | None = None, keterangan: str = "") -> str:
    bulan = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
             "Agustus", "September", "Oktober", "November", "Desember"][periode - 1]
    L = [f"Wilayah: {nama}", f"Tahun anggaran: {tahun}, realisasi s.d. {bulan} "
         f"(waktu berjalan {periode / 12 * 100:.1f}%)"]
    if keterangan:
        L.append(f"Keterangan sumber: {keterangan}")
    L.append("Sumber data: Portal Data SIKD DJPK Kemenkeu. Target waktu = persentase bulan yang sudah berjalan.")

    if hasil is not None:
        L.append("\n## Postur APBD (Level 1)")
        for _, r in hasil[hasil["Level"] == 1].iterrows():
            L.append(f"- {r['Akun']}: anggaran {rp(r['Anggaran'])}, realisasi {rp(r['Realisasi'])} "
                     f"({pct(r['Real%'])}), sisa {rp(r['Sisa'])}")
        L.append("\n## Rincian akun (realisasi% | tahun lalu | rata-rata se-provinsi | risiko | alasan)")
        for _, r in hasil[hasil["Level"] > 1].iterrows():
            L.append(f"- {r['Akun']} [{r['Kelompok']}]: anggaran {rp(r['Anggaran'])}, realisasi "
                     f"{rp(r['Realisasi'])} ({pct(r['Real%'])}) | {pct(r['Real%_TahunLalu'])} | "
                     f"{pct(r['Real%_Pembanding'])} | {r['Risiko']} | {r['Alasan'] or '-'}")

    if peringkat is not None and not peringkat.empty:
        L.append("\n## Perbandingan antar wilayah (% realisasi, diurutkan dari belanja terendah)")
        for _, r in peringkat.iterrows():
            L.append(f"- {r['Wilayah']}: Pendapatan {pct(r['Pendapatan Daerah'])}, PAD {pct(r['PAD'])}, "
                     f"Belanja {pct(r['Belanja Daerah'])}, Barang&Jasa {pct(r['Belanja Barang dan Jasa'])}, "
                     f"Modal {pct(r['Belanja Modal'])}, sisa belanja {rp(r['Sisa Belanja'])}")
    return "\n".join(L)
