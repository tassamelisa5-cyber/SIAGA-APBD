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
    if _kosong(x):
        return "–"
    if x == 0:
        return "Rp0"
    if abs(x) < 1e9:
        return f"Rp{angka(x / 1e6)} juta"
    return f"Rp{angka(x / 1e9)} miliar"


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

    skor, risiko, alasan, jenis = [], [], [], []
    for _, r in d.iterrows():
        s, why, jn = 0, [], []
        g = r["Gap_Waktu"]
        if r["Anggaran"] == 0 and r["Realisasi"] > 0:
            # Arah "berlebih" 1: ada realisasi padahal tidak dianggarkan -> langsung Tinggi
            s, why, jn = 5, [f"ada realisasi {rp(r['Realisasi'])} padahal anggaran Rp0"], ["Tanpa anggaran"]
        elif r["Kelompok"] == "Belanja" and r["Anggaran"] > 0 and r["Real%"] > 100:
            # Arah "berlebih" 2: belanja melebihi pagu -> langsung Tinggi
            s, jn = 5, ["Melebihi pagu"]
            why = [f"belanja melebihi pagu ({pct(r['Real%'])} dari anggaran)"]
        elif r["Kelompok"] in ("Pendapatan", "Belanja") and r["Anggaran"] > 0:
            # Arah "lambat": tertinggal dari target waktu
            if g <= -2 * ambang_waktu:
                s += 2; why.append(f"tertinggal {abs(g):.0f} poin dari target waktu"); jn.append("Tertinggal")
            elif g <= -ambang_waktu:
                s += 1; why.append(f"tertinggal {abs(g):.0f} poin dari target waktu"); jn.append("Tertinggal")
            # Arah "berlebih" 3: belanja jauh lebih cepat dari target waktu (masih <= 100%)
            if r["Kelompok"] == "Belanja" and g >= 2 * ambang_waktu:
                s += 1; why.append(f"lebih cepat {g:.0f} poin dari target waktu (tidak lazim)")
                jn.append("Terlalu cepat")
            if pd.notna(r["Gap_Pembanding"]) and r["Gap_Pembanding"] <= -15:
                s += 1; why.append(f"{abs(r['Gap_Pembanding']):.0f} poin di bawah rata-rata se-provinsi")
            if pd.notna(r["Gap_TahunLalu"]) and r["Gap_TahunLalu"] <= -15:
                s += 1; why.append(f"{abs(r['Gap_TahunLalu']):.0f} poin di bawah capaian tahun lalu")
            if s > 0 and pd.notna(r["Porsi%"]) and r["Porsi%"] >= 10 and r["Level"] > 1:
                s += 1; why.append(f"nilainya besar ({r['Porsi%']:.0f}% dari total {r['Kelompok'].lower()})")
            # Catatan informasi (tanpa skor): pendapatan jauh melampaui target
            if r["Kelompok"] == "Pendapatan" and r["Real%"] > 120:
                why.append(f"info: pendapatan {pct(r['Real%'])} dari target, mungkin target disusun terlalu rendah")
                jn.append("Info")
        skor.append(s)
        risiko.append("Tinggi" if s >= 3 else "Sedang" if s == 2 else "Rendah")
        alasan.append("; ".join(why))
        jenis.append(", ".join(jn))
    d["Skor"], d["Risiko"], d["Alasan"], d["Jenis"] = skor, risiko, alasan, jenis
    return d


# ---------------------------------------------------------------------------
# 1b. Gabungan lintas wilayah, anomali, dan pencarian (dipakai tab Anomali & alat AI)
# ---------------------------------------------------------------------------
JENIS_ANOMALI = ["Tanpa anggaran", "Melebihi pagu", "Terlalu cepat", "Info"]


def gabung_hasil(hasil: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{wilayah: hasil analisis_pemda} -> satu tabel panjang dengan kolom Wilayah."""
    if not hasil:
        return pd.DataFrame()
    return pd.concat([h.assign(Wilayah=w) for w, h in hasil.items()], ignore_index=True)


def cari_anomali(semua: pd.DataFrame) -> pd.DataFrame:
    """Akun dengan arah 'berlebih': tanpa anggaran, melebihi pagu, terlalu cepat, info pendapatan."""
    if semua.empty:
        return semua
    m = semua["Jenis"].apply(lambda j: any(x in j for x in JENIS_ANOMALI))
    a = semua[m].copy()
    urut = {"Tanpa anggaran": 0, "Melebihi pagu": 1, "Terlalu cepat": 2, "Info": 3}
    a["_u"] = a["Jenis"].apply(lambda j: min(urut[x] for x in urut if x in j))
    return a.sort_values(["_u", "Realisasi"], ascending=[True, False]).drop(columns="_u")


KONDISI = {
    "anggaran_nol_ada_realisasi": lambda d: (d["Anggaran"] == 0) & (d["Realisasi"] > 0),
    "melebihi_pagu": lambda d: (d["Kelompok"] == "Belanja") & (d["Real%"] > 100),
    "terlalu_cepat": lambda d: d["Jenis"].str.contains("Terlalu cepat"),
    "tertinggal": lambda d: d["Jenis"].str.contains("Tertinggal"),
    "risiko_tinggi": lambda d: d["Risiko"] == "Tinggi",
    "semua": lambda d: pd.Series(True, index=d.index),
}


def ringkasan_lintas_wilayah(semua: pd.DataFrame, fokus: str = "", maks_anomali: int = 15,
                             per_wilayah: int = 3) -> str:
    """Teks ringkas untuk AI: anomali lintas wilayah + akun risiko tinggi teratas tiap wilayah."""
    if semua.empty:
        return ""
    L = ["\n## Anomali arah 'berlebih' (semua wilayah yang diambil)"]
    a = cari_anomali(semua)
    if a.empty:
        L.append("- Tidak ada.")
    for _, r in a.head(maks_anomali).iterrows():
        L.append(f"- {r['Wilayah']} | {r['Akun']} [{r['Kelompok']}] | {r['Jenis']} | {r['Alasan']}")
    if len(a) > maks_anomali:
        L.append(f"- ... dan {len(a) - maks_anomali} lainnya (gunakan alat cari_data untuk melihat semua)")
    L.append(f"\n## Akun risiko tinggi teratas per wilayah (maks {per_wilayah})")
    t = semua[(semua["Risiko"] == "Tinggi") & (semua["Level"] > 1)]
    for w, g in t.groupby("Wilayah", sort=False):
        if fokus and w == fokus:
            continue
        isi = "; ".join(f"{r['Akun']} {pct(r['Real%'])}" for _, r in
                        g.sort_values(["Skor", "Sisa"], ascending=False).head(per_wilayah).iterrows())
        L.append(f"- {w}: {isi}")
    return "\n".join(L)


def cari_data(semua: pd.DataFrame, kondisi: str = "semua", wilayah: str = "", akun: str = "",
              kelompok: str = "", urutkan: str = "", maks: int = 40) -> str:
    """Pencarian data untuk AI. Mengembalikan tabel teks ringkas."""
    if semua.empty:
        return "Tidak ada data."
    d = semua[semua["Level"] > 0]
    d = d[KONDISI.get(kondisi, KONDISI["semua"])(d)]
    if wilayah:
        d = d[d["Wilayah"].str.contains(wilayah, case=False, regex=False)]
    if akun:
        d = d[d["Akun"].str.contains(akun, case=False, regex=False)]
    if kelompok:
        d = d[d["Kelompok"].str.contains(kelompok, case=False, regex=False, na=False)]
    if urutkan == "realisasi_terendah":
        d = d.sort_values("Real%")
    elif urutkan == "realisasi_tertinggi":
        d = d.sort_values("Real%", ascending=False)
    elif urutkan == "sisa_terbesar":
        d = d.sort_values("Sisa", ascending=False)
    n = len(d)
    if n == 0:
        return "Tidak ditemukan data yang cocok dengan kriteria tersebut."
    baris = [f"Ditemukan {n} baris" + (f" (ditampilkan {maks} pertama)" if n > maks else "") + ":"]
    for _, r in d.head(maks).iterrows():
        baris.append(f"- {r['Wilayah']} | {r['Akun']} [{r['Kelompok']}] | anggaran {rp(r['Anggaran'])} | "
                     f"realisasi {rp(r['Realisasi'])} ({pct(r['Real%'])}) | risiko {r['Risiko']} | "
                     f"{r['Alasan'] or '-'}")
    return "\n".join(baris)


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


# ---------------------------------------------------------------------------
# 4. Rasio keuangan & kepatuhan belanja wajib (UU 1/2022 tentang HKPD)
# ---------------------------------------------------------------------------
BATAS_PEGAWAI = 30.0      # Pasal 146: belanja pegawai paling tinggi 30% dari total belanja APBD
BATAS_INFRA = 40.0        # Pasal 147: belanja infrastruktur pelayanan publik paling rendah 40%


def _tabel_akun(df: pd.DataFrame) -> pd.DataFrame:
    d = bersihkan(df)
    return d.drop_duplicates("Akun", keep="first").set_index("Akun")


def _bagi(a, b):
    return a / b * 100 if b and not _kosong(a) and not _kosong(b) else np.nan


def _komponen(t: pd.DataFrame, kolom: str) -> dict:
    g = lambda akun: float(t.loc[akun, kolom]) if akun in t.index else 0.0
    belanja = g("Belanja Daerah")
    transfer = g("Belanja Bagi Hasil") + g("Belanja Bantuan Keuangan")
    return {
        "kemandirian": _bagi(g("PAD"), g("Pendapatan Daerah")),
        "ketergantungan": _bagi(g("TKDD"), g("Pendapatan Daerah")),
        "pegawai": _bagi(g("Belanja Pegawai"), belanja),
        "modal": _bagi(g("Belanja Modal"), belanja - transfer),
        "surplus": _bagi(g("Pendapatan Daerah") - belanja, g("Pendapatan Daerah")),
    }


RASIO_INFO = {
    "kemandirian": ("Kemandirian fiskal", "PAD ÷ Pendapatan Daerah",
                    "Seberapa besar pendapatan daerah berasal dari usaha sendiri (pajak, retribusi, laba BUMD). "
                    "Makin tinggi makin mandiri."),
    "ketergantungan": ("Ketergantungan transfer pusat", "TKDD ÷ Pendapatan Daerah",
                       "Seberapa besar pendapatan bergantung pada dana dari pemerintah pusat. Makin tinggi, "
                       "makin rentan bila penyaluran dari pusat terlambat atau dipotong."),
    "pegawai": ("Porsi belanja pegawai", "Belanja Pegawai ÷ Belanja Daerah",
                "UU HKPD Pasal 146 membatasi belanja pegawai paling tinggi 30% dari total belanja APBD "
                "(di luar tunjangan guru yang dibiayai TKD), wajib dipenuhi paling lambat 5 tahun sejak UU "
                "diundangkan (2027). Data DJPK belum memisahkan tunjangan guru, sehingga angka ini adalah indikasi."),
    "modal": ("Porsi belanja modal (proksi infrastruktur)",
              "Belanja Modal ÷ (Belanja Daerah − Bagi Hasil − Bantuan Keuangan)",
              "UU HKPD Pasal 147 mewajibkan belanja infrastruktur pelayanan publik paling rendah 40% "
              "(di luar bagi hasil dan transfer ke daerah/desa). Belanja modal dipakai sebagai pendekatan kasar, "
              "karena tidak semua belanja modal adalah infrastruktur dan sebagian infrastruktur dicatat di akun lain."),
    "surplus": ("Surplus (+) / defisit (−)", "(Pendapatan − Belanja) ÷ Pendapatan",
                "Selisih pendapatan dan belanja. Defisit APBD ditutup dari pembiayaan (misalnya SiLPA tahun lalu). "
                "Surplus realisasi yang besar di akhir tahun menandakan banyak anggaran tidak terserap."),
}


def _status(kunci, nilai):
    if _kosong(nilai):
        return "Tidak ada data"
    if kunci == "pegawai":
        return "Melebihi batas" if nilai > BATAS_PEGAWAI else "Sesuai batas"
    if kunci == "modal":
        return "Di bawah batas" if nilai < BATAS_INFRA else "Sesuai batas"
    return "Informasi"


def rasio_keuangan(df: pd.DataFrame, df_pembanding: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rasio dihitung dari APBD (anggaran) dan dari realisasi s.d. bulan berjalan."""
    t = _tabel_akun(df)
    ang, real = _komponen(t, "Anggaran"), _komponen(t, "Realisasi")
    pb = _komponen(_tabel_akun(df_pembanding), "Anggaran") if df_pembanding is not None else {}
    baris = []
    for k, (nama, rumus, arti) in RASIO_INFO.items():
        acuan = (f"Maks. {BATAS_PEGAWAI:.0f}%" if k == "pegawai" else
                 f"Min. {BATAS_INFRA:.0f}%" if k == "modal" else "-")
        baris.append({"Kunci": k, "Rasio": nama, "Rumus": rumus, "APBD %": ang[k], "Realisasi %": real[k],
                      "Se-provinsi %": pb.get(k, np.nan), "Acuan": acuan, "Status": _status(k, ang[k]),
                      "Penjelasan": arti})
    return pd.DataFrame(baris)


def rasio_lintas(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rasio (berbasis APBD) untuk banyak wilayah sekaligus."""
    baris = []
    for nama, df in data.items():
        try:
            k = _komponen(_tabel_akun(df), "Anggaran")
            r = _komponen(_tabel_akun(df), "Realisasi")
        except Exception:
            continue
        baris.append({"Wilayah": nama, "Kemandirian fiskal %": k["kemandirian"],
                      "Ketergantungan transfer %": k["ketergantungan"], "Belanja pegawai %": k["pegawai"],
                      "Belanja modal (proksi infrastruktur) %": k["modal"],
                      "Realisasi belanja %": _bagi(*_realisasi_belanja(df)),
                      "Status pegawai (maks 30%)": _status("pegawai", k["pegawai"]),
                      "Status modal (min 40%)": _status("modal", k["modal"])})
    return pd.DataFrame(baris)


def _realisasi_belanja(df):
    t = _tabel_akun(df)
    if "Belanja Daerah" not in t.index:
        return np.nan, np.nan
    return t.loc["Belanja Daerah", "Realisasi"], t.loc["Belanja Daerah", "Anggaran"]


def ringkasan_rasio_ai(r: pd.DataFrame) -> str:
    L = ["\n## Rasio keuangan & kepatuhan belanja wajib (dihitung dari APBD; realisasi dalam kurung)"]
    for _, x in r.iterrows():
        L.append(f"- {x['Rasio']}: {pct(x['APBD %'])} (realisasi {pct(x['Realisasi %'])}; "
                 f"se-provinsi {pct(x['Se-provinsi %'])}) | acuan {x['Acuan']} | {x['Status']}")
    L.append("- Catatan: batas pegawai 30% (UU HKPD Ps.146) berlaku penuh 2027 dan sedang dibahas perpanjangan "
             "transisinya; porsi belanja modal hanya proksi kasar belanja infrastruktur (Ps.147, min 40%).")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5. Tren bulanan & proyeksi akhir tahun
# ---------------------------------------------------------------------------
AKUN_TREN = ["Pendapatan Daerah", "PAD", "TKDD", "Belanja Daerah", "Belanja Pegawai",
             "Belanja Barang dan Jasa", "Belanja Modal", "Belanja Lainnya"]


def susun_tren(bulanan: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """bulanan = {bulan: df_ekspor}. Hasil: tabel panjang Bulan, Akun, Anggaran, Realisasi, Real%."""
    baris = []
    for b, df in sorted(bulanan.items()):
        t = _tabel_akun(df)
        for a in AKUN_TREN:
            if a in t.index:
                baris.append({"Bulan": b, "Akun": a, "Anggaran": t.loc[a, "Anggaran"],
                              "Realisasi": t.loc[a, "Realisasi"], "Real%": t.loc[a, "Real%"]})
    return pd.DataFrame(baris, columns=["Bulan", "Akun", "Anggaran", "Realisasi", "Real%"])


def proyeksi_akhir_tahun(tren_ini: pd.DataFrame, tren_lalu: pd.DataFrame, akun: str, periode: int) -> dict | None:
    """Perkiraan realisasi Desember bila pola sisa tahun sama dengan tahun lalu.
    faktor = realisasi Des tahun lalu ÷ realisasi bulan yang sama tahun lalu."""
    a = tren_ini[(tren_ini["Akun"] == akun) & (tren_ini["Bulan"] == periode)]
    lalu = tren_lalu[tren_lalu["Akun"] == akun].set_index("Bulan")["Real%"] if not tren_lalu.empty else pd.Series()
    if a.empty or periode >= 12 or periode not in lalu.index or 12 not in lalu.index or not lalu[periode]:
        return None
    sekarang = float(a["Real%"].iloc[0])
    faktor = float(lalu[12]) / float(lalu[periode])
    perkiraan = sekarang * faktor
    if akun.startswith("Belanja"):
        perkiraan = min(perkiraan, 100.0)
    anggaran = float(a["Anggaran"].iloc[0])
    desember_lalu = float(lalu[12]) - float(lalu[11]) if 11 in lalu.index else np.nan
    return {"Akun": akun, "Sekarang %": sekarang, "Perkiraan Desember %": perkiraan,
            "Tidak terserap (Rp)": anggaran * (1 - perkiraan / 100) if perkiraan < 100 else 0.0,
            "Tahun lalu s.d. bulan ini %": float(lalu[periode]), "Tahun lalu Desember %": float(lalu[12]),
            "Lonjakan Desember tahun lalu (poin)": desember_lalu}


def ringkasan_tren_ai(proyeksi: list[dict]) -> str:
    if not proyeksi:
        return ""
    L = ["\n## Proyeksi akhir tahun (jika pola sisa tahun sama dengan tahun lalu)"]
    for p in proyeksi:
        L.append(f"- {p['Akun']}: sekarang {pct(p['Sekarang %'])}, perkiraan Desember {pct(p['Perkiraan Desember %'])}, "
                 f"potensi tidak terserap {rp(p['Tidak terserap (Rp)'])}; tahun lalu naik "
                 f"{pct(p['Lonjakan Desember tahun lalu (poin)'])} poin di bulan Desember saja")
    return "\n".join(L)
