"""
SIAGA-APBD
AI Agent Pemantauan Realisasi APBD & Catatan Audit Berbasis Risiko
Studi kasus: Kabupaten Bengkalis, Provinsi Riau

Alur agent:
  1. Ambil data   : data APBD diambil otomatis dari Portal Data SIKD DJPK Kemenkeu
                    (atau dari file ekspor yang di-upload sebagai cadangan)
  2. Hitung       : Python menghitung realisasi, membandingkan dengan target waktu,
                    rata-rata se-provinsi, dan tahun lalu, lalu memberi tingkat risiko
  3. Tulis (AI)   : AI (Groq) membaca hasil hitungan dan menulis catatan audit,
                    serta menjawab pertanyaan pengguna

File pendukung: djpk_client.py (pengambil data), analysis.py (perhitungan)
"""

import html
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import importlib

import analysis as an
import djpk_client as djpk

# Muat ulang modul pendukung setiap kali aplikasi berjalan, agar setelah file di GitHub diperbarui
# Streamlit Cloud tidak memakai versi lama yang masih tersimpan di memori.
an = importlib.reload(an)
djpk = importlib.reload(djpk)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIAGA-APBD", layout="wide")

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]

WARNA = {"Tinggi": "#FF6B6B", "Sedang": "#FFB547", "Rendah": "#3DDC97"}
ARTI_RISIKO = {
    "Tinggi": "prioritaskan untuk diperiksa atau dimintai penjelasan",
    "Sedang": "perlu dipantau sampai akhir tahun",
    "Rendah": "masih wajar",
}

# Daftar cadangan jika API daftar wilayah DJPK tidak bisa diakses
PROVINSI_CADANGAN = {
    "01": "Provinsi Aceh", "02": "Provinsi Sumatera Utara", "03": "Provinsi Sumatera Barat",
    "04": "Provinsi Riau", "05": "Provinsi Jambi", "06": "Provinsi Sumatera Selatan",
    "07": "Provinsi Bengkulu", "08": "Provinsi Lampung", "09": "Provinsi DKI Jakarta",
    "10": "Provinsi Jawa Barat", "11": "Provinsi Jawa Tengah", "12": "Provinsi DI Yogyakarta",
    "13": "Provinsi Jawa Timur", "14": "Provinsi Kalimantan Barat", "15": "Provinsi Kalimantan Tengah",
    "16": "Provinsi Kalimantan Selatan", "17": "Provinsi Kalimantan Timur", "18": "Provinsi Sulawesi Utara",
    "19": "Provinsi Sulawesi Tengah", "20": "Provinsi Sulawesi Selatan", "21": "Provinsi Sulawesi Tenggara",
    "22": "Provinsi Bali", "23": "Provinsi Nusa Tenggara Barat", "24": "Provinsi Nusa Tenggara Timur",
    "25": "Provinsi Maluku", "26": "Provinsi Papua", "27": "Provinsi Maluku Utara", "28": "Provinsi Banten",
    "29": "Provinsi Bangka Belitung", "30": "Provinsi Gorontalo", "31": "Provinsi Kepulauan Riau",
    "32": "Provinsi Papua Barat", "33": "Provinsi Sulawesi Barat", "34": "Provinsi Kalimantan Utara",
    "35": "Provinsi Papua Selatan", "36": "Provinsi Papua Tengah", "37": "Provinsi Papua Pegunungan",
    "38": "Provinsi Papua Barat Daya",
}
PEMDA_RIAU_CADANGAN = {
    "00": "Provinsi Riau", "01": "Kab. Bengkalis", "02": "Kab. Indragiri Hilir", "03": "Kab. Indragiri Hulu",
    "04": "Kab. Kampar", "05": "Kab. Kuantan Singingi", "06": "Kab. Pelalawan", "07": "Kab. Rokan Hilir",
    "08": "Kab. Rokan Hulu", "09": "Kab. Siak", "10": "Kota Dumai", "11": "Kota Pekanbaru",
    "12": "Kab. Kepulauan Meranti",
}

# Model Groq (cek berkala: https://console.groq.com/docs/models)
# Catatan: llama-3.1-8b-instant sudah dihentikan untuk akun gratis sejak 16 Agustus 2026.
MODEL_GROQ = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

SYSTEM_MEMO = """Anda adalah auditor internal pemerintah (APIP/Inspektorat) di Indonesia yang ahli
dalam pengelolaan keuangan daerah. Tugas Anda menyusun CATATAN PROSEDUR ANALITIS atas realisasi APBD.

Aturan:
- Gunakan HANYA angka yang ada di data. Jangan mengarang atau menghitung angka baru.
- Tingkat risiko sudah dihitung sistem; tugas Anda menjelaskan makna dan implikasinya.
- "Poin" berarti selisih antara dua persentase (mis. realisasi 50% vs target 92% = tertinggal 42 poin).
- Tulis dalam Bahasa Indonesia yang jelas sehingga pembaca awam pun paham. Gunakan format Markdown,
  tanpa emoji, tanpa tabel.
- Analisis ini adalah peringatan dini, BUKAN bukti kecurangan.

Struktur:
### Kesimpulan
2–3 kalimat sederhana: bagaimana kondisi anggaran dan apa yang paling perlu diperhatikan.
### Temuan utama
Maksimal 5, urut dari risiko tertinggi. Untuk tiap temuan tulis judul tebal, lalu poin:
Kondisi (angka) · Pembanding · Risiko/dampak · Langkah pemeriksaan yang disarankan.
### Hal yang perlu ditanyakan ke OPD/BPKAD
### Rekomendasi
### Keterbatasan"""

SYSTEM_CHAT = """Anda adalah asisten auditor internal pemerintah (APIP) yang membantu pengguna memahami
data realisasi APBD. Anda menerima ringkasan data, dan Anda punya ALAT `cari_data` untuk mencari rincian
per akun dan per wilayah dari seluruh data yang sudah diambil sistem.

Aturan:
- Jika pertanyaan butuh rincian yang tidak ada di ringkasan (akun tertentu, wilayah tertentu, anggaran Rp0,
  melebihi pagu, peringkat akun, dll.), WAJIB gunakan alat `cari_data` dulu sebelum menjawab.
- Jawab hanya berdasarkan ringkasan dan hasil alat. Sebutkan nama wilayah, akun, dan angkanya.
- Jangan mengarang angka. Jika hasil alat kosong, katakan tidak ditemukan.
- Bahasa Indonesia sederhana, ringkas, tanpa emoji."""

ALAT_AI = [{
    "type": "function",
    "function": {
        "name": "cari_data",
        "description": "Mencari baris data realisasi APBD yang sudah dianalisis sistem (per wilayah dan per akun), "
                       "lengkap dengan anggaran, realisasi, persentase, tingkat risiko, dan alasannya.",
        "parameters": {
            "type": "object",
            "properties": {
                "kondisi": {"type": "string", "enum": list(an.KONDISI.keys()),
                            "description": "Filter utama. anggaran_nol_ada_realisasi = anggaran Rp0 tetapi ada "
                                           "realisasi; melebihi_pagu = belanja di atas 100%; terlalu_cepat = belanja "
                                           "jauh di atas target waktu; tertinggal = di bawah target waktu."},
                "wilayah": {"type": "string", "description": "Sebagian nama wilayah, mis. 'Dumai' atau 'Bengkalis'. "
                                                            "Kosongkan untuk semua wilayah."},
                "akun": {"type": "string", "description": "Sebagian nama akun, mis. 'Modal' atau 'Hibah'."},
                "kelompok": {"type": "string", "enum": ["", "Pendapatan", "Belanja", "Pembiayaan"]},
                "urutkan": {"type": "string",
                            "enum": ["", "realisasi_terendah", "realisasi_tertinggi", "sisa_terbesar"]},
            },
            "required": ["kondisi"],
        },
    },
}]

CONTOH_PERTANYAAN = [
    "Akun mana yang paling perlu diperiksa?",
    "Akun apa yang anggarannya Rp0 tapi ada realisasinya?",
    "Bagaimana posisi dibanding daerah lain?",
]

# ---------------------------------------------------------------------------
# Gaya tampilan
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {--cyan:#4CC9F0; --violet:#9D8CFF; --ungu:#7B61FF; --biru:#3D5AFE; --hijau:#3DDC97; --amber:#FFB547;
       --coral:#FF6B6B; --teks:#EEF1FA; --redup:#B4BEDB; --garis:rgba(255,255,255,.12);}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {font-family: 'Plus Jakarta Sans', sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 85% -10%, rgba(61,90,254,.28), transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, rgba(157,140,255,.18), transparent 60%), #0B1026;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: linear-gradient(180deg, #121A45 0%, #0D1333 100%);
                           border-right: 1px solid var(--garis);}
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1, h2, h3 {font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.01em;}
h1 {font-size: 2.3rem !important; margin-bottom: 0 !important;}
h1 span[data-heading-text] {background: linear-gradient(90deg, #FFFFFF 0%, #8FDDF7 45%, #9D8CFF 100%);
    -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;}
.subjudul {color: var(--redup); margin-top: .3rem; margin-bottom: 1.4rem;}

/* Tombol utama: gradien */
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: linear-gradient(120deg, var(--biru) 0%, var(--ungu) 60%, #B06CFF 100%); border: 0;
  box-shadow: 0 8px 24px rgba(123,97,255,.35); font-weight: 700; transition: transform .15s, box-shadow .15s;}
.stButton > button[kind="primary"]:hover {transform: translateY(-1px); box-shadow: 0 12px 30px rgba(123,97,255,.5);}
.stButton > button[kind="secondary"] {border: 1px solid rgba(76,201,240,.45); background: rgba(76,201,240,.06);}
.stButton > button[kind="secondary"]:hover {border-color: var(--cyan); color: var(--cyan);}

/* Tab */
.stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: 1px solid var(--garis);}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: .5rem 1rem;}
.stTabs [aria-selected="true"] {background: rgba(76,201,240,.10); color: var(--cyan) !important;}
.stTabs [data-baseweb="tab-highlight"] {background: linear-gradient(90deg, var(--cyan), var(--violet));}

/* Expander dan tabel data */
[data-testid="stExpander"] details {border: 1px solid var(--garis); border-radius: 14px;
                                    background: rgba(255,255,255,.03);}
[data-testid="stDataFrame"] {border: 1px solid var(--garis); border-radius: 12px; overflow: hidden;}

/* Alur 3 langkah */
.alur {display: flex; gap: .8rem; margin: .4rem 0 1.4rem 0; flex-wrap: wrap;}
.langkah {flex: 1; min-width: 200px; border: 1px solid var(--garis); border-top: 3px solid var(--cyan);
          border-radius: 14px; padding: .85rem 1rem;
          background: linear-gradient(160deg, rgba(255,255,255,.07) 0%, rgba(255,255,255,.02) 100%);
          box-shadow: 0 14px 30px rgba(3,6,20,.45);}
.langkah:nth-child(2) {border-top-color: var(--hijau);}
.langkah .no {font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; color: var(--cyan); font-weight: 700;}
.langkah:nth-child(2) .no {color: var(--hijau);}
.langkah .judul {font-weight: 700; margin: .2rem 0; font-size: 1.02rem;}
.langkah .ket {font-size: .85rem; color: var(--redup);}
.langkah.ai {border: 1px solid rgba(255,255,255,.25); border-top: 3px solid #FFFFFF;
             background: linear-gradient(140deg, rgba(61,90,254,.55) 0%, rgba(123,97,255,.55) 100%);
             box-shadow: 0 16px 36px rgba(123,97,255,.30);}
.langkah.ai .no {color: #E9E8FF;}
.langkah.ai .ket {color: #E3E0FF;}
.status {font-size: .78rem; font-weight: 700; margin-top: .35rem; display: inline-block;
         padding: .08rem .55rem; border-radius: 999px; background: rgba(255,255,255,.08);}

/* Kartu angka */
.kartu-baris {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .8rem;
              margin: .3rem 0 1.1rem 0;}
.kartu {border: 1px solid var(--garis); border-radius: 14px; padding: .9rem 1.1rem; position: relative; overflow: hidden;
        background: linear-gradient(160deg, rgba(255,255,255,.07) 0%, rgba(255,255,255,.02) 100%);
        box-shadow: 0 14px 30px rgba(3,6,20,.45);}
.kartu::before {content: ""; position: absolute; left: 0; top: 0; right: 0; height: 3px;
                background: linear-gradient(90deg, var(--cyan), var(--violet));}
.kartu .label {font-size: .82rem; color: var(--redup); letter-spacing: .02em;}
.kartu .nilai {font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; line-height: 1.2;
               margin: .2rem 0; color: #FFFFFF;}
.kartu .nilai.kecil {font-size: 1.2rem;}
.kartu .catatan {font-size: .85rem; color: var(--redup);}

/* Label risiko, temuan */
.lencana {display: inline-block; font-weight: 700; font-size: .78rem; padding: .12rem .6rem;
          border-radius: 999px; border: 1.5px solid currentColor; margin-right: .4rem; white-space: nowrap;
          background: rgba(255,255,255,.04);}
.temuan {border-bottom: 1px solid var(--garis); padding: .6rem 0;}
.temuan .akun {font-weight: 600;}
.temuan .rinci {font-size: .9rem; color: var(--redup); margin-top: .15rem;}
.kepala-bagian {border-left: 4px solid var(--cyan); padding: .5rem 0 .5rem 1rem; margin: .4rem 0 1rem 0;
                background: linear-gradient(90deg, rgba(76,201,240,.10), transparent 70%); border-radius: 0 12px 12px 0;}
.kb-label {font-family: 'Space Grotesk', sans-serif; font-size: .9rem; font-weight: 700; letter-spacing: .12em;
           text-transform: uppercase; color: var(--cyan);}
.kb-teks {font-size: .95rem; margin-top: .25rem; line-height: 1.65; color: #D5DCEF;}
.kartu-temuan {border: 1px solid var(--garis); border-left: 4px solid; border-radius: 12px;
               padding: .9rem 1.1rem; margin-bottom: .9rem;
               background: linear-gradient(160deg, rgba(255,255,255,.06) 0%, rgba(255,255,255,.015) 100%);
               box-shadow: 0 10px 24px rgba(3,6,20,.35);}
.kt-kepala .akun {font-weight: 700; font-size: 1.05rem;}
.kt-inti {margin: .35rem 0 .55rem 0; font-weight: 500;}
.kt-isi {display: grid; grid-template-columns: 150px 1fr; gap: .3rem .9rem; font-size: .92rem;}
.kt-isi .b-label {color: #8C97B8;}
.kt-isi ul {margin: 0; padding-left: 1.1rem;}
@media (max-width: 640px) {.kt-isi {grid-template-columns: 1fr;} .kt-isi .b-label {margin-top: .3rem;}}

/* Kotak AI */
.kotak-ai {border: 1px solid rgba(157,140,255,.55); border-radius: 14px; padding: .3rem 1.3rem 1.1rem 1.3rem;
           background: linear-gradient(160deg, rgba(123,97,255,.14) 0%, rgba(61,90,254,.05) 100%);
           box-shadow: 0 14px 34px rgba(123,97,255,.18);}
.kotak-ai .kepala {font-size: .8rem; color: #C9C1FF; margin: .7rem 0 .2rem 0; letter-spacing: .04em;}
.panduan td {padding: .35rem .7rem .35rem 0; vertical-align: top; border-color: var(--garis) !important;}
</style>
""", unsafe_allow_html=True)


def lencana(risiko):
    return f'<span class="lencana" style="color:{WARNA[risiko]}">{risiko}</span>'


def kartu(label, nilai, catatan="", warna=None, kecil=False):
    gaya = f' style="color:{warna}"' if warna else ""
    return (f'<div class="kartu"><div class="label">{label}</div>'
            f'<div class="nilai{" kecil" if kecil else ""}">{nilai}</div>'
            f'<div class="catatan"{gaya}>{catatan}</div></div>')


def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        return os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# Pengambilan data (cache 6 jam agar tidak membebani server DJPK)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ambil_provinsi(tahun):
    try:
        return djpk.list_provinsi(tahun)
    except Exception:
        return PROVINSI_CADANGAN


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ambil_pemda(prov, tahun):
    try:
        d = djpk.list_pemda(prov, tahun)
        return {k: v for k, v in d.items() if k != "--"}
    except Exception:
        return PEMDA_RIAU_CADANGAN if prov == "04" else {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def ambil_apbd(tahun, periode, prov, pemda):
    df = djpk.fetch_apbd(tahun, periode, prov, pemda)
    return df, df.attrs.get("sumber", ""), df.attrs.get("keterangan", "")


def ambil_aman(tahun, periode, prov, pemda):
    """Kembalikan (df, sumber) atau (None, pesan_error)."""
    try:
        df, sumber, _ = ambil_apbd(tahun, periode, prov, pemda)
        return df, sumber
    except Exception as e:
        return None, str(e)


def ambil_banyak(daftar, judul):
    """daftar = [(label, tahun, periode, prov, pemda), ...] -> {label: df}. Paralel 4 koneksi."""
    hasil, gagal = {}, []
    bar = st.progress(0.0, text=judul)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(ambil_aman, *x[1:]): x[0] for x in daftar}
        for i, f in enumerate(as_completed(futs), 1):
            label = futs[f]
            df, _ = f.result()
            if df is not None and not df.empty:
                hasil[label] = df
            else:
                gagal.append(label)
            bar.progress(i / len(daftar), text=f"{judul} ({i}/{len(daftar)})")
    bar.empty()
    gagal = [g for g in gagal if not g.startswith("__")]
    if gagal:
        st.warning(f"Data tidak tersedia untuk: {', '.join(gagal)}")
    return hasil


def baca_upload(file):
    """Baca file ekspor DJPK (.xls XML) atau Excel/CSV sederhana berkolom Akun, Anggaran, Realisasi."""
    raw = file.getvalue()
    try:
        return djpk.parse_excel_xml(raw)
    except Exception:
        pass
    df = pd.read_csv(io.BytesIO(raw)) if file.name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(raw))
    if not {"Akun", "Anggaran", "Realisasi"}.issubset(df.columns):
        raise ValueError("File harus berformat ekspor DJPK, atau punya kolom Akun, Anggaran, Realisasi.")
    df["Akun"] = df["Akun"].astype(str).str.strip()
    return djpk._beri_level(df[["Akun", "Anggaran", "Realisasi"]].copy())


# ---------------------------------------------------------------------------
# AI (Groq)
# ---------------------------------------------------------------------------
LARANGAN_ALAT = ("PENTING: jangan memanggil alat, kode, python, browser, atau container apa pun "
                  "selain yang disediakan. Jawab langsung dengan teks biasa.")


def _galat_alat(e):
    """Model gpt-oss kadang 'berhalusinasi' memanggil alat bawaannya (mis. container.exec/python)
    yang tidak kita sediakan, sehingga Groq menolak dengan error 400 tool_use_failed."""
    t = str(e).lower()
    return "tool_use_failed" in t or "tool call validation" in t or "called a tool" in t


def _panggil(client, **kw):
    """Memanggil Groq; bila model salah memanggil alat, ulangi sampai 3 kali dengan larangan tegas."""
    for percobaan in range(3):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:
            if not _galat_alat(e) or percobaan == 2:
                raise
            msgs = list(kw["messages"])
            msgs.append({"role": "system", "content": LARANGAN_ALAT})
            kw["messages"] = msgs
            kw["temperature"] = 0.1


def tanya_groq(system, messages, model):
    from groq import Groq

    client = Groq(api_key=get_api_key())
    resp = _panggil(
        client,
        model=model,
        messages=[{"role": "system", "content": system + "\n\n" + LARANGAN_ALAT}] + messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def tanya_groq_dengan_alat(system, messages, model, semua):
    """Tanya-jawab dengan alat pencarian. AI boleh memanggil cari_data beberapa kali sebelum menjawab.
    Mengembalikan (jawaban, daftar_pencarian_yang_dijalankan)."""
    from groq import Groq

    client = Groq(api_key=get_api_key())
    msgs = [{"role": "system", "content": system}] + messages
    log = []
    for _ in range(4):
        resp = _panggil(client, model=model, messages=msgs, temperature=0.2,
                        tools=ALAT_AI, tool_choice="auto")
        m = resp.choices[0].message
        if not getattr(m, "tool_calls", None):
            return m.content, log
        if any(tc.function.name != "cari_data" for tc in m.tool_calls):
            # model memanggil alat yang tidak ada: abaikan, minta jawaban teks
            msgs.append({"role": "system", "content": "Satu-satunya alat yang ada adalah cari_data. " + LARANGAN_ALAT})
            continue
        msgs.append({"role": "assistant", "content": m.content or "",
                     "tool_calls": [{"id": tc.id, "type": "function",
                                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                    for tc in m.tool_calls]})
        for tc in m.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            args = {k: v for k, v in args.items() if k in ("kondisi", "wilayah", "akun", "kelompok", "urutkan") and v}
            hasil = an.cari_data(semua, **args)
            log.append(args)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": hasil})
    # batas putaran tercapai: minta jawaban akhir tanpa alat
    msgs.append({"role": "system", "content": "Sekarang tulis jawaban akhir. " + LARANGAN_ALAT})
    resp = _panggil(client, model=model, messages=msgs, temperature=0.2)
    return resp.choices[0].message.content, log


def pesan_galat(e):
    t = str(e)
    if "429" in t or "rate" in t.lower():
        return "Batas pemakaian Groq per menit tercapai. Tunggu sekitar 1 menit lalu klik 'Tulis ulang'."
    if "401" in t or "invalid_api_key" in t.lower():
        return "API key Groq tidak valid. Periksa GROQ_API_KEY di Secrets."
    if "model" in t.lower() and ("not found" in t.lower() or "decommission" in t.lower()):
        return "Model tidak tersedia. Pilih model lain di Pengaturan lanjutan."
    if _galat_alat(e):
        return ("AI sempat mencoba menjalankan alat yang tidak tersedia (kebiasaan model gpt-oss). "
                "Klik 'Tulis ulang' atau kirim ulang pertanyaan; bila berulang, ganti model di Pengaturan lanjutan.")
    return f"Gagal memanggil Groq: {t}"


def bagian_ai(konteks, kunci, model, semua):
    """Catatan audit (ditulis otomatis sekali per analisis) + tanya-jawab."""
    st.subheader("Langkah 3 · Catatan audit dari AI")
    st.caption(f"Ditulis oleh AI (Groq · {model}) berdasarkan angka hasil langkah 2. "
               "AI tidak menghitung ulang angka; ia menafsirkan dan menyusun catatan. Saat menjawab pertanyaan, "
               "AI dapat mencari sendiri rincian data per akun dan per wilayah.")
    if not get_api_key():
        st.info("GROQ_API_KEY belum diatur di Secrets, jadi bagian AI dinonaktifkan. "
                "Hasil perhitungan di atas tetap berlaku.")
        return

    memo = st.session_state.setdefault("memo", {})
    if kunci not in memo:
        with st.spinner("AI sedang membaca hasil analisis dan menulis catatan audit..."):
            try:
                memo[kunci] = ("ok", tanya_groq(SYSTEM_MEMO, [{"role": "user", "content": konteks}], model))
            except Exception as e:
                memo[kunci] = ("galat", pesan_galat(e))

    status, isi = memo[kunci]
    if status == "ok":
        with st.container(border=True):
            st.markdown(isi)
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("Unduh catatan (.md)", isi, file_name="catatan_audit_siaga_apbd.md")
    else:
        st.error(isi)
        c2 = st.columns([1, 4])[0]
    if c2.button("Tulis ulang", key=f"ulang_{kunci}"):
        memo.pop(kunci, None)
        st.rerun()

    with st.expander("Lihat data persis yang dikirim ke AI"):
        st.code(konteks, language="markdown")

    st.markdown("#### Tanya AI tentang data ini")
    chat = st.session_state.setdefault("chat", {}).setdefault(kunci, [])
    cols = st.columns(len(CONTOH_PERTANYAAN))
    usul = None
    for col, q in zip(cols, CONTOH_PERTANYAAN):
        if col.button(q, key=f"usul_{kunci}_{q}", width="stretch"):
            usul = q
    kotak = st.container()
    with kotak:
        for m in chat:
            with st.chat_message(m["role"]):
                tampil_pencarian(m.get("alat"))
                st.markdown(m["content"])
    with st.container():  # di dalam container -> kolom chat tampil di tempat, halaman tidak melompat ke bawah
        q = st.chat_input("Tulis pertanyaan Anda...") or usul
    if q:
        chat.append({"role": "user", "content": q})
        with kotak:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                log = []
                with st.spinner("AI sedang mencari data dan menyusun jawaban..."):
                    msgs = [{"role": "user", "content": f"RINGKASAN DATA:\n{konteks}"},
                            {"role": "assistant", "content": "Baik, ringkasan sudah saya pahami."}] + \
                           [{"role": c["role"], "content": c["content"]} for c in chat[-10:]]
                    try:
                        jawab, log = tanya_groq_dengan_alat(SYSTEM_CHAT, msgs, model, semua)
                    except Exception as e:
                        try:  # cadangan: model/akun tidak mendukung alat -> jawab dari ringkasan saja
                            jawab = tanya_groq(SYSTEM_CHAT, msgs, model)
                        except Exception as e2:
                            jawab = pesan_galat(e2 if str(e2) else e)
                tampil_pencarian(log)
                st.markdown(jawab)
        chat.append({"role": "assistant", "content": jawab, "alat": log})


def tampil_pencarian(log):
    """Tunjukkan pencarian yang dijalankan AI (bukti agent bertindak, bukan menebak)."""
    if not log:
        return
    teks = []
    for a in log:
        bagian = [f"{k}: {v}" for k, v in a.items()]
        teks.append("cari_data(" + ", ".join(bagian) + ")")
    st.caption("AI menjalankan pencarian data: " + " · ".join(teks))


# ---------------------------------------------------------------------------
# Komponen tampilan
# ---------------------------------------------------------------------------
def alur_agent(sumber, model, ai_selesai, wadah=None):
    ket_ai = "selesai" if ai_selesai else "menunggu"
    (wadah or st).markdown(f"""
<div class="alur">
  <div class="langkah"><div class="no">Langkah 1 · Sistem</div><div class="judul">Ambil data</div>
    <div class="ket">{html.escape(sumber)}</div><div class="status" style="color:{WARNA['Rendah']}">selesai</div></div>
  <div class="langkah"><div class="no">Langkah 2 · Sistem (Python)</div><div class="judul">Hitung &amp; beri tingkat risiko</div>
    <div class="ket">Bandingkan dengan target waktu, rata-rata se-provinsi, dan tahun lalu</div>
    <div class="status" style="color:{WARNA['Rendah']}">selesai</div></div>
  <div class="langkah ai"><div class="no">Langkah 3 · AI</div><div class="judul">Tulis catatan audit &amp; jawab pertanyaan</div>
    <div class="ket">Groq · {html.escape(model)} · dapat mencari data sendiri</div>
    <div class="status" style="color:#FFFFFF">{ket_ai}</div></div>
</div>""", unsafe_allow_html=True)


def ai_ok(kunci):
    return st.session_state.get("memo", {}).get(kunci, ("", ""))[0] == "ok"


def panduan_membaca(periode):
    waktu = an.pct(periode / 12 * 100)
    with st.expander("Cara membaca hasil ini", expanded=True):
        st.markdown(f"""
<table class="panduan">
<tr><td><b>Target waktu</b></td><td>Data s.d. {BULAN[periode - 1]} berarti {periode} dari 12 bulan sudah berjalan,
yaitu <b>{waktu}</b>. Jika anggaran dibelanjakan merata, idealnya realisasi sudah mendekati angka ini.</td></tr>
<tr><td><b>Poin</b></td><td>Selisih antara dua persentase. Contoh: realisasi belanja 50,4% dibanding target {waktu}
berarti <b>tertinggal 41,3 poin</b>.</td></tr>
<tr><td><b>Rata-rata se-provinsi</b></td><td>Realisasi gabungan seluruh pemda di provinsi yang sama, sebagai pembanding.</td></tr>
<tr><td><b>Tahun lalu</b></td><td>Realisasi pada bulan yang sama tahun sebelumnya.</td></tr>
<tr><td><b>Tingkat risiko</b></td><td>
{lencana('Tinggi')} {ARTI_RISIKO['Tinggi']}<br>
{lencana('Sedang')} {ARTI_RISIKO['Sedang']}<br>
{lencana('Rendah')} {ARTI_RISIKO['Rendah']}<br>
<span style="opacity:.75;font-size:.9rem">Sistem memeriksa dua arah. <b>Terlalu lambat:</b> tertinggal dari target waktu,
di bawah rata-rata se-provinsi atau tahun lalu. <b>Berlebihan:</b> ada realisasi padahal anggaran Rp0 dan belanja
melebihi anggaran (keduanya langsung Tinggi), serta belanja yang jauh lebih cepat dari target waktu.
Risiko adalah tanda untuk diperiksa, bukan bukti penyimpangan.</span></td></tr>
</table>""", unsafe_allow_html=True)


# Penjelasan akun (bahasa sederhana) dan saran pemeriksaan berbasis aturan (bukan AI)
KAMUS_AKUN = {
    "PAD": ("Pendapatan Asli Daerah: pendapatan yang dipungut sendiri oleh daerah, yaitu pajak, retribusi, "
            "laba BUMD, dan pendapatan sah lainnya.",
            "Telusuri komponen PAD mana yang paling tertinggal, lalu bandingkan target dengan potensi riil "
            "dan capaian tahun-tahun sebelumnya."),
    "Pajak Daerah": ("Pungutan wajib yang dikelola daerah, misalnya PBB perdesaan/perkotaan, BPHTB, pajak "
                     "restoran, hotel, reklame, dan penerangan jalan.",
                     "Periksa ketetapan dan piutang pajak, wajib pajak yang belum membayar, serta efektivitas "
                     "penagihan dan penyetoran ke kas daerah."),
    "Retribusi Daerah": ("Pembayaran atas jasa atau izin yang diberikan pemda, misalnya parkir, pasar, "
                         "dan pelayanan kesehatan.",
                         "Periksa jumlah objek retribusi, tarif yang berlaku, dan setoran dari OPD pemungut; "
                         "pastikan tidak ada penerimaan yang belum disetor."),
    "Hasil Pengelolaan Kekayaan Daerah yang Dipisahkan": (
        "Bagian laba dari BUMD atau bank daerah yang disetor ke pemda.",
        "Periksa jadwal pembagian dividen, keputusan RUPS, dan kinerja BUMD terkait."),
    "Lain-Lain PAD yang Sah": ("Pendapatan lain milik daerah, misalnya jasa giro, pendapatan BLUD (seperti "
                               "RSUD), denda, dan penjualan aset.",
                               "Periksa rincian sumber pendapatan dan dasar penetapan targetnya; target yang "
                               "terlalu tinggi bisa menjadi penyebab."),
    "TKDD": ("Transfer ke Daerah dari pemerintah pusat, misalnya DAU, DAK, dan Dana Bagi Hasil.",
             "Periksa jadwal dan syarat penyaluran dari pusat. Keterlambatan sering terjadi karena syarat salur "
             "(misalnya laporan) belum dipenuhi pemda."),
    "Pendapatan Transfer Pemerintah Pusat": ("Dana dari pemerintah pusat (DAU, DAK, Dana Bagi Hasil, dan lainnya).",
                                             "Periksa jadwal penyaluran dan syarat salur yang belum dipenuhi pemda."),
    "Pendapatan Lainnya": ("Pendapatan di luar PAD dan transfer pusat, misalnya hibah, dana darurat, dan "
                           "transfer dari provinsi.",
                           "Lihat rincian akun di bawahnya untuk mengetahui sumber yang tertinggal."),
    "Pendapatan Transfer Antar Daerah": ("Bagi hasil atau bantuan keuangan dari provinsi atau daerah lain.",
                                         "Periksa jadwal penyaluran dari provinsi dan kelengkapan syarat pencairan."),
    "Dana Darurat": ("Dana dari pemerintah pusat untuk penanganan bencana atau keadaan darurat.",
                     "Periksa dasar penerimaan dan apakah sudah dimasukkan ke dalam perubahan APBD."),
    "Pendapatan Hibah": ("Pemberian dari pihak lain kepada pemda.",
                         "Periksa perjanjian hibah dan pencatatannya."),
    "Belanja Pegawai": ("Gaji, tunjangan, dan honorarium aparatur sipil negara.",
                        "Bandingkan dengan jumlah pegawai dan jadwal pembayaran; telusuri apakah anggaran tunjangan "
                        "(misalnya TPP) disusun berlebih atau ada pembayaran yang tertunda."),
    "Belanja Barang dan Jasa": ("Belanja operasional, misalnya alat tulis, perjalanan dinas, pemeliharaan, "
                                "dan jasa pihak ketiga.",
                                "Periksa rencana dan realisasi pengadaan, kegiatan yang belum berjalan, serta risiko "
                                "penumpukan pertanggungjawaban (SPJ) di bulan Desember."),
    "Belanja Modal": ("Belanja untuk membangun atau membeli aset, misalnya jalan, jembatan, gedung, "
                      "kendaraan, dan peralatan.",
                      "Periksa daftar kontrak dan progres fisik pekerjaan, jadwal termin, serta risiko pembayaran "
                      "penuh atas pekerjaan yang belum selesai menjelang tutup tahun."),
    "Belanja Lainnya": ("Kelompok belanja yang terdiri atas hibah, bantuan sosial, bantuan keuangan, bagi hasil, "
                        "dan belanja tidak terduga.",
                        "Lihat rincian akun di bawahnya untuk mengetahui bagian yang tertinggal."),
    "Belanja Hibah": ("Pemberian uang/barang kepada lembaga, organisasi kemasyarakatan, atau instansi lain.",
                      "Periksa naskah perjanjian hibah (NPHD), daftar penerima dan kelayakannya, jadwal penyaluran, "
                      "serta laporan pertanggungjawaban penerima."),
    "Belanja Bantuan Sosial": ("Bantuan kepada masyarakat untuk melindungi dari risiko sosial.",
                               "Periksa keputusan penetapan penerima, kesesuaian kriteria, dan bukti penyaluran."),
    "Belanja Bantuan Keuangan": ("Transfer dari pemda ke desa (misalnya Alokasi Dana Desa) atau pemda lain.",
                                 "Periksa jadwal penyaluran dan syarat pencairan yang belum dipenuhi penerima."),
    "Belanja Bagi Hasil": ("Bagian pajak dan retribusi daerah yang wajib dibagikan ke desa.",
                           "Periksa dasar perhitungan dan jadwal penyaluran; keterlambatan merugikan keuangan desa."),
    "Belanja Tidak Terduga": ("Cadangan untuk keadaan darurat, bencana, atau kebutuhan mendesak.",
                              "Realisasi rendah wajar bila tidak ada keadaan darurat. Jika ada pemakaian, pastikan "
                              "didukung penetapan status darurat."),
    "Belanja Bunga": ("Pembayaran bunga atas pinjaman daerah.",
                      "Periksa jadwal pembayaran bunga sesuai perjanjian pinjaman."),
    "Belanja Subsidi": ("Bantuan biaya produksi kepada perusahaan/lembaga agar harga layanan terjangkau.",
                        "Periksa dasar penetapan subsidi dan penerimanya."),
}
SARAN_KHUSUS = {
    "Tanpa anggaran": "Periksa dasar hukum realisasi tanpa anggaran ini: apakah sudah ditampung dalam perubahan APBD "
                      "atau peraturan kepala daerah, dan apakah bukti pendukungnya lengkap.",
    "Melebihi pagu": "Periksa dokumen pencairan (SP2D) yang membuat realisasi melampaui anggaran dan dasar hukumnya.",
    "Terlalu cepat": "Periksa apakah ada pembayaran di muka, salah pencatatan akun, atau anggaran yang akan habis "
                     "sebelum akhir tahun.",
}


def _baris(label, isi):
    return f'<div class="b-label">{label}</div><div class="b-isi">{isi}</div>'


def saran_untuk(x):
    """Saran pemeriksaan berbasis aturan (bukan AI) untuk satu akun."""
    saran = KAMUS_AKUN.get(x["Akun"], ("", ""))[1]
    return " ".join([SARAN_KHUSUS[j] for j in SARAN_KHUSUS if j in x["Jenis"]] + ([saran] if saran else []))


def kartu_temuan(x):
    """Satu akun berisiko, dijelaskan rinci dan terstruktur."""
    waktu = x["Waktu%"]
    pendapatan = x["Kelompok"] == "Pendapatan"
    kata = "diterima" if pendapatan else "dibelanjakan"
    arti = KAMUS_AKUN.get(x["Akun"], ("", ""))[0]

    if "Tanpa anggaran" in x["Jenis"]:
        inti = f"Ada realisasi {an.rp(x['Realisasi'])}, padahal akun ini tidak dianggarkan (Rp0)."
    elif "Melebihi pagu" in x["Jenis"]:
        inti = f"Sudah {kata} {an.pct(x['Real%'])}, melebihi anggaran yang tersedia."
    elif "Terlalu cepat" in x["Jenis"]:
        inti = f"Sudah {kata} {an.pct(x['Real%'])}, jauh lebih cepat dari target waktu {an.pct(waktu)}."
    else:
        inti = f"Baru {kata} {an.pct(x['Real%'])}, padahal target waktu {an.pct(waktu)}."

    isi = []
    if arti:
        isi.append(_baris("Apa ini", html.escape(arti)))
    isi.append(_baris("Kondisi", f"Anggaran {an.rp(x['Anggaran'])}, sudah {kata} {an.rp(x['Realisasi'])}"
                                 + (f" ({an.pct(x['Real%'])}). Belum {kata}: {an.rp(x['Sisa'])}."
                                    if x["Anggaran"] > 0 else ".")))
    if x["Anggaran"] > 0 and x["Gap_Waktu"] < 0:
        seharusnya = x["Anggaran"] * waktu / 100
        isi.append(_baris("Seharusnya", f"Jika merata, sampai bulan ini seharusnya sudah sekitar "
                                        f"<b>{an.rp(seharusnya)}</b> ({an.pct(waktu)}). Kurang sekitar "
                                        f"<b>{an.rp(seharusnya - x['Realisasi'])}</b>, atau "
                                        f"{an.selisih_kata(x['Gap_Waktu'])}."))
    pb = []
    if pd.notna(x.get("Real%_Pembanding")):
        pb.append(f"rata-rata se-provinsi {an.pct(x['Real%_Pembanding'])}")
    if pd.notna(x.get("Real%_TahunLalu")):
        pb.append(f"bulan yang sama tahun lalu {an.pct(x['Real%_TahunLalu'])}")
    if pb:
        isi.append(_baris("Pembanding", "; ".join(pb).capitalize() + "."))
    alasan = [a.replace("info: ", "") for a in x["Alasan"].split("; ") if a]
    if alasan:
        isi.append(_baris("Mengapa berisiko",
                          "<ul>" + "".join(f"<li>{html.escape(a[:1].upper() + a[1:])}</li>" for a in alasan)
                          + "</ul>"))
    saran_all = saran_untuk(x)
    if saran_all:
        isi.append(_baris("Saran pemeriksaan", html.escape(saran_all)))

    return (f'<div class="kartu-temuan" style="border-left-color:{WARNA[x["Risiko"]]}">'
            f'<div class="kt-kepala">{lencana(x["Risiko"])}<span class="akun">{html.escape(x["Akun"])}</span>'
            f'<span style="opacity:.6;font-size:.85rem"> · {x["Kelompok"]}</span></div>'
            f'<div class="kt-inti">{html.escape(inti)}</div>'
            f'<div class="kt-isi">{"".join(isi)}</div></div>')


def daftar_temuan(h, nama=""):
    r = h[(h["Risiko"] != "Rendah") & (h["Level"] > 1)].sort_values(["Skor", "Sisa"], ascending=False)
    n_t, n_s = int((r["Risiko"] == "Tinggi").sum()), int((r["Risiko"] == "Sedang").sum())
    st.markdown(f"""
<div class="kepala-bagian">
  <div class="kb-label">Pengukuran risiko{(" · " + html.escape(nama)) if nama else ""}</div>
  <div class="kb-teks">Bagian ini menampilkan <b>semua akun anggaran yang diberi tanda risiko oleh sistem</b>:
  {lencana('Tinggi')}<b>{n_t}</b> akun dan {lencana('Sedang')}<b>{n_s}</b> akun, diurutkan dari yang paling
  berisiko. Setiap akun dijelaskan: apa artinya, kondisinya, berapa yang seharusnya sudah terealisasi,
  pembandingnya, alasan diberi tanda, dan saran pemeriksaan. Akun berisiko rendah tidak ditampilkan di sini
  (lihat tab Tabel lengkap).</div>
</div>""", unsafe_allow_html=True)
    if r.empty:
        st.success("Tidak ada akun dengan risiko sedang atau tinggi.")
        return
    kartu_list = [kartu_temuan(x) for _, x in r.iterrows()]
    tampil = 5
    st.markdown("".join(kartu_list[:tampil]), unsafe_allow_html=True)
    if len(kartu_list) > tampil:
        with st.expander(f"Lihat {len(kartu_list) - tampil} akun berisiko lainnya"):
            st.markdown("".join(kartu_list[tampil:]), unsafe_allow_html=True)


def rinci_anomali(x):
    alasan = x["Alasan"].replace("info: ", "")
    alasan = alasan[:1].upper() + alasan[1:]
    if "Tanpa anggaran" in x["Jenis"]:
        return f"{alasan}."
    return f"Anggaran {an.rp(x['Anggaran'])}, realisasi {an.rp(x['Realisasi'])}. {alasan}."


def daftar_anomali(semua, fokus=None):
    a = an.cari_anomali(semua)
    st.markdown("""
<div class="kepala-bagian">
  <div class="kb-label">Deteksi anomali</div>
  <div class="kb-teks">Bagian ini mencari akun yang realisasinya <b>berlebihan atau tidak wajar</b> di semua wilayah
  yang datanya diambil: ada realisasi padahal anggarannya Rp0, belanja melebihi anggaran, belanja jauh lebih cepat
  dari target waktu, atau pendapatan jauh melampaui target (sebagai informasi).</div>
</div>""", unsafe_allow_html=True)
    if a.empty:
        st.success("Tidak ditemukan anomali pada wilayah yang dianalisis.")
        return
    baris = []
    for _, x in a.iterrows():
        label = lencana(x["Risiko"]) if x["Jenis"] != "Info" else \
            '<span class="lencana" style="color:#868e96">Info</span>'
        tebal = "font-weight:700;" if fokus and x["Wilayah"] == fokus else ""
        baris.append(f'<div class="temuan">{label}<span class="akun">{html.escape(x["Akun"])}</span>'
                     f'<span style="opacity:.75;{tebal}"> · {html.escape(x["Wilayah"])}</span>'
                     f'<span style="opacity:.6;font-size:.85rem"> · {x["Jenis"]}</span>'
                     f'<div class="rinci">{html.escape(rinci_anomali(x))}</div></div>')
    st.markdown("".join(baris), unsafe_allow_html=True)


def tabel_lengkap(h):
    t = pd.DataFrame({
        "Akun": h.apply(lambda r: (" " * (r["Level"] - 1)) + r["Akun"], axis=1),
        "Anggaran (Rp miliar)": h["Anggaran"] / 1e9,
        "Realisasi (Rp miliar)": h["Realisasi"] / 1e9,
        "Realisasi": h["Real%"],
        "Dibanding target waktu": h["Gap_Waktu"].map(an.selisih_kata),
        "Tahun lalu": h["Real%_TahunLalu"],
        "Rata-rata se-provinsi": h["Real%_Pembanding"],
        "Risiko": h["Risiko"],
        "Alasan": h["Alasan"].replace("", "–"),
    })
    t.loc[h["Kelompok"] == "Pembiayaan", ["Dibanding target waktu", "Risiko"]] = ["–", "–"]
    sty = (t.style
           .format({"Anggaran (Rp miliar)": lambda v: an.angka(v), "Realisasi (Rp miliar)": lambda v: an.angka(v),
                    "Realisasi": an.pct, "Tahun lalu": an.pct, "Rata-rata se-provinsi": an.pct})
           .map(lambda v: f"color:{WARNA[v]};font-weight:700" if v in WARNA else "", subset=["Risiko"])
           .apply(lambda r: ["font-weight:700" if lv == 1 else "" for lv in [h.loc[r.name, "Level"]] * len(r)],
                  axis=1))
    st.dataframe(sty, hide_index=True, width="stretch", height=min(38 * (len(t) + 1), 760),
                 column_config={"Akun": st.column_config.TextColumn(width="large"),
                                "Alasan": st.column_config.TextColumn(width="large")})


def grafik_kelompok(h, kelompok, periode):
    d = h[(h["Kelompok"] == kelompok) & (h["Anggaran"] > 0)]
    waktu = periode / 12 * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["Real%"], y=d["Akun"], orientation="h", name="Realisasi",
        marker_color=[WARNA[r] for r in d["Risiko"]],
        text=[an.pct(v) for v in d["Real%"]], textposition="outside", cliponaxis=False,
        customdata=d[["Risiko", "Alasan"]].fillna(""),
        hovertemplate="<b>%{y}</b><br>Realisasi %{x:.1f}%<br>Risiko %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
        showlegend=False))
    if d["Real%_Pembanding"].notna().any():
        fig.add_trace(go.Scatter(x=d["Real%_Pembanding"], y=d["Akun"], mode="markers", name="Rata-rata se-provinsi",
                                 marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#B4BEDB"))))
    fig.add_vline(x=waktu, line_dash="dash", line_color="#4CC9F0")
    fig.add_annotation(x=waktu, y=1.02, yref="paper", text=f"Target waktu {an.pct(waktu)}", showarrow=False,
                       font=dict(color="#4CC9F0"), xanchor="left")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#D5DCEF"),
                      height=60 + 34 * len(d), margin=dict(l=10, r=40, t=30, b=10),
                      xaxis=dict(title="Realisasi (% dari anggaran)", range=[0, max(110, d["Real%"].max() + 15)]),
                      yaxis=dict(autorange="reversed", title=""), legend=dict(orientation="h", y=-0.15),
                      bargap=0.35)
    return fig


def grafik_peringkat(p, fokus, periode):
    waktu = periode / 12 * 100
    warna = ["#FF6B6B" if w == fokus else "#5B7CFA" for w in p["Wilayah"]]
    fig = go.Figure(go.Bar(x=p["Belanja Daerah"], y=p["Wilayah"], orientation="h", marker_color=warna,
                           text=[an.pct(v) for v in p["Belanja Daerah"]], textposition="outside",
                           cliponaxis=False, name="Realisasi belanja daerah"))
    fig.add_trace(go.Scatter(x=p["Belanja Modal"], y=p["Wilayah"], mode="markers", name="Realisasi belanja modal",
                             marker=dict(symbol="circle", size=9, color="#EEF1FA",
                                         line=dict(width=1, color="#0B1026"))))
    fig.add_vline(x=waktu, line_dash="dash", line_color="#4CC9F0")
    fig.add_annotation(x=waktu, y=1.02, yref="paper", text=f"Target waktu {an.pct(waktu)}", showarrow=False,
                       font=dict(color="#4CC9F0"), xanchor="left")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#D5DCEF"),
                      height=80 + 32 * len(p), margin=dict(l=10, r=40, t=30, b=10),
                      xaxis=dict(title="Realisasi (% dari anggaran)", range=[0, 110]),
                      yaxis=dict(autorange="reversed", title=""), legend=dict(orientation="h", y=-0.12))
    return fig


def tabel_peringkat(p):
    t = p.copy()
    t["Anggaran Belanja"] = t["Anggaran Belanja"] / 1e9
    t["Sisa Belanja"] = t["Sisa Belanja"] / 1e9
    t = t.rename(columns={"Anggaran Belanja": "Anggaran belanja (Rp miliar)",
                          "Sisa Belanja": "Belum terealisasi (Rp miliar)",
                          "Gap Belanja vs Waktu": "Belanja vs target waktu"})
    t["Belanja vs target waktu"] = t["Belanja vs target waktu"].map(an.selisih_kata)
    fmt = {k: an.pct for k in an.AKUN_KUNCI}
    fmt.update({"Anggaran belanja (Rp miliar)": an.angka, "Belum terealisasi (Rp miliar)": an.angka})
    st.dataframe(t.style.format(fmt), hide_index=True, width="stretch")


def unduh_excel(lembar: dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nama, df in lembar.items():
            df.to_excel(w, sheet_name=nama[:31], index=False)
    return buf.getvalue()



# ---------------------------------------------------------------------------
# Rasio keuangan & kepatuhan belanja wajib
# ---------------------------------------------------------------------------
WARNA_STATUS = {"Melebihi batas": WARNA["Tinggi"], "Di bawah batas": WARNA["Tinggi"],
                "Sesuai batas": WARNA["Rendah"], "Informasi": "#9D8CFF", "Tidak ada data": "#8C97B8"}


def lencana_status(status):
    return f'<span class="lencana" style="color:{WARNA_STATUS.get(status, "#8C97B8")}">{status}</span>'


def bagian_rasio(r, nama=""):
    st.markdown(f"""
<div class="kepala-bagian">
  <div class="kb-label">Rasio keuangan &amp; kepatuhan{(" · " + html.escape(nama)) if nama else ""}</div>
  <div class="kb-teks">Bagian ini menilai <b>komposisi</b> APBD, bukan kecepatan realisasinya: seberapa mandiri
  pendapatan daerah, dan apakah porsi belanja pegawai serta belanja infrastruktur sesuai batas
  <b>UU 1/2022 tentang HKPD</b>. Rasio kepatuhan dihitung dari APBD (anggaran), karena batas undang-undang
  berlaku pada penyusunan anggaran; angka realisasi ditampilkan sebagai pembanding.</div>
</div>""", unsafe_allow_html=True)
    kartu_list = []
    for _, x in r.iterrows():
        warna = WARNA_STATUS.get(x["Status"], "#8C97B8")
        pb = (f"Se-provinsi {an.pct(x['Se-provinsi %'])}. " if pd.notna(x["Se-provinsi %"]) else "")
        isi = [_baris("Rumus", html.escape(x["Rumus"])),
               _baris("Hasil", f"APBD <b>{an.pct(x['APBD %'])}</b> · realisasi s.d. bulan ini "
                               f"{an.pct(x['Realisasi %'])}. {pb}Acuan: {x['Acuan']}."),
               _baris("Penjelasan", html.escape(x["Penjelasan"]))]
        kartu_list.append(
            f'<div class="kartu-temuan" style="border-left-color:{warna}">'
            f'<div class="kt-kepala">{lencana_status(x["Status"])}<span class="akun">{html.escape(x["Rasio"])}</span></div>'
            f'<div class="kt-isi" style="margin-top:.5rem">{"".join(isi)}</div></div>')
    st.markdown("".join(kartu_list), unsafe_allow_html=True)
    st.caption("Batas belanja pegawai 30% (Pasal 146) berlaku penuh mulai 2027; per Mei 2026 pemerintah membahas "
               "perpanjangan masa transisinya. Status di sini adalah indikasi untuk ditelusuri, bukan kesimpulan "
               "pelanggaran.")


def grafik_rasio_lintas(rl, fokus=None):
    """Scatter: kemandirian fiskal (x) vs realisasi belanja (y), satu titik per wilayah."""
    d = rl.dropna(subset=["Kemandirian fiskal %", "Realisasi belanja %"])
    warna = ["#FF6B6B" if w == fokus else "#4CC9F0" for w in d["Wilayah"]]
    fig = go.Figure(go.Scatter(
        x=d["Kemandirian fiskal %"], y=d["Realisasi belanja %"], mode="markers+text", text=d["Wilayah"],
        textposition="top center", textfont=dict(size=10, color="#B4BEDB"),
        marker=dict(size=13, color=warna, line=dict(width=1, color="#0B1026")),
        hovertemplate="<b>%{text}</b><br>Kemandirian %{x:.1f}%<br>Realisasi belanja %{y:.1f}%<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#D5DCEF"),
                      height=460, margin=dict(l=10, r=20, t=20, b=10),
                      xaxis=dict(title="Kemandirian fiskal (PAD ÷ pendapatan, %)", gridcolor="rgba(255,255,255,.08)"),
                      yaxis=dict(title="Realisasi belanja (%)", gridcolor="rgba(255,255,255,.08)"))
    return fig


def tabel_rasio_lintas(rl):
    fmt = {c: an.pct for c in rl.columns if c.endswith("%")}

    def warnai(v):
        return f"color:{WARNA_STATUS[v]};font-weight:600" if v in WARNA_STATUS else ""
    gaya = rl.style.format(fmt)
    gaya = (gaya.map if hasattr(gaya, "map") else gaya.applymap)(
        warnai, subset=["Status pegawai (maks 30%)", "Status modal (min 40%)"])
    st.dataframe(gaya, hide_index=True, width="stretch")


def bagian_rasio_lintas(rl, fokus=None):
    if rl is None or rl.empty:
        st.info("Data pembanding antar wilayah tidak tersedia.")
        return
    n_peg = int((rl["Status pegawai (maks 30%)"] == "Melebihi batas").sum())
    n_mod = int((rl["Status modal (min 40%)"] == "Di bawah batas").sum())
    st.markdown(f"**Kepatuhan antar wilayah:** {n_peg} dari {len(rl)} wilayah porsi belanja pegawainya di atas 30%, "
                f"dan {n_mod} dari {len(rl)} porsi belanja modalnya di bawah 40% (dihitung dari APBD).")
    st.caption("Grafik: setiap titik satu wilayah. Makin ke kanan makin mandiri pendapatannya; makin ke atas makin "
               "tinggi realisasi belanjanya. Titik merah = wilayah yang dipilih.")
    st.plotly_chart(grafik_rasio_lintas(rl, fokus), width="stretch", key=f"rl_{fokus}")
    tabel_rasio_lintas(rl.sort_values("Belanja pegawai %", ascending=False))


# ---------------------------------------------------------------------------
# Tren bulanan
# ---------------------------------------------------------------------------
BULAN_PENDEK = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def ambil_tren(tahun, periode, prov, pemda):
    daftar = [(f"{tahun}|{b}", tahun, b, prov, pemda) for b in range(1, periode + 1)]
    daftar += [(f"{tahun - 1}|{b}", tahun - 1, b, prov, pemda) for b in range(1, 13)]
    data = ambil_banyak(daftar, "Mengambil data tiap bulan dari DJPK")
    ini = {int(k.split("|")[1]): v for k, v in data.items() if k.startswith(f"{tahun}|")}
    lalu = {int(k.split("|")[1]): v for k, v in data.items() if k.startswith(f"{tahun - 1}|")}
    return an.susun_tren(ini), an.susun_tren(lalu)


def grafik_tren(ini, lalu, akun, periode, tahun, proy=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=BULAN_PENDEK, y=[(b / 12 * 100) for b in range(1, 13)], name="Target waktu (merata)",
                             mode="lines", line=dict(color="#8C97B8", dash="dot", width=2)))
    l = lalu[lalu["Akun"] == akun].sort_values("Bulan")
    if not l.empty:
        fig.add_trace(go.Scatter(x=[BULAN_PENDEK[b - 1] for b in l["Bulan"]], y=l["Real%"], name=f"Tahun {tahun - 1}",
                                 mode="lines+markers", line=dict(color="#9D8CFF", dash="dash", width=2),
                                 marker=dict(size=6)))
    i = ini[ini["Akun"] == akun].sort_values("Bulan")
    if not i.empty:
        fig.add_trace(go.Scatter(x=[BULAN_PENDEK[b - 1] for b in i["Bulan"]], y=i["Real%"], name=f"Tahun {tahun}",
                                 mode="lines+markers", line=dict(color="#4CC9F0", width=4), marker=dict(size=9)))
    if proy:
        fig.add_trace(go.Scatter(x=[BULAN_PENDEK[periode - 1], "Des"], y=[proy["Sekarang %"], proy["Perkiraan Desember %"]],
                                 name="Proyeksi (pola tahun lalu)", mode="lines+markers",
                                 line=dict(color="#4CC9F0", dash="dot", width=3),
                                 marker=dict(size=10, symbol="diamond-open")))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#D5DCEF"),
                      height=440, margin=dict(l=10, r=20, t=20, b=10), hovermode="x unified",
                      yaxis=dict(title="Realisasi kumulatif (% dari anggaran)", range=[0, 110],
                                 gridcolor="rgba(255,255,255,.08)"),
                      xaxis=dict(gridcolor="rgba(255,255,255,.05)"), legend=dict(orientation="h", y=-0.15))
    return fig


def bagian_tren(kunci, kode, tahun, periode):
    st.markdown("""
<div class="kepala-bagian">
  <div class="kb-label">Tren bulanan &amp; proyeksi akhir tahun</div>
  <div class="kb-teks">Grafik <b>kurva S</b>: realisasi kumulatif dari Januari sampai bulan terpilih, dibandingkan
  dengan tahun lalu dan target waktu. Dari sini terlihat <b>kapan serapan melambat</b> dan apakah realisasi
  biasanya <b>menumpuk di bulan Desember</b>, yang merupakan risiko audit (pekerjaan dan SPJ dikebut di akhir tahun).
  Proyeksi akhir tahun memakai pola tahun lalu: jika sisa tahun berjalan seperti tahun lalu, berapa yang akhirnya
  terserap.</div>
</div>""", unsafe_allow_html=True)
    if kode is None:
        st.info("Tren bulanan hanya tersedia pada sumber data 'Ambil otomatis dari DJPK' mode Satu pemda, "
                "karena sistem perlu mengambil data setiap bulan.")
        return None
    simpan = st.session_state.setdefault("tren", {})
    if kunci not in simpan:
        st.caption(f"Sistem akan mengambil {periode + 12} data bulanan dari DJPK ({periode} bulan tahun {tahun} dan "
                   f"12 bulan tahun {tahun - 1}). Data disimpan sementara 6 jam.")
        if st.button("Ambil data tren bulanan", type="primary", key=f"tren_{kunci}"):
            simpan[kunci] = ambil_tren(tahun, periode, *kode)
            st.rerun()
        return None
    ini, lalu = simpan[kunci]
    if ini.empty:
        st.warning("Data bulanan tidak berhasil diambil. Coba lagi nanti.")
        if st.button("Coba lagi", key=f"tren_ulang_{kunci}"):
            simpan.pop(kunci, None)
            st.rerun()
        return None
    pilihan = [a for a in an.AKUN_TREN if a in set(ini["Akun"])]
    akun = st.selectbox("Akun", pilihan, index=pilihan.index("Belanja Daerah") if "Belanja Daerah" in pilihan else 0,
                        key=f"tren_akun_{kunci}")
    proy_semua = [p for p in (an.proyeksi_akhir_tahun(ini, lalu, a, periode) for a in pilihan) if p]
    proy = next((p for p in proy_semua if p["Akun"] == akun), None)
    st.plotly_chart(grafik_tren(ini, lalu, akun, periode, tahun, proy), width="stretch", key=f"gt_{kunci}_{akun}")
    if proy:
        kartu_html = [
            kartu("Realisasi sekarang", an.pct(proy["Sekarang %"]),
                  f"tahun lalu pada bulan yang sama {an.pct(proy['Tahun lalu s.d. bulan ini %'])}"),
            kartu("Perkiraan akhir tahun", an.pct(proy["Perkiraan Desember %"]),
                  f"tahun lalu berakhir di {an.pct(proy['Tahun lalu Desember %'])}",
                  WARNA["Tinggi"] if proy["Perkiraan Desember %"] < 85 else None),
            kartu("Potensi tidak terserap", an.rp(proy["Tidak terserap (Rp)"]), "jika pola sama dengan tahun lalu",
                  kecil=True),
        ]
        if pd.notna(proy["Lonjakan Desember tahun lalu (poin)"]):
            lonjak = proy["Lonjakan Desember tahun lalu (poin)"]
            kartu_html.append(kartu("Lonjakan Desember tahun lalu", f"+{an.angka(lonjak, 1)} poin",
                                    "naik dalam satu bulan terakhir saja",
                                    WARNA["Sedang"] if lonjak >= 15 else None))
        st.markdown(f'<div class="kartu-baris">{"".join(kartu_html)}</div>', unsafe_allow_html=True)
    elif periode < 12:
        st.caption("Proyeksi tidak dapat dihitung karena data tahun lalu belum lengkap.")
    with st.expander("Tabel proyeksi semua akun"):
        if proy_semua:
            t = pd.DataFrame(proy_semua)
            fmt = {c: an.pct for c in t.columns if c.endswith("%")}
            fmt["Tidak terserap (Rp)"] = an.rp
            fmt["Lonjakan Desember tahun lalu (poin)"] = lambda v: an.angka(v, 1)
            st.dataframe(t.style.format(fmt), hide_index=True, width="stretch")
        else:
            st.write("Belum ada proyeksi.")
    st.caption("Proyeksi = realisasi sekarang × (realisasi Desember tahun lalu ÷ realisasi bulan yang sama tahun lalu). "
               "Ini perkiraan sederhana, bukan kepastian.")
    return proy_semua


# ---------------------------------------------------------------------------
# Kertas kerja audit (Excel)
# ---------------------------------------------------------------------------
KOLOM_TEMUAN = {"Akun": "Akun", "Kelompok": "Kelompok", "Risiko": "Risiko", "Skor": "Skor",
                "Anggaran": "Anggaran (Rp)", "Realisasi": "Realisasi (Rp)", "Real%": "Realisasi %",
                "Waktu%": "Target waktu %", "Gap_Waktu": "Selisih vs target (poin)",
                "Real%_Pembanding": "Se-provinsi %", "Real%_TahunLalu": "Tahun lalu %",
                "Sisa": "Belum terealisasi (Rp)", "Alasan": "Alasan risiko"}


def tabel_temuan(h, wilayah=False):
    r = h[(h["Risiko"] != "Rendah") & (h["Level"] > 1)].sort_values(["Skor", "Sisa"], ascending=False).copy()
    kol = (["Wilayah"] if wilayah else []) + [k for k in KOLOM_TEMUAN if k in r.columns]
    t = r[kol].rename(columns=KOLOM_TEMUAN)
    t["Saran pemeriksaan"] = [saran_untuk(x) for _, x in r.iterrows()]
    t["Tanggapan auditee"] = ""
    t["Kesimpulan auditor"] = ""
    return t


def tabel_anomali(a):
    if a is None or a.empty:
        return None
    kol = ["Wilayah", "Jenis"] + [k for k in KOLOM_TEMUAN if k in a.columns]
    t = a[kol].rename(columns=KOLOM_TEMUAN).rename(columns={"Jenis": "Jenis anomali"})
    t["Saran pemeriksaan"] = [saran_untuk(x) for _, x in a.iterrows()]
    return t


def tabel_semua(h):
    kol = [k for k in KOLOM_TEMUAN if k in h.columns]
    return h[["Level"] + kol].rename(columns=KOLOM_TEMUAN)


def kertas_kerja(info: dict, lembar: dict) -> bytes:
    """Excel berformat: lembar Ringkasan + lembar-lembar data, header berwarna, kolom lebar, Rp & % rapi."""
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"Keterangan": list(info.keys()), "Isi": [str(v) for v in info.values()]}).to_excel(
            w, sheet_name="Ringkasan", index=False)
        for nama, df in lembar.items():
            if df is not None and not df.empty:
                df.to_excel(w, sheet_name=nama[:31], index=False)
        kepala = PatternFill("solid", fgColor="1E2A63")
        isi_risiko = {"Tinggi": "FFD9D9", "Sedang": "FFEFD1", "Melebihi batas": "FFD9D9", "Di bawah batas": "FFD9D9",
                      "Sesuai batas": "D7F7E9"}
        for ws in w.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = kepala
                c.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[1].height = 32
            for j, col in enumerate(ws.iter_cols(min_row=1, max_row=ws.max_row), 1):
                judul = str(col[0].value or "")
                panjang = max(len(str(c.value or "")) for c in col[: min(len(col), 60)])
                lebar = 60 if judul in ("Alasan risiko", "Saran pemeriksaan", "Penjelasan", "Isi", "Catatan") else \
                    min(max(12, panjang + 2), 38)
                ws.column_dimensions[get_column_letter(j)].width = lebar
                for c in col[1:]:
                    if "(Rp)" in judul and isinstance(c.value, (int, float)):
                        c.number_format = '#,##0'
                    elif (judul.endswith("%") or "(poin)" in judul) and isinstance(c.value, (int, float)):
                        c.number_format = '0.0'
                    if lebar == 60:
                        c.alignment = Alignment(wrap_text=True, vertical="top")
                    if isinstance(c.value, str) and c.value in isi_risiko:
                        c.fill = PatternFill("solid", fgColor=isi_risiko[c.value])
                        c.font = Font(bold=True)
    return buf.getvalue()


def peringkat_excel(p):
    if p is None or p.empty:
        return None
    ganti = {a: f"{a} %" for a in an.AKUN_KUNCI}
    ganti.update({"Anggaran Belanja": "Anggaran belanja (Rp)", "Sisa Belanja": "Belum terealisasi (Rp)",
                  "Gap Belanja vs Waktu": "Belanja vs target (poin)"})
    return p.rename(columns=ganti)


def lembar_ai(kunci):
    m = st.session_state.get("memo", {}).get(kunci)
    if not m or m[0] != "ok":
        return None
    return pd.DataFrame({"Catatan": m[1].splitlines()})


def bagian_unduh(info, lembar, nama_file):
    st.subheader("Langkah 4 · Unduh kertas kerja audit")
    st.caption("File Excel berisi beberapa lembar: " + ", ".join(["Ringkasan"] + [k for k, v in lembar.items()
                                                                                    if v is not None and not v.empty])
               + ". Lembar Temuan risiko menyediakan kolom kosong 'Tanggapan auditee' dan 'Kesimpulan auditor' "
               "untuk diisi saat pemeriksaan.")
    st.download_button("Unduh kertas kerja (.xlsx)", kertas_kerja(info, lembar), file_name=nama_file,
                       type="primary", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------------------------------------------------------------------
# Halaman hasil
# ---------------------------------------------------------------------------
def tampil_pemda(nama, tahun, periode, h, peringkat, keterangan, model, sumber, semua,
                 df_now=None, df_pb=None, data=None, kode=None, ambang=15):
    waktu = periode / 12 * 100
    idx = h.set_index("Akun")
    kunci = f"pemda|{nama}|{tahun}|{periode}"
    rasio = an.rasio_keuangan(df_now, df_pb) if df_now is not None else None
    rasio_lain = an.rasio_lintas(data) if data and len(data) > 1 else None

    st.markdown(f"## {nama}")
    st.markdown(f'<div class="subjudul">Realisasi APBD s.d. {BULAN[periode - 1]} {tahun}. {html.escape(keterangan)}</div>',
                unsafe_allow_html=True)
    wadah_alur = st.empty()

    # --- Ringkasan angka
    def nilai(akun):
        return idx.loc[akun, "Real%"] if akun in idx.index else None

    kartu_html = [kartu("Target waktu", an.pct(waktu), f"{periode} dari 12 bulan berjalan")]
    for akun in ["Pendapatan Daerah", "Belanja Daerah", "Belanja Modal"]:
        v = nilai(akun)
        if v is None:
            continue
        gap = v - waktu
        kartu_html.append(kartu(f"Realisasi {akun.lower()}", an.pct(v), an.selisih_kata(gap),
                                WARNA["Tinggi"] if gap <= -30 else WARNA["Sedang"] if gap <= -15 else None))
    n_t = int(((h["Risiko"] == "Tinggi") & (h["Level"] > 1)).sum())
    n_s = int(((h["Risiko"] == "Sedang") & (h["Level"] > 1)).sum())
    kartu_html.append(kartu("Akun perlu perhatian", f"{n_t} + {n_s}",
                            f'<span style="color:{WARNA["Tinggi"]}">{n_t} tinggi</span> · '
                            f'<span style="color:{WARNA["Sedang"]}">{n_s} sedang</span>'))
    st.markdown(f'<div class="kartu-baris">{"".join(kartu_html)}</div>', unsafe_allow_html=True)
    panduan_membaca(periode)

    # --- Langkah 2: hasil perhitungan sistem
    st.subheader("Langkah 2 · Hasil perhitungan sistem")
    n_anom = len(an.cari_anomali(semua))
    t1, t5, t6, t7, t2, t3, t4 = st.tabs(["Pengukuran risiko", f"Anomali ({n_anom})", "Tren bulanan",
                                          "Rasio & kepatuhan", "Grafik", "Tabel lengkap", "Perbandingan antar pemda"])
    with t1:
        daftar_temuan(h, nama)
    with t5:
        daftar_anomali(semua, fokus=nama)
    with t6:
        proyeksi = bagian_tren(kunci, kode, tahun, periode)
    with t7:
        if rasio is None:
            st.info("Rasio tidak dapat dihitung.")
        else:
            bagian_rasio(rasio, nama)
            if rasio_lain is not None:
                st.markdown("#### Dibandingkan dengan pemda lain se-provinsi")
                bagian_rasio_lintas(rasio_lain, nama)
    with t2:
        st.caption("Batang = realisasi tiap akun (warna = tingkat risiko). Garis abu-abu = rata-rata se-provinsi. "
                   "Garis biru putus-putus = target waktu.")
        for kel in ["Pendapatan", "Belanja"]:
            st.markdown(f"**{kel}**")
            st.plotly_chart(grafik_kelompok(h, kel, periode), width="stretch", key=f"g_{kel}")
    with t3:
        tabel_lengkap(h)
        st.caption("Unduh semua hasil dalam satu file Excel di bagian Langkah 4 (paling bawah).")
    with t4:
        if peringkat is None or peringkat.empty:
            st.info("Centang 'Semua pemda se-provinsi' di sidebar untuk melihat perbandingan.")
        else:
            st.caption("Diurutkan dari realisasi belanja terendah. Batang merah = pemda yang dipilih.")
            st.plotly_chart(grafik_peringkat(peringkat, nama, periode), width="stretch")
            tabel_peringkat(peringkat)

    # --- Langkah 3: AI (konteks disusun setelah semua hitungan, termasuk rasio & tren)
    konteks = an.ringkasan_untuk_ai(nama, tahun, periode, h, peringkat, keterangan)
    if rasio is not None:
        konteks += an.ringkasan_rasio_ai(rasio)
    konteks += an.ringkasan_tren_ai(proyeksi or [])
    konteks += an.ringkasan_lintas_wilayah(semua, fokus=nama)
    st.divider()
    bagian_ai(konteks, kunci, model, semua)

    # --- Langkah 4: kertas kerja
    st.divider()
    info = {"Aplikasi": "SIAGA-APBD · AI Agent pemantauan realisasi APBD", "Wilayah": nama, "Tahun anggaran": tahun,
            "Realisasi sampai": BULAN[periode - 1], "Target waktu (%)": round(waktu, 1),
            "Batas tertinggal (poin)": ambang, "Akun risiko tinggi": n_t, "Akun risiko sedang": n_s,
            "Anomali (semua wilayah yang diambil)": n_anom, "Sumber data": f"Portal Data SIKD DJPK Kemenkeu ({sumber})",
            "Diunduh": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "Catatan": "Hasil adalah peringatan dini untuk perencanaan pemeriksaan, bukan bukti penyimpangan. "
                       "Catatan AI wajib diverifikasi auditor."}
    anomali = an.cari_anomali(semua)
    lembar = {
        "Temuan risiko": tabel_temuan(h),
        "Semua akun": tabel_semua(h),
        "Anomali": tabel_anomali(anomali),
        "Rasio & kepatuhan": rasio.drop(columns=["Kunci"]) if rasio is not None else None,
        "Rasio antar pemda": rasio_lain,
        "Perbandingan": peringkat_excel(peringkat),
        "Proyeksi akhir tahun": pd.DataFrame(proyeksi) if proyeksi else None,
        "Catatan AI": lembar_ai(kunci),
    }
    bagian_unduh(info, lembar, f"Kertas_kerja_SIAGA-APBD_{nama}_{tahun}-{periode:02d}.xlsx".replace(" ", "_"))
    alur_agent(f"Portal Data SIKD DJPK Kemenkeu ({sumber})", model, ai_ok(kunci), wadah_alur)


def tampil_multi(judul, tahun, periode, data, analisis, semua, model, kunci, ambang=15):
    """Mode Provinsi / Nasional: peringkat antar wilayah + anomali + rasio + jumlah akun berisiko tinggi."""
    waktu = periode / 12 * 100
    peringkat = an.tabel_peringkat(data, periode)
    jml = {nama: int(((h["Risiko"] == "Tinggi") & (h["Level"] > 1)).sum()) for nama, h in analisis.items()}
    peringkat["Akun risiko tinggi"] = peringkat["Wilayah"].map(jml)
    rl = an.rasio_lintas(data)
    konteks = an.ringkasan_untuk_ai(judul, tahun, periode, None, peringkat)
    konteks += "\n\n## Jumlah akun berisiko tinggi per wilayah\n" + "\n".join(f"- {k}: {v}" for k, v in jml.items())
    if not rl.empty:
        konteks += "\n\n## Rasio per wilayah (dari APBD): kemandirian | belanja pegawai (maks 30%) | belanja modal (min 40%)\n"
        konteks += "\n".join(f"- {r['Wilayah']}: {an.pct(r['Kemandirian fiskal %'])} | {an.pct(r['Belanja pegawai %'])} "
                             f"({r['Status pegawai (maks 30%)']}) | {an.pct(r['Belanja modal (proksi infrastruktur) %'])} "
                             f"({r['Status modal (min 40%)']})" for _, r in rl.iterrows())
    konteks += an.ringkasan_lintas_wilayah(semua)

    st.markdown(f"## {judul}")
    st.markdown(f'<div class="subjudul">Perbandingan realisasi APBD s.d. {BULAN[periode - 1]} {tahun}</div>',
                unsafe_allow_html=True)
    wadah_alur = st.empty()

    terendah = peringkat.iloc[0]
    st.markdown('<div class="kartu-baris">' + "".join([
        kartu("Wilayah dibandingkan", str(len(data))),
        kartu("Target waktu", an.pct(waktu), f"{periode} dari 12 bulan berjalan"),
        kartu("Realisasi belanja terendah", html.escape(terendah["Wilayah"]),
              f'{an.pct(terendah["Belanja Daerah"])} · {an.selisih_kata(terendah["Belanja Daerah"] - waktu)}',
              WARNA["Tinggi"], kecil=True),
        kartu("Total belanja belum terealisasi", an.rp(peringkat["Sisa Belanja"].sum()), kecil=True),
    ]) + "</div>", unsafe_allow_html=True)
    panduan_membaca(periode)

    st.subheader("Langkah 2 · Hasil perhitungan sistem")
    n_anom = len(an.cari_anomali(semua))
    t1, t3, t4, t2 = st.tabs(["Peringkat wilayah", f"Anomali ({n_anom})", "Rasio & kepatuhan", "Detail per wilayah"])
    with t3:
        daftar_anomali(semua)
    with t1:
        st.caption("Diurutkan dari realisasi belanja terendah. Batang = realisasi belanja daerah; "
                   "titik putih = realisasi belanja modal; garis biru = target waktu.")
        st.plotly_chart(grafik_peringkat(peringkat, None, periode), width="stretch")
        tabel_peringkat(peringkat)
    with t4:
        st.markdown("""
<div class="kepala-bagian">
  <div class="kb-label">Rasio keuangan &amp; kepatuhan antar wilayah</div>
  <div class="kb-teks">Membandingkan <b>komposisi</b> APBD semua wilayah: kemandirian fiskal (PAD ÷ pendapatan),
  porsi belanja pegawai terhadap batas <b>maksimal 30%</b> (UU HKPD Pasal 146), dan porsi belanja modal sebagai
  pendekatan belanja infrastruktur terhadap batas <b>minimal 40%</b> (Pasal 147). Dihitung dari APBD (anggaran).</div>
</div>""", unsafe_allow_html=True)
        bagian_rasio_lintas(rl)
    with t2:
        pilih = st.selectbox("Pilih wilayah", list(peringkat["Wilayah"]))
        h = analisis[pilih]
        daftar_temuan(h, pilih)
        if pilih in data:
            with st.expander("Rasio keuangan wilayah ini"):
                bagian_rasio(an.rasio_keuangan(data[pilih]), pilih)
        with st.expander("Tabel lengkap"):
            tabel_lengkap(h)

    st.divider()
    bagian_ai(konteks, kunci, model, semua)

    st.divider()
    berisiko = semua[(semua["Risiko"] != "Rendah") & (semua["Level"] > 1)]
    info = {"Aplikasi": "SIAGA-APBD · AI Agent pemantauan realisasi APBD", "Cakupan": judul, "Tahun anggaran": tahun,
            "Realisasi sampai": BULAN[periode - 1], "Target waktu (%)": round(waktu, 1),
            "Batas tertinggal (poin)": ambang, "Wilayah dibandingkan": len(data), "Anomali": n_anom,
            "Sumber data": "Portal Data SIKD DJPK Kemenkeu", "Diunduh": datetime.now().strftime("%d-%m-%Y %H:%M"),
            "Catatan": "Hasil adalah peringatan dini untuk perencanaan pemeriksaan, bukan bukti penyimpangan. "
                       "Catatan AI wajib diverifikasi auditor."}
    lembar = {"Peringkat": peringkat_excel(peringkat), "Rasio & kepatuhan": rl, "Anomali": tabel_anomali(an.cari_anomali(semua)),
              "Akun berisiko": tabel_temuan(berisiko, wilayah=True) if not berisiko.empty else None,
              "Catatan AI": lembar_ai(kunci)}
    bagian_unduh(info, lembar, f"Kertas_kerja_SIAGA-APBD_{judul}_{tahun}-{periode:02d}.xlsx".replace(" ", "_"))
    alur_agent("Portal Data SIKD DJPK Kemenkeu", model, ai_ok(kunci), wadah_alur)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("SIAGA-APBD")
st.markdown('<div class="subjudul">AI Agent pemantauan realisasi APBD dan catatan audit berbasis risiko '
            '· Studi kasus Kabupaten Bengkalis</div>', unsafe_allow_html=True)

sb = st.sidebar
sekarang = datetime.now()
sb.markdown("### Wilayah dan periode")
sumber = sb.radio("Sumber data", ["Ambil otomatis dari DJPK", "Upload file ekspor DJPK"],
                  label_visibility="collapsed")
tahun = sb.selectbox("Tahun anggaran", list(range(sekarang.year + 1, 2014, -1)), index=1)
periode = sb.selectbox("Realisasi sampai bulan", range(1, 13), index=sekarang.month - 1,
                       format_func=lambda b: BULAN[b - 1])

if sumber == "Ambil otomatis dari DJPK":
    mode = sb.radio("Cakupan", ["Satu pemda", "Satu provinsi", "Nasional (antar provinsi)"])
    prov_map = ambil_provinsi(tahun)
    kode_prov = list(prov_map.keys())
    if mode != "Nasional (antar provinsi)":
        prov = sb.selectbox("Provinsi", kode_prov, index=kode_prov.index("04") if "04" in kode_prov else 0,
                            format_func=lambda k: prov_map[k])
    if mode == "Satu pemda":
        pemda_map = ambil_pemda(prov, tahun)
        kode_pemda = list(pemda_map.keys())
        if not kode_pemda:
            sb.error("Daftar pemda tidak dapat diambil.")
            st.stop()
        pemda = sb.selectbox("Pemda", kode_pemda, index=kode_pemda.index("01") if "01" in kode_pemda else 0,
                             format_func=lambda k: pemda_map[k])
        sb.markdown("### Pembanding")
        cek_tl = sb.checkbox("Bulan yang sama tahun lalu", True)
        cek_peer = sb.checkbox("Semua pemda se-provinsi", True)
else:
    sb.caption("File .xls dari tombol Excel di portal DJPK, atau Excel/CSV berkolom Akun, Anggaran, Realisasi.")
    nama_up = sb.text_input("Nama pemda", "Kab. Bengkalis")
    f_now = sb.file_uploader("File realisasi (wajib)", type=["xls", "xlsx", "csv"])
    f_tl = sb.file_uploader("File bulan yang sama tahun lalu (opsional)", type=["xls", "xlsx", "csv"])
    f_pb = sb.file_uploader("File gabungan se-provinsi (opsional)", type=["xls", "xlsx", "csv"])

with sb.expander("Pengaturan lanjutan"):
    ambang = st.slider("Batas 'tertinggal' dari target waktu (poin)", 5, 30, 15,
                       help="Akun dianggap tertinggal jika realisasinya lebih rendah sekian poin dari target waktu. "
                            "Tertinggal 2x batas ini mendapat skor lebih berat.")
    model = st.selectbox("Model AI (Groq)", MODEL_GROQ)

jalan = sb.button("Jalankan analisis", type="primary", width="stretch",
                  disabled=(sumber != "Ambil otomatis dari DJPK" and f_now is None))
sb.caption("Sumber data: Portal Data SIKD DJPK Kemenkeu. Hasil analisis adalah peringatan dini, "
           "bukan bukti penyimpangan.")

# ---------------------------------------------------------------------------
# Proses
# ---------------------------------------------------------------------------
if jalan:
    st.session_state.pop("hasil", None)
    try:
        if sumber != "Ambil otomatis dari DJPK":
            df_now = baca_upload(f_now)
            df_tl = baca_upload(f_tl) if f_tl else None
            df_pb = baca_upload(f_pb) if f_pb else None
            h = an.analisis_pemda(df_now, periode, df_tl, df_pb, ambang)
            st.session_state["hasil"] = dict(jenis="pemda", nama=nama_up, h=h, peringkat=None,
                                             keterangan="Data dari file upload.", sumber="file upload",
                                             semua=an.gabung_hasil({nama_up: h}), df_now=df_now, df_pb=df_pb,
                                             data=None, kode=None)
        elif mode == "Satu pemda":
            with st.spinner(f"Mengambil data {pemda_map[pemda]} dari DJPK..."):
                df_now, info = ambil_aman(tahun, periode, prov, pemda)
            if df_now is None:
                st.error(f"Gagal mengambil data dari DJPK: {info}\n\nCoba lagi nanti, atau pilih "
                         "'Upload file ekspor DJPK'.")
                st.stop()
            _, _, ket = ambil_apbd(tahun, periode, prov, pemda)
            daftar = [("__agregat__", tahun, periode, prov, "--")]
            if cek_tl:
                daftar.append(("__tahun_lalu__", tahun - 1, periode, prov, pemda))
            if cek_peer:
                daftar += [(pemda_map[k], tahun, periode, prov, k) for k in pemda_map if k != pemda]
            lain = ambil_banyak(daftar, "Mengambil data pembanding")
            agregat = lain.pop("__agregat__", None)
            h = an.analisis_pemda(df_now, periode, lain.pop("__tahun_lalu__", None), agregat, ambang)
            # analisis juga pemda lain, agar anomali & pencarian AI mencakup seluruh provinsi
            analisis = {pemda_map[pemda]: h}
            analisis.update({w: an.analisis_pemda(df, periode, df_pembanding=agregat, ambang_waktu=ambang)
                             for w, df in lain.items()})
            peringkat = None
            lain[pemda_map[pemda]] = df_now
            if cek_peer:
                peringkat = an.tabel_peringkat(lain, periode)
            st.session_state["hasil"] = dict(jenis="pemda", nama=pemda_map[pemda], h=h, peringkat=peringkat,
                                             keterangan=ket, sumber=info, semua=an.gabung_hasil(analisis),
                                             df_now=df_now, df_pb=agregat, data=lain if cek_peer else None,
                                             kode=(prov, pemda))
        elif mode == "Satu provinsi":
            pemda_map = ambil_pemda(prov, tahun)
            daftar = [(v, tahun, periode, prov, k) for k, v in pemda_map.items()]
            daftar.append(("__agregat__", tahun, periode, prov, "--"))
            data = ambil_banyak(daftar, f"Mengambil data pemda di {prov_map[prov]}")
            agregat = data.pop("__agregat__", None)
            if not data:
                st.error("Tidak ada data yang berhasil diambil.")
                st.stop()
            analisis = {w: an.analisis_pemda(df, periode, df_pembanding=agregat, ambang_waktu=ambang)
                        for w, df in data.items()}
            st.session_state["hasil"] = dict(jenis="multi", judul=prov_map[prov], data=data, analisis=analisis,
                                             semua=an.gabung_hasil(analisis))
        else:
            daftar = [(v.replace("Provinsi ", "Se-Prov. "), tahun, periode, k, "--") for k, v in prov_map.items()]
            data = ambil_banyak(daftar, "Mengambil data gabungan 38 provinsi")
            if not data:
                st.error("Tidak ada data yang berhasil diambil.")
                st.stop()
            analisis = {w: an.analisis_pemda(df, periode, ambang_waktu=ambang) for w, df in data.items()}
            st.session_state["hasil"] = dict(jenis="multi", judul="Nasional (gabungan per provinsi)",
                                             data=data, analisis=analisis, semua=an.gabung_hasil(analisis))
        st.session_state["param"] = (tahun, periode)
        if "hasil" in st.session_state:
            st.session_state["hasil"]["ambang"] = ambang
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ---------------------------------------------------------------------------
# Tampilkan hasil (disimpan di session agar tanya-jawab tidak mengambil ulang data)
# ---------------------------------------------------------------------------
H = st.session_state.get("hasil")
if H is None:
    alur_agent("Portal Data SIKD DJPK Kemenkeu", MODEL_GROQ[0], False)
    st.markdown("""
Aplikasi ini membantu auditor internal pemerintah melihat **pos anggaran mana yang realisasinya tertinggal
atau tidak wajar**, lalu meminta AI menuliskan catatan audit dan langkah pemeriksaan yang disarankan.

**Mulai:** pilih wilayah dan bulan di panel kiri, lalu klik **Jalankan analisis**.
""")
else:
    t_, p_ = st.session_state["param"]
    if H["jenis"] == "pemda":
        tampil_pemda(H["nama"], t_, p_, H["h"], H["peringkat"], H["keterangan"], model, H["sumber"], H["semua"],
                     H.get("df_now"), H.get("df_pb"), H.get("data"), H.get("kode"), H.get("ambang", 15))
    else:
        tampil_multi(H["judul"], t_, p_, H["data"], H["analisis"], H["semua"], model, f"multi|{H['judul']}|{t_}|{p_}",
                     H.get("ambang", 15))
