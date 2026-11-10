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

import analysis as an
import djpk_client as djpk

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIAGA-APBD", layout="wide")

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]

WARNA = {"Tinggi": "#e5484d", "Sedang": "#f08c00", "Rendah": "#2f9e74"}
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
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
.subjudul {opacity: .75; margin-top: .2rem; margin-bottom: 1.2rem;}
.alur {display: flex; gap: .6rem; margin: .4rem 0 1.4rem 0; flex-wrap: wrap;}
.langkah {flex: 1; min-width: 200px; border: 1px solid rgba(128,128,128,.3); border-radius: 10px;
          padding: .7rem .9rem;}
.langkah .no {font-size: .75rem; letter-spacing: .04em; text-transform: uppercase; opacity: .65;}
.langkah .judul {font-weight: 600; margin: .1rem 0;}
.langkah .ket {font-size: .85rem; opacity: .75;}
.langkah.ai {border-color: #4c6ef5; background: rgba(76,110,245,.08);}
.status {font-size: .8rem; font-weight: 600;}
.kartu-baris {display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: .7rem;
              margin: .3rem 0 1rem 0;}
.kartu {border: 1px solid rgba(128,128,128,.3); border-radius: 10px; padding: .8rem 1rem;}
.kartu .label {font-size: .85rem; opacity: .75;}
.kartu .nilai {font-size: 1.7rem; font-weight: 650; line-height: 1.25; margin: .15rem 0;}
.kartu .nilai.kecil {font-size: 1.15rem;}
.kartu .catatan {font-size: .85rem;}
.lencana {display: inline-block; font-weight: 700; font-size: .8rem; padding: .1rem .5rem;
          border-radius: 999px; border: 1.5px solid currentColor; margin-right: .4rem; white-space: nowrap;}
.temuan {border-bottom: 1px solid rgba(128,128,128,.2); padding: .55rem 0;}
.temuan .akun {font-weight: 600;}
.temuan .rinci {font-size: .9rem; opacity: .85; margin-top: .15rem;}
.kepala-bagian {border-left: 4px solid #4c6ef5; padding: .3rem 0 .3rem .9rem; margin: .4rem 0 1rem 0;}
.kb-label {font-size: .85rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #4c6ef5;}
.kb-teks {font-size: .95rem; margin-top: .25rem; line-height: 1.65;}
.kartu-temuan {border: 1px solid rgba(128,128,128,.28); border-left: 4px solid; border-radius: 8px;
               padding: .8rem 1rem; margin-bottom: .8rem;}
.kt-kepala .akun {font-weight: 650; font-size: 1.05rem;}
.kt-inti {margin: .35rem 0 .55rem 0; font-weight: 500;}
.kt-isi {display: grid; grid-template-columns: 150px 1fr; gap: .3rem .9rem; font-size: .92rem;}
.kt-isi .b-label {opacity: .65;}
.kt-isi ul {margin: 0; padding-left: 1.1rem;}
@media (max-width: 640px) {.kt-isi {grid-template-columns: 1fr;} .kt-isi .b-label {margin-top: .3rem;}}
.kotak-ai {border: 1px solid #4c6ef5; border-radius: 10px; padding: .2rem 1.2rem 1rem 1.2rem;
           background: rgba(76,110,245,.05);}
.kotak-ai .kepala {font-size: .8rem; opacity: .75; margin: .7rem 0 .2rem 0;}
.panduan td {padding: .25rem .6rem .25rem 0; vertical-align: top;}
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
def tanya_groq(system, messages, model):
    from groq import Groq

    client = Groq(api_key=get_api_key())
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + messages,
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
        resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2,
                                              tools=ALAT_AI, tool_choice="auto")
        m = resp.choices[0].message
        if not getattr(m, "tool_calls", None):
            return m.content, log
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
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
    return resp.choices[0].message.content, log


def pesan_galat(e):
    t = str(e)
    if "429" in t or "rate" in t.lower():
        return "Batas pemakaian Groq per menit tercapai. Tunggu sekitar 1 menit lalu klik 'Tulis ulang'."
    if "401" in t or "invalid_api_key" in t.lower():
        return "API key Groq tidak valid. Periksa GROQ_API_KEY di Secrets."
    if "model" in t.lower() and ("not found" in t.lower() or "decommission" in t.lower()):
        return "Model tidak tersedia. Pilih model lain di Pengaturan lanjutan."
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
    <div class="status" style="color:#4c6ef5">{ket_ai}</div></div>
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


def kartu_temuan(x):
    """Satu akun berisiko, dijelaskan rinci dan terstruktur."""
    waktu = x["Waktu%"]
    pendapatan = x["Kelompok"] == "Pendapatan"
    kata = "diterima" if pendapatan else "dibelanjakan"
    arti, saran = KAMUS_AKUN.get(x["Akun"], ("", ""))

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
    saran_all = [SARAN_KHUSUS[j] for j in SARAN_KHUSUS if j in x["Jenis"]] + ([saran] if saran else [])
    if saran_all:
        isi.append(_baris("Saran pemeriksaan", html.escape(" ".join(saran_all))))

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
                                 marker=dict(symbol="line-ns", size=16, line=dict(width=3, color="#868e96"))))
    fig.add_vline(x=waktu, line_dash="dash", line_color="#4c6ef5")
    fig.add_annotation(x=waktu, y=1.02, yref="paper", text=f"Target waktu {an.pct(waktu)}", showarrow=False,
                       font=dict(color="#4c6ef5"), xanchor="left")
    fig.update_layout(height=60 + 34 * len(d), margin=dict(l=10, r=40, t=30, b=10),
                      xaxis=dict(title="Realisasi (% dari anggaran)", range=[0, max(110, d["Real%"].max() + 15)]),
                      yaxis=dict(autorange="reversed", title=""), legend=dict(orientation="h", y=-0.15),
                      bargap=0.35)
    return fig


def grafik_peringkat(p, fokus, periode):
    waktu = periode / 12 * 100
    warna = ["#e5484d" if w == fokus else "#74a9cf" for w in p["Wilayah"]]
    fig = go.Figure(go.Bar(x=p["Belanja Daerah"], y=p["Wilayah"], orientation="h", marker_color=warna,
                           text=[an.pct(v) for v in p["Belanja Daerah"]], textposition="outside",
                           cliponaxis=False, name="Realisasi belanja daerah"))
    fig.add_trace(go.Scatter(x=p["Belanja Modal"], y=p["Wilayah"], mode="markers", name="Realisasi belanja modal",
                             marker=dict(symbol="circle", size=9, color="#343a40",
                                         line=dict(width=1, color="white"))))
    fig.add_vline(x=waktu, line_dash="dash", line_color="#4c6ef5")
    fig.add_annotation(x=waktu, y=1.02, yref="paper", text=f"Target waktu {an.pct(waktu)}", showarrow=False,
                       font=dict(color="#4c6ef5"), xanchor="left")
    fig.update_layout(height=80 + 32 * len(p), margin=dict(l=10, r=40, t=30, b=10),
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
# Halaman hasil
# ---------------------------------------------------------------------------
def tampil_pemda(nama, tahun, periode, h, peringkat, keterangan, model, sumber, semua):
    waktu = periode / 12 * 100
    idx = h.set_index("Akun")
    kunci = f"pemda|{nama}|{tahun}|{periode}"
    konteks = an.ringkasan_untuk_ai(nama, tahun, periode, h, peringkat, keterangan)
    konteks += an.ringkasan_lintas_wilayah(semua, fokus=nama)

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
    t1, t5, t2, t3, t4 = st.tabs(["Pengukuran risiko", f"Anomali ({n_anom})", "Grafik", "Tabel lengkap",
                                  "Perbandingan antar pemda"])
    with t1:
        daftar_temuan(h, nama)
    with t5:
        daftar_anomali(semua, fokus=nama)
    with t2:
        st.caption("Batang = realisasi tiap akun (warna = tingkat risiko). Garis abu-abu = rata-rata se-provinsi. "
                   "Garis biru putus-putus = target waktu.")
        for kel in ["Pendapatan", "Belanja"]:
            st.markdown(f"**{kel}**")
            st.plotly_chart(grafik_kelompok(h, kel, periode), width="stretch", key=f"g_{kel}")
    with t3:
        tabel_lengkap(h)
        lembar = {"Analisis": h}
        if peringkat is not None and not peringkat.empty:
            lembar["Perbandingan"] = peringkat
        st.download_button("Unduh hasil analisis (.xlsx)", unduh_excel(lembar),
                           file_name=f"SIAGA-APBD_{nama}_{tahun}-{periode:02d}.xlsx")
    with t4:
        if peringkat is None or peringkat.empty:
            st.info("Centang 'Bandingkan dengan pemda se-provinsi' di sidebar untuk melihat perbandingan.")
        else:
            st.caption("Diurutkan dari realisasi belanja terendah. Batang merah = pemda yang dipilih.")
            st.plotly_chart(grafik_peringkat(peringkat, nama, periode), width="stretch")
            tabel_peringkat(peringkat)

    st.divider()
    bagian_ai(konteks, kunci, model, semua)
    alur_agent(f"Portal Data SIKD DJPK Kemenkeu ({sumber})", model, ai_ok(kunci), wadah_alur)


def tampil_multi(judul, tahun, periode, data, analisis, semua, model, kunci):
    """Mode Provinsi / Nasional: peringkat antar wilayah + anomali + jumlah akun berisiko tinggi."""
    waktu = periode / 12 * 100
    peringkat = an.tabel_peringkat(data, periode)
    jml = {nama: int(((h["Risiko"] == "Tinggi") & (h["Level"] > 1)).sum()) for nama, h in analisis.items()}
    peringkat["Akun risiko tinggi"] = peringkat["Wilayah"].map(jml)
    konteks = an.ringkasan_untuk_ai(judul, tahun, periode, None, peringkat)
    konteks += "\n\n## Jumlah akun berisiko tinggi per wilayah\n" + "\n".join(f"- {k}: {v}" for k, v in jml.items())
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
    t1, t3, t2 = st.tabs(["Peringkat wilayah", f"Anomali ({n_anom})", "Detail per wilayah"])
    with t3:
        daftar_anomali(semua)
    with t1:
        st.caption("Diurutkan dari realisasi belanja terendah. Batang = realisasi belanja daerah; "
                   "titik gelap = realisasi belanja modal; garis biru = target waktu.")
        st.plotly_chart(grafik_peringkat(peringkat, None, periode), width="stretch")
        tabel_peringkat(peringkat)
        st.download_button("Unduh peringkat (.xlsx)", unduh_excel({"Peringkat": peringkat}),
                           file_name=f"SIAGA-APBD_peringkat_{tahun}-{periode:02d}.xlsx")
    with t2:
        pilih = st.selectbox("Pilih wilayah", list(peringkat["Wilayah"]))
        h = analisis[pilih]
        daftar_temuan(h, pilih)
        with st.expander("Tabel lengkap"):
            tabel_lengkap(h)

    st.divider()
    bagian_ai(konteks, kunci, model, semua)
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
                                             semua=an.gabung_hasil({nama_up: h}))
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
            if cek_peer:
                lain[pemda_map[pemda]] = df_now
                peringkat = an.tabel_peringkat(lain, periode)
            st.session_state["hasil"] = dict(jenis="pemda", nama=pemda_map[pemda], h=h, peringkat=peringkat,
                                             keterangan=ket, sumber=info, semua=an.gabung_hasil(analisis))
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
        tampil_pemda(H["nama"], t_, p_, H["h"], H["peringkat"], H["keterangan"], model, H["sumber"], H["semua"])
    else:
        tampil_multi(H["judul"], t_, p_, H["data"], H["analisis"], H["semua"], model, f"multi|{H['judul']}|{t_}|{p_}")
