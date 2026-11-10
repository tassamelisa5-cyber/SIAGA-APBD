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
data realisasi APBD berikut. Jawab dalam Bahasa Indonesia sederhana, ringkas, tanpa emoji, hanya
berdasarkan data yang diberikan. Jika data tidak cukup, katakan terus terang. Jangan mengarang angka."""

CONTOH_PERTANYAAN = [
    "Akun mana yang paling perlu diperiksa?",
    "Apa risikonya jika belanja menumpuk di Desember?",
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


def pesan_galat(e):
    t = str(e)
    if "429" in t or "rate" in t.lower():
        return "Batas pemakaian Groq per menit tercapai. Tunggu sekitar 1 menit lalu klik 'Tulis ulang'."
    if "401" in t or "invalid_api_key" in t.lower():
        return "API key Groq tidak valid. Periksa GROQ_API_KEY di Secrets."
    if "model" in t.lower() and ("not found" in t.lower() or "decommission" in t.lower()):
        return "Model tidak tersedia. Pilih model lain di Pengaturan lanjutan."
    return f"Gagal memanggil Groq: {t}"


def bagian_ai(konteks, kunci, model):
    """Catatan audit (ditulis otomatis sekali per analisis) + tanya-jawab."""
    st.subheader("Langkah 3 · Catatan audit dari AI")
    st.caption(f"Ditulis oleh AI (Groq · {model}) berdasarkan angka hasil langkah 2. "
               "AI tidak menghitung ulang angka; ia menafsirkan dan menyusun catatan.")
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
                st.markdown(m["content"])
    with st.container():  # di dalam container -> kolom chat tampil di tempat, halaman tidak melompat ke bawah
        q = st.chat_input("Tulis pertanyaan Anda...") or usul
    if q:
        chat.append({"role": "user", "content": q})
        with kotak:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("AI sedang menjawab..."):
                    try:
                        msgs = [{"role": "user", "content": f"DATA:\n{konteks}"},
                                {"role": "assistant", "content": "Baik, data sudah saya pahami."}] + chat[-10:]
                        jawab = tanya_groq(SYSTEM_CHAT, msgs, model)
                    except Exception as e:
                        jawab = pesan_galat(e)
                st.markdown(jawab)
        chat.append({"role": "assistant", "content": jawab})


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
    <div class="ket">Groq · {html.escape(model)}</div>
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
<span style="opacity:.75;font-size:.9rem">Dihitung sistem dari: seberapa jauh tertinggal dari target waktu, apakah di bawah
rata-rata se-provinsi atau tahun lalu, apakah nilainya besar, dan apakah ada realisasi tanpa anggaran.
Risiko adalah tanda untuk diperiksa, bukan bukti penyimpangan.</span></td></tr>
</table>""", unsafe_allow_html=True)


def daftar_temuan(h):
    r = h[(h["Risiko"] != "Rendah") & (h["Level"] > 1)].sort_values(["Skor", "Sisa"], ascending=False)
    if r.empty:
        st.success("Tidak ada akun dengan risiko sedang atau tinggi.")
        return
    baris = []
    for _, x in r.iterrows():
        rinci = (f"Realisasi {an.pct(x['Real%'])} dari anggaran {an.rp(x['Anggaran'])} "
                 f"(belum terealisasi {an.rp(x['Sisa'])}). Alasan: {x['Alasan']}.")
        if x["Anggaran"] == 0:
            rinci = f"Anggaran Rp0 tetapi realisasi {an.rp(x['Realisasi'])}. Alasan: {x['Alasan']}."
        baris.append(f'<div class="temuan">{lencana(x["Risiko"])}<span class="akun">{html.escape(x["Akun"])}</span>'
                     f'<span style="opacity:.6;font-size:.85rem"> · {x["Kelompok"]}</span>'
                     f'<div class="rinci">{html.escape(rinci)}</div></div>')
    tampil = 6
    st.caption(f"{len(baris)} akun berisiko sedang/tinggi, diurutkan dari yang paling berisiko.")
    st.markdown("".join(baris[:tampil]), unsafe_allow_html=True)
    if len(baris) > tampil:
        with st.expander(f"Lihat {len(baris) - tampil} akun lainnya"):
            st.markdown("".join(baris[tampil:]), unsafe_allow_html=True)


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
def tampil_pemda(nama, tahun, periode, h, peringkat, keterangan, model, sumber):
    waktu = periode / 12 * 100
    idx = h.set_index("Akun")
    kunci = f"pemda|{nama}|{tahun}|{periode}"
    konteks = an.ringkasan_untuk_ai(nama, tahun, periode, h, peringkat, keterangan)

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
    t1, t2, t3, t4 = st.tabs(["Akun yang perlu perhatian", "Grafik", "Tabel lengkap", "Perbandingan antar pemda"])
    with t1:
        daftar_temuan(h)
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
    bagian_ai(konteks, kunci, model)
    alur_agent(f"Portal Data SIKD DJPK Kemenkeu ({sumber})", model, ai_ok(kunci), wadah_alur)


def tampil_multi(judul, tahun, periode, data, agregat, model, kunci):
    """Mode Provinsi / Nasional: peringkat antar wilayah + jumlah akun berisiko tinggi."""
    waktu = periode / 12 * 100
    peringkat = an.tabel_peringkat(data, periode)
    jml = {}
    for nama, df in data.items():
        h = an.analisis_pemda(df, periode, df_pembanding=agregat)
        jml[nama] = int(((h["Risiko"] == "Tinggi") & (h["Level"] > 1)).sum())
    peringkat["Akun risiko tinggi"] = peringkat["Wilayah"].map(jml)
    konteks = an.ringkasan_untuk_ai(judul, tahun, periode, None, peringkat)
    konteks += "\n\n## Jumlah akun berisiko tinggi per wilayah\n" + "\n".join(f"- {k}: {v}" for k, v in jml.items())

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
    t1, t2 = st.tabs(["Peringkat wilayah", "Detail per wilayah"])
    with t1:
        st.caption("Diurutkan dari realisasi belanja terendah. Batang = realisasi belanja daerah; "
                   "titik gelap = realisasi belanja modal; garis biru = target waktu.")
        st.plotly_chart(grafik_peringkat(peringkat, None, periode), width="stretch")
        tabel_peringkat(peringkat)
        st.download_button("Unduh peringkat (.xlsx)", unduh_excel({"Peringkat": peringkat}),
                           file_name=f"SIAGA-APBD_peringkat_{tahun}-{periode:02d}.xlsx")
    with t2:
        pilih = st.selectbox("Pilih wilayah", list(peringkat["Wilayah"]))
        h = an.analisis_pemda(data[pilih], periode, df_pembanding=agregat)
        daftar_temuan(h)
        with st.expander("Tabel lengkap"):
            tabel_lengkap(h)

    st.divider()
    bagian_ai(konteks, kunci, model)
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
                                             keterangan="Data dari file upload.", sumber="file upload")
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
            h = an.analisis_pemda(df_now, periode, lain.pop("__tahun_lalu__", None),
                                  lain.pop("__agregat__", None), ambang)
            peringkat = None
            if cek_peer:
                lain[pemda_map[pemda]] = df_now
                peringkat = an.tabel_peringkat(lain, periode)
            st.session_state["hasil"] = dict(jenis="pemda", nama=pemda_map[pemda], h=h, peringkat=peringkat,
                                             keterangan=ket, sumber=info)
        elif mode == "Satu provinsi":
            pemda_map = ambil_pemda(prov, tahun)
            daftar = [(v, tahun, periode, prov, k) for k, v in pemda_map.items()]
            daftar.append(("__agregat__", tahun, periode, prov, "--"))
            data = ambil_banyak(daftar, f"Mengambil data pemda di {prov_map[prov]}")
            agregat = data.pop("__agregat__", None)
            if not data:
                st.error("Tidak ada data yang berhasil diambil.")
                st.stop()
            st.session_state["hasil"] = dict(jenis="multi", judul=prov_map[prov], data=data, agregat=agregat)
        else:
            daftar = [(v.replace("Provinsi ", "Se-Prov. "), tahun, periode, k, "--") for k, v in prov_map.items()]
            data = ambil_banyak(daftar, "Mengambil data gabungan 38 provinsi")
            if not data:
                st.error("Tidak ada data yang berhasil diambil.")
                st.stop()
            st.session_state["hasil"] = dict(jenis="multi", judul="Nasional (gabungan per provinsi)",
                                             data=data, agregat=None)
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
        tampil_pemda(H["nama"], t_, p_, H["h"], H["peringkat"], H["keterangan"], model, H["sumber"])
    else:
        tampil_multi(H["judul"], t_, p_, H["data"], H["agregat"], model, f"multi|{H['judul']}|{t_}|{p_}")
