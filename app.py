"""
SIAGA-APBD
AI Agent Pemantauan Realisasi APBD & Catatan Audit Berbasis Risiko
Studi kasus: Kabupaten Bengkalis, Provinsi Riau

Alur agent:
  1. SENSING  : mengambil data APBD otomatis dari Portal Data SIKD DJPK Kemenkeu
                (atau dari file ekspor yang di-upload sebagai cadangan)
  2. THINKING : Python menghitung realisasi, membandingkan dengan waktu berjalan,
                pemda lain, dan tahun lalu, lalu memberi skor risiko per akun
  3. ACTING   : AI (Groq) menulis catatan audit & menjawab pertanyaan pengguna

File pendukung: djpk_client.py (pengambil data), analysis.py (perhitungan)
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analysis as an
import djpk_client as djpk

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SIAGA-APBD", page_icon="🛡️", layout="wide")

BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]

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
- Gunakan HANYA angka yang ada di data. Jangan mengarang atau menghitung ulang angka baru.
- Skor dan tingkat risiko sudah dihitung sistem; Anda menjelaskan makna dan implikasinya.
- Bahasa Indonesia formal, ringkas, dan jelas. Gunakan format Markdown.
- Analisis ini adalah red flag/peringatan dini, BUKAN bukti kecurangan.

Struktur catatan:
1. **Ringkasan Eksekutif** (3–4 kalimat)
2. **Temuan Utama** (maksimal 5, urut dari risiko tertinggi). Untuk tiap temuan tulis:
   Kondisi (angka) · Kriteria pembanding (waktu berjalan/daerah lain/tahun lalu) ·
   Risiko & dampak · Prosedur audit lanjutan yang disarankan
3. **Hal yang Perlu Diklarifikasi ke OPD/BPKAD**
4. **Rekomendasi**
5. **Keterbatasan Analisis**"""

SYSTEM_CHAT = """Anda adalah asisten auditor internal pemerintah (APIP) yang membantu pengguna memahami
data realisasi APBD berikut. Jawab dalam Bahasa Indonesia, ringkas, berdasarkan data yang diberikan saja.
Jika data tidak cukup untuk menjawab, katakan terus terang. Jangan mengarang angka."""


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
# Pengambilan data (dengan cache 6 jam agar tidak membebani server DJPK)
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
            df, info = f.result()
            if df is not None and not df.empty:
                hasil[label] = df
            else:
                gagal.append(label)
            bar.progress(i / len(daftar), text=f"{judul} ({i}/{len(daftar)})")
    bar.empty()
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


def bagian_ai(konteks, kunci, model):
    """Tab catatan audit + chat. `kunci` membedakan hasil analisis yang berbeda."""
    if not get_api_key():
        st.info("🔑 GROQ_API_KEY belum diatur (Streamlit Secrets / file .env). "
                "Analisis angka tetap berjalan, tetapi fitur AI dinonaktifkan.")
        return

    st.subheader("🤖 Catatan Prosedur Analitis (AI)")
    memo = st.session_state.setdefault("memo", {})
    if st.button("✍️ Buat Catatan Audit", type="primary", key=f"btn_{kunci}"):
        with st.spinner("AI sedang menyusun catatan audit..."):
            try:
                memo[kunci] = tanya_groq(SYSTEM_MEMO, [{"role": "user", "content": konteks}], model)
            except Exception as e:
                st.error(f"Gagal memanggil Groq: {e}\n\nCek API key dan nama model di "
                         "https://console.groq.com/docs/models")
    if kunci in memo:
        st.markdown(memo[kunci])
        st.download_button("📄 Unduh catatan (.md)", memo[kunci], file_name="catatan_audit_siaga_apbd.md")

    st.divider()
    st.subheader("💬 Tanya AI tentang data ini")
    chat = st.session_state.setdefault("chat", {}).setdefault(kunci, [])
    for m in chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("Contoh: Pos belanja mana yang paling perlu diperiksa sebelum tutup tahun?"):
        chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                try:
                    msgs = [{"role": "user", "content": f"DATA:\n{konteks}"},
                            {"role": "assistant", "content": "Baik, data sudah saya pahami."}] + chat[-10:]
                    jawab = tanya_groq(SYSTEM_CHAT, msgs, model)
                except Exception as e:
                    jawab = f"Gagal memanggil Groq: {e}"
            st.markdown(jawab)
        chat.append({"role": "assistant", "content": jawab})


# ---------------------------------------------------------------------------
# Tampilan
# ---------------------------------------------------------------------------
WARNA_RISIKO = {"Tinggi": "#d62728", "Sedang": "#ff9f1c", "Rendah": "#2a9d8f"}


def tabel_tampil(h):
    t = h[["Akun", "Kelompok", "Level", "Anggaran", "Realisasi", "Sisa", "Real%",
           "Real%_TahunLalu", "Real%_Pembanding", "Gap_Waktu", "Risiko", "Alasan"]].copy()
    for c in ["Anggaran", "Realisasi", "Sisa"]:
        t[c] = t[c] / 1e9
    t["Akun"] = t.apply(lambda r: ("    " * (r["Level"] - 1)) + r["Akun"], axis=1)
    return t.drop(columns="Level")


KOLOM_CFG = {
    "Anggaran": st.column_config.NumberColumn("Anggaran (Rp M)", format="%.2f"),
    "Realisasi": st.column_config.NumberColumn("Realisasi (Rp M)", format="%.2f"),
    "Sisa": st.column_config.NumberColumn("Sisa (Rp M)", format="%.2f"),
    "Real%": st.column_config.NumberColumn("Realisasi %", format="%.1f"),
    "Real%_TahunLalu": st.column_config.NumberColumn("Thn Lalu %", format="%.1f"),
    "Real%_Pembanding": st.column_config.NumberColumn("Pembanding %", format="%.1f"),
    "Gap_Waktu": st.column_config.NumberColumn("Gap vs Waktu (poin)", format="%.1f"),
}


def grafik_akun(h, periode):
    d = h[(h["Level"] > 1) & h["Kelompok"].isin(["Pendapatan", "Belanja"]) & (h["Anggaran"] > 0)].copy()
    d = d.iloc[::-1]
    fig = px.bar(d, x="Real%", y="Akun", color="Risiko", orientation="h",
                 color_discrete_map=WARNA_RISIKO, hover_data={"Alasan": True},
                 category_orders={"Akun": list(d["Akun"])}, height=max(420, 26 * len(d)))
    if d["Real%_Pembanding"].notna().any():
        fig.add_trace(go.Scatter(x=d["Real%_Pembanding"], y=d["Akun"], mode="markers",
                                 marker=dict(symbol="diamond", size=10, color="#264653"),
                                 name="Pembanding"))
    fig.add_vline(x=periode / 12 * 100, line_dash="dash", line_color="gray",
                  annotation_text=f"Waktu berjalan {periode / 12 * 100:.0f}%")
    fig.update_layout(xaxis_title="Realisasi (%)", yaxis_title="", legend_title="",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def grafik_peringkat(p, fokus=None, periode=12):
    d = p.copy()
    d["Fokus"] = d["Wilayah"].apply(lambda w: "Dipilih" if w == fokus else "Lainnya")
    fig = px.bar(d, x="Belanja Daerah", y="Wilayah", orientation="h", color="Fokus",
                 color_discrete_map={"Dipilih": "#d62728", "Lainnya": "#8ab6d6"},
                 height=max(380, 28 * len(d)), hover_data=["Belanja Modal", "Pendapatan Daerah"])
    fig.add_trace(go.Scatter(x=d["Belanja Modal"], y=d["Wilayah"], mode="markers",
                             marker=dict(symbol="circle", size=9, color="#264653"), name="Belanja Modal %"))
    fig.add_vline(x=periode / 12 * 100, line_dash="dash", line_color="gray", annotation_text="Waktu berjalan")
    fig.update_layout(xaxis_title="Realisasi Belanja Daerah (%)", yaxis_title="", legend_title="",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def unduh_excel(lembar: dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nama, df in lembar.items():
            df.to_excel(w, sheet_name=nama[:31], index=False)
    return buf.getvalue()


def tampil_pemda(nama, tahun, periode, h, peringkat, keterangan, model, sumber):
    waktu = periode / 12 * 100
    idx = h.set_index("Akun")

    def m(akun):
        return idx.loc[akun, "Real%"] if akun in idx.index else None

    st.markdown(f"### {nama} — realisasi s.d. {BULAN[periode - 1]} {tahun}")
    st.caption(f"Sumber: Portal Data SIKD DJPK Kemenkeu ({sumber}). {keterangan}")
    c = st.columns(5)
    c[0].metric("Waktu berjalan", an.pct(waktu))
    for col, akun in zip(c[1:4], ["Pendapatan Daerah", "Belanja Daerah", "Belanja Modal"]):
        v = m(akun)
        col.metric(akun, an.pct(v), delta=f"{v - waktu:.1f} poin vs waktu" if v is not None else None)
    c[4].metric("Akun risiko tinggi", int((h["Risiko"] == "Tinggi").sum()))

    konteks = an.ringkasan_untuk_ai(nama, tahun, periode, h, peringkat, keterangan)
    t1, t2, t3, t4, t5 = st.tabs(["📊 Analisis", "🚩 Daftar Risiko", "🏆 Perbandingan", "🤖 AI Auditor", "📥 Unduh"])
    with t1:
        st.plotly_chart(grafik_akun(h, periode), width="stretch")
        st.dataframe(tabel_tampil(h), column_config=KOLOM_CFG, hide_index=True, width="stretch")
    with t2:
        r = h[h["Risiko"] != "Rendah"].sort_values(["Skor", "Sisa"], ascending=False)
        if r.empty:
            st.success("Tidak ada akun berisiko sedang/tinggi.")
        else:
            st.dataframe(tabel_tampil(r), column_config=KOLOM_CFG, hide_index=True, width="stretch")
        with st.expander("ℹ️ Cara sistem memberi skor risiko"):
            st.markdown(
                "- Tertinggal ≥ ambang dari waktu berjalan: **+1**, ≥ 2× ambang: **+2**\n"
                "- ≥ 15 poin di bawah pembanding (gabungan se-provinsi): **+1**\n"
                "- ≥ 15 poin di bawah periode sama tahun lalu: **+1**\n"
                "- Belanja melebihi pagu: **+2**\n"
                "- Pos material (≥ 10% dari total kelompoknya): **+1**\n"
                "- Ada realisasi padahal pagu Rp0: langsung **Tinggi**\n\n"
                "Skor ≥ 3 = **Tinggi**, 2 = **Sedang**, lainnya **Rendah**.")
    with t3:
        if peringkat is None or peringkat.empty:
            st.info("Aktifkan 'Bandingkan dengan pemda se-provinsi' untuk melihat perbandingan.")
        else:
            st.plotly_chart(grafik_peringkat(peringkat, nama, periode), width="stretch")
            st.dataframe(peringkat, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.1f") for k in an.AKUN_KUNCI})
    with t4:
        bagian_ai(konteks, f"pemda|{nama}|{tahun}|{periode}", model)
        with st.expander("Lihat data yang dikirim ke AI"):
            st.code(konteks, language="markdown")
    with t5:
        lembar = {"Analisis": h}
        if peringkat is not None and not peringkat.empty:
            lembar["Perbandingan"] = peringkat
        st.download_button("📥 Unduh hasil analisis (.xlsx)", unduh_excel(lembar),
                           file_name=f"SIAGA-APBD_{nama}_{tahun}-{periode:02d}.xlsx")


def tampil_multi(judul, tahun, periode, data, agregat, model, kunci):
    """Mode Provinsi / Nasional: peringkat antar wilayah + jumlah akun berisiko."""
    peringkat = an.tabel_peringkat(data, periode)
    jml = {nama: int((an.analisis_pemda(df, periode, df_pembanding=agregat)["Risiko"] == "Tinggi").sum())
           for nama, df in data.items()}
    peringkat["Akun Risiko Tinggi"] = peringkat["Wilayah"].map(jml)

    st.markdown(f"### {judul} — realisasi s.d. {BULAN[periode - 1]} {tahun}")
    c = st.columns(4)
    c[0].metric("Wilayah dianalisis", len(data))
    c[1].metric("Waktu berjalan", an.pct(periode / 12 * 100))
    c[2].metric("Belanja terendah", peringkat.iloc[0]["Wilayah"], an.pct(peringkat.iloc[0]["Belanja Daerah"]),
                delta_color="off")
    c[3].metric("Total sisa belanja", an.rp(peringkat["Sisa Belanja"].sum()))

    konteks = an.ringkasan_untuk_ai(judul, tahun, periode, None, peringkat)
    konteks += "\n\n## Jumlah akun berisiko tinggi per wilayah\n" + "\n".join(f"- {k}: {v}" for k, v in jml.items())
    t1, t2, t3 = st.tabs(["🏆 Peringkat", "🔍 Detail Wilayah", "🤖 AI Auditor"])
    with t1:
        st.plotly_chart(grafik_peringkat(peringkat, None, periode), width="stretch")
        st.dataframe(peringkat, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.1f") for k in an.AKUN_KUNCI})
        st.download_button("📥 Unduh peringkat (.xlsx)", unduh_excel({"Peringkat": peringkat}),
                           file_name=f"SIAGA-APBD_peringkat_{tahun}-{periode:02d}.xlsx")
    with t2:
        pilih = st.selectbox("Pilih wilayah", list(data.keys()))
        h = an.analisis_pemda(data[pilih], periode, df_pembanding=agregat)
        st.plotly_chart(grafik_akun(h, periode), width="stretch")
        st.dataframe(tabel_tampil(h), column_config=KOLOM_CFG, hide_index=True, width="stretch")
    with t3:
        bagian_ai(konteks, kunci, model)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("🛡️ SIAGA-APBD")
st.markdown("**AI Agent Pemantauan Realisasi APBD & Catatan Audit Berbasis Risiko** · "
            "_Studi kasus: Kabupaten Bengkalis_")

sb = st.sidebar
sb.header("⚙️ Pengaturan")
sumber = sb.radio("Sumber data", ["Ambil otomatis dari DJPK", "Upload file ekspor DJPK"])
sekarang = datetime.now()
tahun = sb.selectbox("Tahun anggaran", list(range(sekarang.year + 1, 2014, -1)),
                     index=1)
periode = sb.selectbox("Realisasi s.d. bulan", range(1, 13), index=sekarang.month - 1,
                       format_func=lambda b: BULAN[b - 1])
ambang = sb.slider("Ambang tertinggal dari waktu (poin)", 5, 30, 15,
                   help="Akun dianggap tertinggal jika realisasi% lebih rendah sekian poin dari waktu berjalan.")
model = sb.selectbox("Model AI (Groq)", MODEL_GROQ)

if sumber == "Ambil otomatis dari DJPK":
    mode = sb.radio("Mode pemantauan", ["Satu Pemda", "Satu Provinsi", "Nasional (antar provinsi)"])
    prov_map = ambil_provinsi(tahun)
    kode_prov = list(prov_map.keys())
    if mode != "Nasional (antar provinsi)":
        prov = sb.selectbox("Provinsi", kode_prov, index=kode_prov.index("04") if "04" in kode_prov else 0,
                            format_func=lambda k: prov_map[k])
    if mode == "Satu Pemda":
        pemda_map = ambil_pemda(prov, tahun)
        kode_pemda = list(pemda_map.keys())
        if not kode_pemda:
            sb.error("Daftar pemda tidak dapat diambil.")
            st.stop()
        pemda = sb.selectbox("Pemda", kode_pemda, index=kode_pemda.index("01") if "01" in kode_pemda else 0,
                             format_func=lambda k: pemda_map[k])
        cek_tl = sb.checkbox("Bandingkan dengan periode sama tahun lalu", True)
        cek_peer = sb.checkbox("Bandingkan dengan pemda se-provinsi", True)
    jalan = sb.button("🔎 Jalankan Analisis", type="primary", width="stretch")
else:
    sb.caption("File .xls hasil tombol Excel di portal DJPK, atau Excel/CSV berkolom Akun, Anggaran, Realisasi.")
    nama_up = sb.text_input("Nama pemda", "Kab. Bengkalis")
    f_now = sb.file_uploader("File realisasi (wajib)", type=["xls", "xlsx", "csv"])
    f_tl = sb.file_uploader("File periode sama tahun lalu (opsional)", type=["xls", "xlsx", "csv"])
    f_pb = sb.file_uploader("File pembanding, mis. Se-Provinsi (opsional)", type=["xls", "xlsx", "csv"])
    jalan = sb.button("🔎 Jalankan Analisis", type="primary", width="stretch", disabled=f_now is None)

sb.divider()
sb.caption("Data: Portal Data SIKD DJPK Kemenkeu. Hasil analisis adalah peringatan dini (red flag), "
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
                                             keterangan="Data dari file upload.", sumber="upload")
        elif mode == "Satu Pemda":
            with st.spinner(f"Mengambil data {pemda_map[pemda]} dari DJPK..."):
                df_now, info = ambil_aman(tahun, periode, prov, pemda)
            if df_now is None:
                st.error(f"Gagal mengambil data dari DJPK: {info}\n\nCoba lagi nanti, atau gunakan mode "
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
        elif mode == "Satu Provinsi":
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
# Tampilkan hasil (disimpan di session agar chat tidak mengambil ulang data)
# ---------------------------------------------------------------------------
H = st.session_state.get("hasil")
if H is None:
    st.info("👈 Pilih wilayah dan periode di sidebar, lalu klik **Jalankan Analisis**.")
    st.markdown("""
**Cara kerja agent:**
1. **Sensing** – mengambil data realisasi APBD langsung dari Portal Data SIKD DJPK Kemenkeu.
2. **Thinking** – menghitung realisasi tiap akun dan membandingkannya dengan waktu berjalan,
   periode yang sama tahun lalu, serta pemda lain; lalu memberi skor risiko.
3. **Acting** – AI menyusun catatan prosedur analitis (temuan, risiko, prosedur audit lanjutan)
   dan menjawab pertanyaan Anda.
""")
else:
    t_, p_ = st.session_state["param"]
    if H["jenis"] == "pemda":
        tampil_pemda(H["nama"], t_, p_, H["h"], H["peringkat"], H["keterangan"], model, H["sumber"])
    else:
        tampil_multi(H["judul"], t_, p_, H["data"], H["agregat"], model, f"multi|{H['judul']}|{t_}|{p_}")
