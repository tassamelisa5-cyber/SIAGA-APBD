# 🛡️ SIAGA-APBD

**AI Agent Pemantauan Realisasi APBD & Catatan Audit Berbasis Risiko**
_Studi kasus: Kabupaten Bengkalis, Provinsi Riau_

SIAGA-APBD membantu auditor internal pemerintah (APIP/Inspektorat), BPKAD, dan pemangku kepentingan
memantau realisasi APBD serta mendeteksi lebih awal pos anggaran yang berisiko, sebagai alat bantu
**prosedur analitis audit**.

## Cara Kerja Agent

| Tahap | Yang dilakukan |
|---|---|
| **Sensing** | Mengambil data realisasi APBD otomatis dari Portal Data SIKD DJPK Kemenkeu (atau upload file sebagai cadangan) |
| **Thinking** | Python menghitung realisasi tiap akun dan membandingkannya dengan waktu berjalan, periode sama tahun lalu, dan pemda lain, lalu memberi skor risiko |
| **Acting** | AI (Groq) menyusun catatan prosedur analitis: temuan, risiko, prosedur audit lanjutan, rekomendasi, dan menjawab pertanyaan pengguna |

Semua angka dihitung oleh Python, bukan oleh AI, untuk menghindari kesalahan hitung atau angka karangan (halusinasi).

## Fitur
- 3 mode pemantauan: **Satu Pemda**, **Satu Provinsi**, **Nasional (antar provinsi)**
- Skor risiko per akun (Tinggi / Sedang / Rendah) beserta alasannya
- Grafik realisasi vs waktu berjalan dan peringkat antar pemda
- Catatan audit otomatis dan chat tanya-jawab dengan AI
- Unduh hasil analisis (Excel) dan catatan audit (Markdown)

## Teknologi
Python · Streamlit · Pandas · Plotly · Groq API (`openai/gpt-oss-120b`) · Requests/BeautifulSoup

## Struktur File
| File | Fungsi |
|---|---|
| `app.py` | Tampilan aplikasi & integrasi AI |
| `analysis.py` | Perhitungan dan skor risiko |
| `djpk_client.py` | Pengambil data dari Portal SIKD DJPK |
| `requirements.txt` | Daftar library |

## Menjalankan di Laptop
```bash
pip install -r requirements.txt
```
Buat file `.streamlit/secrets.toml` berisi:
```toml
GROQ_API_KEY = "gsk_xxxxxxxx"
```
Lalu jalankan:
```bash
streamlit run app.py
```
Aplikasi terbuka di http://localhost:8501

## Sumber Data
Portal Data SIKD, Direktorat Jenderal Perimbangan Keuangan (DJPK), Kementerian Keuangan RI.

## Keterbatasan
- Data DJPK berupa total per jenis akun (tanpa rincian kegiatan/transaksi), sehingga hasil analisis
  adalah **peringatan dini (red flag), bukan bukti penyimpangan**.
- Portal DJPK bukan API resmi yang terdokumentasi; jika situs berubah, `djpk_client.py` perlu disesuaikan.
- Belum menggunakan database; riwayat analisis belum tersimpan.

## Pengembangan Lanjutan
- Modul audit APBDes per desa (data realisasi APBDes dari DPMD)
- Database untuk menyimpan data bulanan & jejak audit (audit trail)
- Otomatisasi laporan berkala (mis. dengan n8n)

## Tim
- **Tassa Melisa Wati** – D4 Akuntansi Keuangan Publik · analisis kebutuhan audit, sumber data, dan laporan
- **Ammar Syuhada** – pengujian, pengelolaan repository, dan deployment

Kode aplikasi disusun dengan bantuan asisten AI (Claude, Anthropic) berdasarkan rancangan kebutuhan,
sumber data, dan logika audit yang ditentukan tim.

_Tugas Mata Kuliah Teknologi Audit._
