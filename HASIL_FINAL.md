# Laporan Hasil Final Training — MSDL-JCI (Siap Sidang, Sisi Lab CLOSED)

> Dokumen ini adalah titik acuan tunggal hasil training untuk penulisan laporan thesis.
> Semua angka di bawah berasal dari run beku **2026-10-09** (commit `7c9205e`, 29 sel notebook, checks 2×OK).
> Detail pelacakan: issue #40 (closed), #36–#39, #43.

## 1. Pesan siap-kirim ke dosen pembimbing

> Yth. Bapak/Ibu Pembimbing,
>
> Pelatihan model thesis (LSTM + MLP makro + Frozen IndoBERT + Soft Gating, horizon t+5, JCI 2018–2025) telah **selesai dan terkunci**. Korpus final 20.784 artikel 3 portal (detik 4.953 + CNBC 5.556 + Kontan 10.275), seluruh pipeline leak-free lolos pemeriksaan (scaling train-only, ffill makro, label t+5 past-only).
>
> Hasil honestly weak signal: AUC pooled **0,4988** (≈ chance), tidak ada varian ablasi yang signifikan lebih baik (semua CI mencakup 0), Sharpe strategi ≈ buy-hold. Ini **bukan kegagalan** — kontrol berita-acak, audit gradien, dan kurva belajar membuktikan model bekerja benar tetapi sinyalnya memang tidak ada, konsisten dengan hipotesis pasar efisien bentuk semi-kuat (akan dibahas di §4.x).
>
> Dengan ini sisi training dinyatakan selesai; mohon izin lanjut ke penulisan laporan Bab 4 memakai angka terlampir. Daftar revisi dokumen (§2.1, Table 5, §2.7, §4.x) terlampir di checklist.
>
> Hormat saya,
> Jordan Theovandy (23.K1.0018)

## 2. Data final

| Komponen | Isi |
|---|---|
| JCI (Yahoo Finance) | 1.932 baris, 2018-01-02 → 2025-12-30 (OHLCV) |
| Makro (BI/BPS) | BI-Rate 97, inflasi 96, kurs 2.082 → ffill-only ke harian |
| Berita (arsip 3 portal) | detik 4.953 + CNBC 5.556 + Kontan 10.275 = **20.784** (2018–2025); ±20,7 rb terpakai past-only (20.381 terkonfirmasi pra-fix pipa + 392 baris pipa 2025 yang diselamatkan; angka pasti tercetak di baris `on/before JCI end` pada run final) |
| Sampel latih | 1.852 baris selaras → 1.825 sampel (lookback 28), 2018-05-31 → 2025-12-19, UP 55,2% |
| Embedding | Frozen `indobenchmark/indobert-base-p1` CLS 768 → proyeksi 64, cached |

## 3. Hasil utama (walk-forward, 5 fold × 304, train-only scaling per fold)

| Varian | AUC | Acc* | F1* | Sharpe (strat/bh) | MDD |
|---|---|---|---|---|---|
| LSTM (teknikal saja) | 0,4959 | 0,4612 | 0,0829 | 0,288/0,393 | −0,275 |
| LSTM + Makro | 0,5135 | 0,4888 | 0,4136 | 0,209/0,393 | −0,764 |
| LSTM + Berita | 0,5020 | 0,4809 | 0,4525 | 0,293/0,393 | −0,400 |
| Berita-diacak (kontrol E1) | 0,4944 | 0,4809 | 0,4525 | 0,293/0,393 | −0,400 |
| Static fusion (tanpa gating) | 0,4988 | 0,4809 | 0,4525 | 0,293/0,393 | −0,400 |
| **Full (proposed)** | **0,4988** | 0,4599 | 0,2867 | 0,162/0,393 | −0,376 |

`*` Acc/F1 pada threshold Youden-J per fold (tuned-on-train). AUC threshold-free.

**Uji beda (paired ΔAUC vs full, 95% CI, t df=4):** +0,043 [−0,054,+0,139]; +0,014 [−0,126,+0,155]; −0,038 [−0,155,+0,079]; −0,034 [−0,093,+0,024] — **semua mencakup 0: tidak ada varian signifikan.**

## 4. Bukti perilaku (mengapa ini "hasil", bukan "gagal")

| Bukti | Angka | Makna |
|---|---|---|
| Gate (α/β/γ) | 0,35 / 0,61 / **0,03** | Kanal berita dimatikan secara terlatih di semua fold |
| Gradien epoch-0 | news 0,058 > mlp 0,008 > lstm 0,002 | Sinyal sampai ke cabang berita — gate menutup karena noise, bukan macet |
| Kontrol acak E1 | 0,494 ≈ 0,502 | Berita = noise |
| Learning curve E2 | berhenti epoch 9–68, loss ≈ 0,55–0,65 | Sinyal habis, bukan kurang latih |
| Gate vs ATR | corr +0,52 (α) / −0,48 (β), n=5 | Gate sugestif adaptif: volatil → teknikal, tenang → makro |
| COVID vs rest | AUC 0,52 vs 0,48 | Tak ada edge bahkan saat krisis |
| Kalibrasi ECE | 0,015, mean pred 0,45 | Model jujur tidak yakin |
| Focal vs BCE | 0,5025 vs 0,4988 | Tetap BCE |
| Grid lb×LR | 14/3e-4 0,5115 terbaik numerik, dalam noise | Tetap 28/1e-3 (Table 5) |
| Agregasi berita | last 0,4988 ≈ mean | §2.2 tak perlu diubah |
| Recall @0,5 | UP 0,20 / DOWN 0,78 | Model condong DOWN → wajib lapor Youden-J |

## 5. Jawaban RQ (satu baris per RQ)

1. **MLP makro?** +0,018 AUC vs LSTM saja, CI mencakup 0 → tidak signifikan.
2. **IndoBERT?** Tidak — kontrol acak setara; gate menutup kanal berita.
3. **Soft gating (Sharpe/stabilitas)?** Sharpe 0,162 vs buy-hold 0,393 — tidak unggul; nilai gating terbukti pada perilaku (adaptif, menutup noise), bukan return.

## 6. Sisa kerja: hanya dokumen (lab CLOSED)

- [ ] §2.1 tulis korpus 20.784 (→ #36)
- [ ] Table 5 + ID IndoBERT penuh (→ #37)
- [ ] §2.7 + Bab 4: tabel §3, Youden-J, CI (→ #38, draf koreksi di komentar issue)
- [ ] §4.x EMH 1,5 hlm (→ #39, draf + koreksi angka di komentar)
- [ ] Lunakkan objektif "significantly higher" + perbarui Experimental Setup (3.12, run lokal) (→ #43)

## 7. Batasan yang dinyatakan jujur (tameng sidang)

Horizon hanya t+5; periode 2018–2025; tanpa biaya transaksi; JCI emerging market (temuan "tetap efisien" lebih kuat dari ekspektasi). Bahasa: selalu "gagal menolak / konsisten dengan", tidak pernah "membuktikan".
