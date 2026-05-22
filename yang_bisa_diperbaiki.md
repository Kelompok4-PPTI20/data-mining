# Evaluasi Notebook — List Improvement (untuk dievaluasi)

> Diorganisir per phase. Setiap item ditandai dampaknya terhadap rubric:
> - **[RUBRIC]** = langsung nyentuh kriteria Excellent di PDF
> - **[QUALITY]** = improvement metodologis, nggak wajib tapi naikin kedalaman analisa
> - **[HYGIENE]** = cleanup/reproducibility, bukan ngubah hasil
> - ~~strikethrough~~ = sudah dipertimbangkan tapi dicoret, alasan dicantumkan

---

## Phase 1 — Preprocessing

### 1.1 Scaling pilihan untuk `Balance` [QUALITY]
StandardScaler langsung dipakai tanpa analisa per-feature. Khususnya `Balance` punya zero-spike ~36% — distribusinya bimodal (banyak nol + ekor kontinu), bukan Gaussian.
- Kandidat: `RobustScaler` (median/IQR, tahan zero-spike), `log1p` transform, atau split jadi dua feature (`HasBalance` binary + `BalanceAmount` kontinu di subset non-zero).
- Hipotesa: bisa naikin silhouette score K-Means karena ngurangin distorsi distance.

### 1.2 `NumOfProducts` di-scale padahal diskrit [QUALITY]
Cell 11 masukin `NumOfProducts` ke `scale_cols` bareng feature kontinu. Padahal valuesnya cuma {1,2,3,4} — ordinal, bukan kontinu.
- Opsi: keluarin dari scaler, biarin sebagai ordinal integer; ATAU treat sebagai categorical (OHE).
- Dampak: distance metric K-Means lebih bener interpretasinya.

### 1.3 Konsistensi rationale binning [RUBRIC]
Cell 12 pake `pd.cut` (domain bins) untuk Credit/Age/Tenure/Balance, tapi `pd.qcut` (equal frequency) untuk Salary. Tidak ada justifikasi kenapa Salary beda.
- Rubric Excellent: "Discretization is domain-justified (not arbitrary)". Tambahin satu-dua kalimat kenapa Salary pake quartile (mis. nggak ada anchor domain yang jelas untuk salary tier di multi-country dataset).

### 1.4 Alternative unsupervised binning [QUALITY]
Test binning lain selain `cut`/`qcut`. Karena directive "discovery not prediction", binning supervised (Chi-square vs target, MDLP-supervised, Decision-tree binning) otomatis kecoret. Yang masih valid:
- **Fuzzy binning** (soft membership)
- **Clustering-based binning** (K-Means/DBSCAN pada variabel kontinu itu sendiri)
- **MDLP entropy binning pada distribusi variabelnya sendiri** (bukan vs target)
- ~~Chi-square vs `Exited` / Optimal binning vs target~~ — coret, supervised.

### 1.5 Feature selection: hasil MI nggak dipakai? [HYGIENE]
Cell 10 hitung MI scores + threshold 0.005. Tapi di cell 11 (Path A clustering matrix) semua feature numerik dipake, nggak ada yang di-drop berdasarkan MI threshold.
- Cek: apakah ada feature yang MI < 0.005 dan harusnya dikeluarin dari clustering input? Atau memang semua lolos threshold? Kalau semua lolos, dokumentasikan eksplisit ("all features retained, none below threshold").

### 1.6 Outlier retention tanpa justifikasi spesifik [RUBRIC]
Cell 13 bilang "Outliers Retained → Legitimate behavioral segments". Rubric Excellent: "ALL issues handled with justification". Justifikasi sekarang masih generic — bisa diperkuat dengan: misalnya, outlier di `Age` adalah customer 60–92 tahun (valid demographic), outlier di `Balance` adalah high-net-worth (valid wealth segment), bukan data error.

### 1.7 Encoding choice tidak dibandingkan [QUALITY]
OHE dipake untuk Geography/Gender. Alternatif: target encoding (tapi ini supervised — coret), frequency encoding, atau ordinal untuk Gender (binary). Untuk Geography 3-level, OHE memang paling defensible. Tambahin satu baris di report kenapa OHE dipilih (cardinality rendah, no implicit ordering).

---

## Phase 2 — Clustering

### 2.1 K-Means + OHE secara teknis bermasalah [RUBRIC]
K-Means pake euclidean distance — OHE binary (0/1) di-mix sama feature kontinu yang sudah di-scale, distance jadi nggak meaningful. Bukti: silhouette di notebook **0.10–0.14** di semua K (rendah).
- Opsi A (per saran Ko Henry): drop kategorikal dari K-Means input, jadiin variabel profiling post-hoc aja.
- Opsi B: **K-Prototypes** (`kmodes` library) — canonical untuk mixed numerical+categorical, pake gabungan Euclidean + Hamming distance.
- Opsi C: Embedding/dimensionality reduction dulu (FAMD components → K-Means di FAMD space).

### 2.2 DBSCAN dengan data kategorikal [QUALITY]
DBSCAN sekarang pake `metric='euclidean'` di X_cl yang sudah include OHE — sama masalah dengan #2.1.
- Kandidat: **Gower distance** (`gower` library) atau **Hamming** metric untuk binary part.
- ε dan min_samples juga harus di-retune kalau ganti metric.

### 2.3 `OPTIMAL_K = 3` di-hardcode [RUBRIC]
Cell 17: `OPTIMAL_K = 3   # Update based on Elbow + Silhouette findings`. Komen-nya jujur bilang harus di-update, tapi belum dilakukan validasi eksplisit di markdown.
- Rubric Excellent: "both Elbow and Silhouette interpreted correctly". Tambahin satu paragraf yang spesifik nyebut: "Elbow menunjukkan kink di K=X, Silhouette puncak di K=Y, jadi kita pilih K=Z karena ..."

### 2.4 Gap statistic untuk menambah validasi K [QUALITY]
Silhouette + Elbow sekarang ambigu (silhouette flat 0.10–0.14). **Gap statistic** (`gap-statistic` package atau implementasi manual via Tibshirani 2001) bisa dijadiin tie-breaker. ~~Visualisasi PCA/t-SNE~~ — coret, FAMD 2D projection sudah ada di cell 22.

### 2.5 K-Means vs Hierarchical agreement tidak diukur [RUBRIC]
Cell 23 markdown bilang "Ward Hierarchical agrees closely with K-Means" — tapi nggak ada metric. Tambahin **Adjusted Rand Index (ARI)** atau **Normalized Mutual Information (NMI)** antara K-Means labels dan Hierarchical labels.
- Rubric Excellent untuk Phase 2 minta "cluster validity" — ini langsung jawab itu.

### 2.6 Cluster profile table di cell 23 masih placeholder [RUBRIC]
Markdown cell 23 isinya angka contoh: "~4,200 | ~8%". Setelah notebook di-run, table ini harus di-update dengan angka aktual + nama persona yang sesuai data.
- Rubric Excellent: "every cluster gets a named profile with domain meaning".

### 2.7 DBSCAN ε dan min_samples nggak data-driven [QUALITY]
Cell 21: `EPSILON = 1.5, MIN_SAMPLES = 10` hardcoded. k-NN plot di cell 20 ada, tapi pemilihan ε=1.5 nggak dilink ke elbow di plot tersebut. Tambahin observasi: "elbow k-NN plot di ε≈X, jadi kita pakai ε=X".

---

## Phase 3 — Association Rule Mining

### 3.1 Granularity Tenure_Band rendah [QUALITY]
Cell 12: `Tenure_Band` cuma 3 bin (`New_Customer`, `Established`, `Loyal`) dengan cut [-1, 2, 5, 10]. Bisa lebih granular (4 bin) untuk catch pattern di middle range — atau dijustifikasi kenapa 3 cukup.

### 3.2 Anti-rules (rules → Retained) di-drop [QUALITY]
Cell 25 drop `Churn_Status_Retained` dari transaction matrix. Defensible (hindari tautological rules), tapi: rules yang mining retention profile bisa kasih insight komplementer (mis. "Active + 2 Products + France → Retained, lift 1.6").
- Opsi: dua pass — satu mine churn antecedent, satu mine retention antecedent. Bagus untuk diskusi "what did we discover".

### 3.3 Tambah interestingness measures [QUALITY]
Sekarang: Support, Confidence, Lift, Conviction. Bisa tambah:
- **Leverage** (P(A,B) − P(A)·P(B)) — sudah ada di mlxtend output, tinggal include
- **Kulczynski** + **Imbalance Ratio (IR)** — bagus untuk rare items
- **Chi-square test** of independence — buat statistical significance per rule

### 3.4 `min_support` lowering rationale [RUBRIC]
Cell 26: lowering 0.05 → 0.03 untuk hit 10 rules. Komen di code sudah bagus. Pastikan ini juga masuk ke writeup/laporan akhir — assessor bisa nanyain "kenapa 0.03 bukan 0.05".

### 3.5 Top-N rules dipilih by lift saja [QUALITY]
Cell 29: `nlargest(10, 'lift')`. Bagus, tapi rule dengan lift extreme bisa jadi rule dengan support kecil. Tambah filter sekunder atau presentasi multi-axis (mis. top 10 by lift × confidence × support composite, atau Pareto plot).

---

## Phase 4 — Anomaly Detection

### 4.1 Cell 38 markdown masih placeholder `[update]` [RUBRIC]
Markdown report cell 38 isinya `[update]` di hampir semua angka. Harus diisi dari hasil run aktual sebelum submission.
- Rubric Excellent: "All three methods applied AND systematically compared" — comparison table-nya literally kosong sekarang.

### 4.2 IsolationForest `contamination=0.05` arbitrary [QUALITY]
IQR flag rate biasanya lebih tinggi dari 5% di dataset ini (Age skewed). Set contamination=0.05 bikin IF undershoot dibanding IQR/Z. Bisa:
- Tune contamination sesuai natural rate observed di IQR (mis. ~10–15%)
- Atau pake `contamination='auto'` dan justifikasi
- Atau jalankan multiple contamination values + sensitivity analysis

### 4.3 Tambah Local Outlier Factor (LOF) [QUALITY]
Rubric minta 3 method (IQR + Z + IF), sudah memenuhi. Tapi **LOF** (density-based) komplementer ke IF (tree-based) dan IQR/Z (statistical). Tambah sebagai 4th method = bonus depth, sekaligus cross-check density-based assumption yang juga ada di DBSCAN.

### 4.4 Method agreement metrics [RUBRIC]
Cell 35 sudah hitung composite score (count of methods agreeing). Tambahin pairwise metric: **Jaccard index** atau **Cohen's kappa** antar pasangan method (IQR↔Z, IQR↔IF, IF↔DBSCAN).
- Rubric Excellent: "systematically compared" — ini jawab itu lebih formal.

### 4.5 Cross-reference DBSCAN → Phase 4 [RUBRIC ★]
Cell 35 sudah load `dbscan_outlier_indices.npy` dan masukin ke composite — **sudah ketemu rubric requirement #5 (hard rule)**. Pastikan ini di-highlight di writeup karena explicitly graded.

### 4.6 Anomaly classification rule-based, bisa di-trace [QUALITY]
Cell 36 `classify_anomaly()` pake if-else. Setelah classification, generate breakdown table: "Class A: N records, Class B: N records, Class C: N records dengan sub-breakdown". Ini bukti "each anomaly classified with specific evidence".

---

## Cross-cutting / Hygiene

### 5.1 Refactor notebook → modular `.py` [HYGIENE]
Pisah notebook jadi modules:
- `src/preprocessing.py` (Path A + Path B builders)
- `src/clustering.py` (K-Means, Hierarchical, DBSCAN wrappers + validation)
- `src/arm.py` (Apriori pipeline)
- `src/anomaly.py` (IQR, Z, IF, composite)
- Notebook tinggal orchestration + visualisasi.

Bagus untuk Phase 5 dashboard (Dash/Streamlit import langsung dari modules) dan untuk demo "engineering quality" di expo.

### 5.2 Markdown placeholder cleanup [RUBRIC]
Sebelum submission, sweep markdown cells untuk:
- Cell 23: cluster profile table → angka aktual
- Cell 38: anomaly comparison table → angka aktual
- Cell 39: Q1–Q4 KDD synthesis → final wording

### 5.3 Output directory consistency [HYGIENE]
Notebook nulis ke `../data/processed/` dan `../outputs/`. Pastikan kedua folder exist dan di-gitignore yang sesuai (raw CSV bisa di-track, processed bisa di-regenerate jadi ignore).

### 5.4 Reproducibility check [HYGIENE]
RANDOM_STATE=42 sudah konsisten. Tambahan: `requirements.txt` dengan versi pinned (pandas, sklearn, mlxtend, prince, kmodes kalau dipakai).

---

## Prioritas (saran)

**Must-do sebelum final submission (rubric-critical):**
- 2.6 (cluster profile table) — kosong sekarang
- 4.1 (anomaly comparison table) — kosong sekarang
- 2.3 (justifikasi OPTIMAL_K)
- 5.2 (markdown sweep)
- 2.1 ATAU dokumentasi eksplisit kenapa K-Means+OHE diterima meskipun silhouette rendah

**High-value depth (Excellent vs Good):**
- 2.5 (ARI/NMI antar algoritma)
- 4.4 (Jaccard/kappa antar method)
- 1.6 (outlier justification specific)
- 3.4 (writeup min_support lowering)

**Nice-to-have (kalau ada waktu):**
- 1.1, 1.2 (scaling alternatives)
- 2.4 (gap statistic)
- 3.3 (more interestingness measures)
- 4.3 (LOF as 4th method)
- 5.1 (refactor ke .py)
