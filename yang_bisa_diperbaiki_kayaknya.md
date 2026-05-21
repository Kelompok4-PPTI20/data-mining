1. Scaling vs Transformation utk normalisasi data, belum dianalisa lebih lanjut, yang mana lebih cocok, soalnya gua langsung tabrak pake standard scaler. Khususnya `Balance` yang punya zero-spike ~36% — RobustScaler atau log1p mungkin lebih cocok daripada StandardScaler.

2. Tes binning yang lain — opsi unsupervised aja, karena directive (PDF + context.md) bilang "discovery, not prediction" jadi binning yang pake target `Exited` (Chi-square vs target, Monotonic, Optimal, Decision-tree binning) udah otomatis kecoret. Yang masih valid:
   - Fuzzy binning
   - Clustering-based binning (K-Means / DBSCAN pada variabel kontinu itu sendiri)
   - MDLP entropy binning **pada distribusi variabelnya sendiri**, bukan vs target

3. Jujur masukin data kategori OHE ke K means itu sebenarnya ga bagus karena K means itu pake euclidean distance, jadi OHE nya ga bisa ngukur jarak dengan baik. Bukti: silhouette di notebook cuma 0.10–0.14 di semua K. Alternatif: pake teknik embedding/representasi vektor yang bisa menangkap hubungan antar kategori, atau langsung drop kategorikalnya kalau kata Ko Henry kalau mau pake K means (jadiin variabel profiling post-hoc aja).

4. Untuk DBScan harusnya data kategori bisa, cuman gua belum cari tahu lebih lanjut. Kandidat: Gower distance atau Hamming metric.

5. Refactor dari notebook.ipynb jadi proses py yang lebih modular dan terstruktur. Pisahkan kode menjadi fungsi-fungsi yang jelas dengan input dan output yang terdefinisi dengan baik. Gunakan kelas untuk mengelompokkan fungsi-fungsi terkait.

6. ~~Belum ada analisis entropy sama chi-square untuk menentukan batas binning yang optimal.~~ **Coret** — sama issue kayak #2 (supervised, conflict sama directive). Tapi kalau yang dimaksud entropy *variabelnya sendiri* (MDLP gaya unsupervised), itu masih valid dan bisa digabung ke #2. Note: notebook udah punya Mutual Information untuk *feature selection* (cell 11), itu yang dapet poin di rubric PDF "correlation + entropy measures".

7. Hasil silhouette dan elbow method masih ambigu, mungkin perlu dicoba metode lain seperti gap statistic untuk menentukan jumlah cluster yang optimal. Untuk visualisasi cluster, notebook udah ada FAMD 2D projection (cell 22) yang fungsinya mirip PCA/t-SNE untuk mixed data — jadi sub-item visualisasi sudah ke-cover, yang masih open cuma gap statistic.
