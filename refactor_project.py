"""
═══════════════════════════════════════════════════════════════════════════════
SKRIP REFACTOR PROJECT SKRIPSI — Klasifikasi Komentar XGBoost & IndoBERT
═══════════════════════════════════════════════════════════════════════════════

Strategi: VERSI FINAL = v2 / v2b
         File usang/duplikat DIPINDAHKAN ke folder _archive (TIDAK DIHAPUS).

Cara pakai:
    1. Letakkan skrip ini di folder ROOT project Anda.
    2. Jalankan DRY-RUN dulu (lihat rencana tanpa mengubah apa pun):
           python refactor_project.py
    3. Kalau sudah yakin, jalankan eksekusi nyata:
           python refactor_project.py --apply

Skrip ini AMAN: default-nya hanya menampilkan rencana (dry-run).
Tidak ada file yang dihapus permanen; semuanya dipindah ke _archive/.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import shutil
from datetime import datetime

# ── KONFIGURASI ──────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
APPLY     = '--apply' in sys.argv
ARCHIVE   = os.path.join(ROOT, '_archive')
LOG_LINES = []

def log(msg):
    print(msg)
    LOG_LINES.append(msg)

def ensure_dir(path):
    if APPLY:
        os.makedirs(path, exist_ok=True)

def move_to_archive(rel_path, reason=''):
    """Pindahkan file/folder ke _archive dengan struktur yang dipertahankan."""
    src = os.path.join(ROOT, rel_path)
    if not os.path.exists(src):
        return False
    dst = os.path.join(ARCHIVE, rel_path)
    tag = f"  [ARSIP] {rel_path}"
    if reason:
        tag += f"   <- {reason}"
    log(tag)
    if APPLY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    return True

def move_file(src_rel, dst_rel):
    """Pindahkan/rename file ke lokasi struktur baru."""
    src = os.path.join(ROOT, src_rel)
    if not os.path.exists(src):
        return False
    dst = os.path.join(ROOT, dst_rel)
    log(f"  [PINDAH] {src_rel}  ->  {dst_rel}")
    if APPLY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
    return True

def copy_file(src_rel, dst_rel):
    src = os.path.join(ROOT, src_rel)
    if not os.path.exists(src):
        return False
    dst = os.path.join(ROOT, dst_rel)
    log(f"  [SALIN]  {src_rel}  ->  {dst_rel}")
    if APPLY:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return True

# ═══════════════════════════════════════════════════════════════════════════════
log("="*79)
log(f"REFACTOR PROJECT SKRIPSI  —  mode: {'EKSEKUSI NYATA' if APPLY else 'DRY-RUN (simulasi)'}")
log(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Root : {ROOT}")
log("="*79)

# ── 1. BUAT STRUKTUR FOLDER BARU YANG RAPI ──────────────────────────────────
log("\n[1] MEMBUAT STRUKTUR FOLDER BARU")
NEW_DIRS = [
    '01_scraping',
    '02_cleaning',
    '03_labelling',
    '04_preprocessing',
    '05_splitting',
    '06_modelling',
    '07_evaluation',
    '08_app',
    '09_visualization',
    'data/raw',
    'data/interim',
    'data/processed',
    'data/splits',
    'dictionaries',
    'models/ml',          # XGBoost & LR (.pkl)
    'models/indobert',    # folder model transformer
    'results/confusion_matrix',
    'results/cv',
    'results/spot_check',
    'outputs/figures',
    'outputs/docs',
    'docs',
]
for d in NEW_DIRS:
    full = os.path.join(ROOT, d)
    log(f"  [BUAT] {d}/")
    ensure_dir(full)
ensure_dir(ARCHIVE)

# ── 2. KAMUS (dictionaries) ─────────────────────────────────────────────────
log("\n[2] MERAPIKAN KAMUS -> dictionaries/")
for f in ['emoji_dict.csv','kbbi_dict.csv','profanity_dict.csv',
          'slang_dict.csv','stopword_dict.csv']:
    move_file(f, f'dictionaries/{f}')

# ── 3. DATA MENTAH (raw scraping) ───────────────────────────────────────────
log("\n[3] MERAPIKAN DATA SCRAPING -> data/raw/ & 01_scraping/")
move_file('01_data_scraping/scraping_ig.ipynb',     '01_scraping/scraping_instagram.ipynb')
move_file('01_data_scraping/scraping_tiktok.ipynb', '01_scraping/scraping_tiktok.ipynb')
move_file('01_data_scraping/combined_all_data.csv', 'data/raw/combined_all_data.csv')
# folder hasil_scraping (banyak csv mentah) -> data/raw/
move_file('01_data_scraping/hasil_scraping', 'data/raw/hasil_scraping')
# sisa folder lama diarsipkan
move_to_archive('01_data_scraping', 'folder scraping lama (isi sudah dipindah)')

# ── 4. CLEANING ─────────────────────────────────────────────────────────────
log("\n[4] MERAPIKAN CLEANING -> 02_cleaning/ & data/interim/")
move_file('02_data_cleaning/Data_Cleaning_Pipeline.ipynb', '02_cleaning/Data_Cleaning_Pipeline.ipynb')
move_file('02_data_cleaning/Data_Cleaning_Pipeline_Data_Labelling.ipynb', '02_cleaning/Data_Cleaning_Labelling.ipynb')
move_file('02_data_cleaning/cleaned_data_v2.csv', 'data/interim/cleaned_data_v2.csv')
# arsipkan versi cleaning lama & gambar perbandingan lama
move_to_archive('02_data_cleaning', 'folder cleaning lama (file final sudah dipindah)')

# ── 5. LABELLING ────────────────────────────────────────────────────────────
log("\n[5] MERAPIKAN LABELLING -> 03_labelling/ & data/")
move_file('03_data_labelling/Persiapan_Data_Labeling.ipynb',  '03_labelling/01_Persiapan_Labeling.ipynb')
move_file('03_data_labelling/Labeling_Otomatis_IndoBERT.ipynb','03_labelling/02_Labeling_Otomatis_IndoBERT.ipynb')
move_file('03_data_labelling/Spot_Check_Noise.ipynb',          '03_labelling/03_Spot_Check_Noise.ipynb')
move_file('03_data_labelling/data_labeling.csv',               'data/interim/data_labeling.csv')
move_file('03_data_labelling/labelled_data_final.csv',         'data/processed/labelled_data_final.csv')
move_file('03_data_labelling/indobert_labeler',               'models/indobert/indobert_labeler')
# arsipkan sisa labelling lama
move_to_archive('03_data_labelling', 'folder labelling lama (file final sudah dipindah)')

# ── 6. PREPROCESSING (FINAL = terpusat / v2) ────────────────────────────────
log("\n[6] MERAPIKAN PREPROCESSING -> 04_preprocessing/ & data/processed/")
# Notebook preprocessing FINAL berdasarkan timestamp: preprocessing_v2_final.ipynb (16 Jun 11:02)
# CATATAN: preprocessing_v2_80_20_gabungan_1.ipynb (16 Jun 22:50) isinya = SPLITTING + eksperimen,
#          sehingga ditempatkan di 05_splitting (lihat bagian [7]), bukan di sini.
move_file('preprocessing_v2_final.ipynb', '04_preprocessing/Preprocessing_v2_Final.ipynb')
# arsipkan notebook preprocessing terpusat lama (4 Jun) yang sudah usang
move_to_archive('04_data_preprocessing/00_Preprocessing_Terpusat.ipynb', 'preprocessing terpusat versi lama (4 Jun)')
# Output preprocessing final (v2) dari prep_v2/
move_file('prep_v2/data_preprocessed_v2.csv', 'data/processed/data_preprocessed_v2.csv')
# Hasil preprocessing terpisah xgb/idb (final)
move_file('04_data_preprocessing/data_preprocessed_xgboost.csv', 'data/processed/preprocessed_xgboost.csv')
move_file('04_data_preprocessing/data_preprocessed_indobert.csv','data/processed/preprocessed_indobert.csv')
# arsipkan folder preprocessing lama & notebook seprianto
move_to_archive('04_data_preprocessing', 'folder preprocessing lama (final sudah dipindah)')

# ── 7. SPLITTING (FINAL = splits_v2b) ───────────────────────────────────────
log("\n[7] MERAPIKAN SPLITTING -> data/splits/ (FINAL = splits_v2b)")
move_file('00_Data_Splitter_Final.ipynb', '05_splitting/Data_Splitter_Final.ipynb')
# Notebook splitting+eksperimen FINAL (timestamp terbaru 16 Jun 22:50, berisi desain 3 skenario + 9 eksperimen)
move_file('preprocessing_v2_80_20_gabungan_1.ipynb', '05_splitting/Splitting_dan_Eksperimen_Final.ipynb')
for f in ['v2b_manual_train.csv','v2b_manual_test.csv',
          'v2b_agab_train.csv','v2b_agab_test.csv',
          'v2b_bgab_train.csv','v2b_bgab_test.csv']:
    move_file(f'splits_v2b/{f}', f'data/splits/{f}')
# arsipkan SEMUA folder splits lama yang membingungkan
move_to_archive('splits_v2b', 'folder splits (isi final sudah dipindah)')
move_to_archive('splits',    'splits versi lama v1')
move_to_archive('splits_v2', 'splits versi antara (bukan final)')

# ── 8. MODELLING NOTEBOOKS (FINAL = 01..05 di root) ─────────────────────────
log("\n[8] MERAPIKAN NOTEBOOK MODELLING -> 06_modelling/")
modelling_nb = {
    '01_LR_A_Manual.ipynb':         '06_modelling/01_LR_A_Manual.ipynb',
    '01_LR_A_Gabungan.ipynb':       '06_modelling/01_LR_A_Gabungan.ipynb',
    '01_LR_B_Gabungan.ipynb':       '06_modelling/01_LR_B_Gabungan.ipynb',
    '02_XGB_NoTune_A_Manual.ipynb': '06_modelling/02_XGB_NoTune_A_Manual.ipynb',
    '02_XGB_NoTune_A_Gabungan.ipynb':'06_modelling/02_XGB_NoTune_A_Gabungan.ipynb',
    '02_XGB_NoTune_B_Gabungan.ipynb':'06_modelling/02_XGB_NoTune_B_Gabungan.ipynb',
    '03_XGB_Tuned_A_Manual.ipynb':  '06_modelling/03_XGB_Tuned_A_Manual.ipynb',
    '03_XGB_Tuned_A_Gabungan.ipynb':'06_modelling/03_XGB_Tuned_A_Gabungan.ipynb',
    '03_XGB_Tuned_B_Gabungan.ipynb':'06_modelling/03_XGB_Tuned_B_Gabungan.ipynb',
    '04_IndoBERT_A_Manual.ipynb':   '06_modelling/04_IndoBERT_A_Manual.ipynb',
    '04_IndoBERT_A_Gabungan.ipynb': '06_modelling/04_IndoBERT_A_Gabungan.ipynb',
    '04_IndoBERT_B_Gabungan.ipynb': '06_modelling/04_IndoBERT_B_Gabungan.ipynb',
}
for src, dst in modelling_nb.items():
    move_file(src, dst)

# ── 9. EVALUATION / REKAP ───────────────────────────────────────────────────
log("\n[9] MERAPIKAN EVALUASI -> 07_evaluation/")
move_file('05_Rekapitulasi.ipynb', '07_evaluation/Rekapitulasi.ipynb')

# ── 10. MODEL FILES (FINAL = v2) ────────────────────────────────────────────
log("\n[10] MERAPIKAN MODEL -> models/ml/ (FINAL = _v2_) ; arsipkan v1")
FINAL_ML = [
    'XGB_v2_Tuned_A_Manual.pkl','XGB_v2_Tuned_A_Gabungan.pkl','XGB_v2_Tuned_B_Gabungan.pkl',
    'XGB_v2_NoTune_A_Manual.pkl','XGB_v2_NoTune_A_Gabungan.pkl','XGB_v2_NoTune_B_Gabungan.pkl',
    'LR_v2_A_Manual.pkl','LR_v2_A_Gabungan.pkl','LR_v2_B_Gabungan.pkl',
    'vec_XGB_v2_Tuned_A_Manual.pkl','vec_XGB_v2_Tuned_A_Gabungan.pkl','vec_XGB_v2_Tuned_B_Gabungan.pkl',
    'vec_XGB_v2_NoTune_A_Manual.pkl','vec_XGB_v2_NoTune_A_Gabungan.pkl','vec_XGB_v2_NoTune_B_Gabungan.pkl',
    'vec_LR_v2_A_Manual.pkl','vec_LR_v2_A_Gabungan.pkl','vec_LR_v2_B_Gabungan.pkl',
]
for f in FINAL_ML:
    move_file(f'models/{f}', f'models/ml/{f}')

# Model IndoBERT final (folder safetensors) -> models/indobert/
# Prioritas: models/indobert_X  >  root/indobert_X
for d in ['indobert_A_Manual','indobert_A_Gabungan','indobert_B_Gabungan']:
    if os.path.exists(os.path.join(ROOT, 'models', d)):
        move_file(f'models/{d}', f'models/indobert/{d}')
    elif os.path.exists(os.path.join(ROOT, d)):
        move_file(d, f'models/indobert/{d}')

# Arsipkan SEMUA model versi lama (tanpa _v2_) yang masih di models/
# CATATAN: lewati subfolder 'ml' & 'indobert' (struktur baru) agar tidak ikut terarsip.
log("\n   -> Mengarsipkan model versi lama (non-v2) ...")
models_dir = os.path.join(ROOT, 'models')
PROTECTED = {'ml', 'indobert'}
if os.path.isdir(models_dir):
    for f in sorted(os.listdir(models_dir)):
        if f in PROTECTED:
            continue
        fp = os.path.join(models_dir, f)
        if os.path.isfile(fp) and f.endswith('.pkl'):
            # file final v2 sudah dipindah; sisa .pkl = versi lama
            move_to_archive(f'models/{f}', 'model versi lama (non-v2)')
        elif os.path.isdir(fp) and f.startswith('indobert'):
            move_to_archive(f'models/{f}', 'folder indobert duplikat')

# Folder indobert duplikat di root (kalau masih ada setelah pemindahan)
for d in ['indobert_A_Manual','indobert_A_Gabungan','indobert_B_Gabungan','indobert_labeler']:
    if os.path.exists(os.path.join(ROOT, d)):
        move_to_archive(d, 'folder indobert duplikat di root')

# ── 11. RESULTS ─────────────────────────────────────────────────────────────
log("\n[11] MERAPIKAN RESULTS -> results/ (FINAL = confusion_matrix_v2)")
# Confusion matrix v2 (final)
cm_v2 = os.path.join(ROOT, 'results', 'confusion_matrix_v2')
if os.path.isdir(cm_v2):
    for f in os.listdir(cm_v2):
        move_file(f'results/confusion_matrix_v2/{f}', f'results/confusion_matrix/{f}')
    move_to_archive('results/confusion_matrix_v2', 'folder CM v2 (isi sudah dipindah)')
# CV results v2
for f in ['cv_XGB_v2_Tuned_A_Gabungan.csv','cv_XGB_v2_Tuned_A_Manual.csv','cv_XGB_v2_Tuned_B_Gabungan.csv']:
    move_file(f'results/{f}', f'results/cv/{f}')
# Spot check final
move_file('results/tabel_spot_check.csv', 'results/spot_check/tabel_spot_check.csv')
move_file('results/spot_check_hasil.png', 'results/spot_check/spot_check_hasil.png')
move_file('results/kalimat_spot_check_bab3.txt', 'results/spot_check/kalimat_spot_check_bab3.txt')
# folder spot_check (band) -> results/spot_check/
for f in ['distribusi_per_band.png','spot_check_band_rendah.csv',
          'spot_check_band_sedang.csv','spot_check_band_tinggi.csv']:
    move_file(f'spot_check/{f}', f'results/spot_check/{f}')
move_to_archive('spot_check', 'folder spot_check lama (isi sudah dipindah)')

# Arsipkan CM versi lama (v1) yang ada langsung di results/
log("\n   -> Mengarsipkan confusion matrix versi lama (v1) ...")
res_dir = os.path.join(ROOT, 'results')
if os.path.isdir(res_dir):
    for f in sorted(os.listdir(res_dir)):
        fp = os.path.join(res_dir, f)
        if os.path.isfile(fp) and f.startswith('CM_') and f.endswith('.png'):
            move_to_archive(f'results/{f}', 'confusion matrix v1 (lama)')
        # cv & hist lama
        if os.path.isfile(fp) and (f.startswith('cv_') or f.startswith('hist_')) and f not in []:
            # hist IndoBERT tetap berguna -> pindah ke results/cv kalau cv, ke confusion kalau hist
            if f.startswith('hist_'):
                move_file(f'results/{f}', f'results/{f}')  # biarkan di results (training history)
            else:
                move_to_archive(f'results/{f}', 'cv lama (non-v2)')

# ── 12. APLIKASI STREAMLIT (FINAL = sentiment_app_v2_1.py, terbaru 17 Jun) ────────────────────
log("\n[12] MERAPIKAN APLIKASI -> 08_app/ (FINAL = sentiment_app_v2_1.py)")
# FINAL (berdasarkan timestamp terbaru 17 Jun 20:28): sentiment_app_v2_1.py
move_file('sentiment_app_v2_1.py', '08_app/sentiment_app.py')
move_to_archive('sentiment_app.py',    'app versi 1 (lama)')
move_to_archive('sentiment_app_fix.py','app versi fix (lama)')
move_to_archive('sentiment_app_v2.py', 'app v2 (digantikan v2_1)')

# ── 13. VISUALISASI & DIAGRAM ───────────────────────────────────────────────
log("\n[13] MERAPIKAN VISUALISASI -> 09_visualization/ & outputs/figures/")
move_file('Visualisasi_Skripsi.ipynb', '09_visualization/Visualisasi_Skripsi.ipynb')
move_file('Bab2_Diagram_Alur.ipynb',   '09_visualization/Bab2_Diagram_Alur.ipynb')
move_file('Bab3_Visualisasi_Data_Aktual.ipynb','09_visualization/Bab3_Visualisasi_Data_Aktual.ipynb')
move_file('diagram_alur_skripsi.ipynb','09_visualization/diagram_alur_skripsi.ipynb')
# Output gambar -> outputs/figures/ (FINAL = output_bab2, paling lengkap 9 gambar)
out_bab2 = os.path.join(ROOT,'output_bab2')
if os.path.isdir(out_bab2):
    for f in os.listdir(out_bab2):
        move_file(f'output_bab2/{f}', f'outputs/figures/{f}')
    move_to_archive('output_bab2','folder gambar bab2 (isi sudah dipindah)')
# output_visualisasi (gambar data aktual) -> outputs/figures/
ov = os.path.join(ROOT,'output_visualisasi')
if os.path.isdir(ov):
    for f in os.listdir(ov):
        move_file(f'output_visualisasi/{f}', f'outputs/figures/{f}')
    move_to_archive('output_visualisasi','folder visualisasi (isi sudah dipindah)')
# output_gambar (duplikat output_bab2) -> arsip
move_to_archive('output_gambar', 'duplikat output_bab2')

# ── 14. ARSIPKAN SEMUA NOTEBOOK & FILE EKSPERIMEN LAMA DI ROOT ──────────────
log("\n[14] MENGARSIPKAN NOTEBOOK & FILE EKSPERIMEN LAMA DI ROOT")
OLD_ROOT_FILES = [
    # notebook join/preprocessing/splitter versi lama
    '00_A1_join_csv.ipynb','00_A_Data_Preprocessing.ipynb','join_csv.ipynb',
    '00_Data_Splitter.ipynb','00_Data_Splitter_Fixed.ipynb',
    'preprocessing_v2_80_20_gabungan.ipynb',
    'preprocessing_v2_notebook.ipynb','preprocessing_v2_xgboost.ipynb',
    'Preprocessing_IndoBERT.ipynb','indobert_tuning.ipynb',
    'Spot_Check_Band_Confidence.ipynb','Spot_Check_Final.ipynb',
    'd121191078_skripsi_code_xgboost.ipynb','d121191078_skripsi_code_xgboost copy.ipynb',
    # csv eksperimen lama versi 2.0-4.0
    'data_preprocessing_2.0.csv','data_preprocessing_3.0.csv','data_preprocessing_4.0.csv',
    'data_preprocessed_indobert.csv','data_preprocessed_indobert_2.0.csv',
    'cleaned_data.csv','cleaned_data_baru.csv',
    'labelled_data_baru.csv','labelled_data_filtered.csv','labelled_data_final.csv',
    'data_labeling.csv','spot_check_200.csv',
    # gambar lama di root
    'analisis_confidence_otomatis.png','analisis_preprocessing.png',
    'feature_importance_xgboost_3.0.png','feature_importance_xgboost_4.0.png',
    # binary fasttext besar (opsional arsip)
    'cc.id.300.bin','cc.id.300.bin.gz',
    # sesi & struktur lama
    'instagram_session.json','struktur_project.txt',
]
for f in OLD_ROOT_FILES:
    move_to_archive(f, 'file/notebook eksperimen lama')

# folder modelling & preprocessing antara yang tersisa
move_to_archive('04_data_modelling', 'folder modelling eksperimen lama')
move_to_archive('prep_v2', 'folder prep_v2 (csv final sudah dipindah)')

# ── 15. SISA FOLDER MODELS KOSONG / DUPLIKAT ────────────────────────────────
log("\n[15] PEMBERSIHAN AKHIR")
# kalau folder models/ sudah kosong dari file lama, biarkan (sudah ada ml/ & indobert/)

log("\n" + "="*79)
log("RENCANA REFACTOR SELESAI DISUSUN.")
if not APPLY:
    log(">> Ini DRY-RUN. Tidak ada file yang diubah.")
    log(">> Jalankan ulang dengan:  python refactor_project.py --apply")
else:
    log(">> SELESAI DIEKSEKUSI. File lama ada di folder _archive/")
log("="*79)

# tulis log
log_path = os.path.join(ROOT, 'refactor_log.txt')
if APPLY:
    with open(log_path,'w',encoding='utf-8') as fp:
        fp.write('\n'.join(LOG_LINES))
    print(f"\nLog tersimpan: {log_path}")
