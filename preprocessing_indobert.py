"""
preprocessing_indobert.py
Modul preprocessing IndoBERT (5 tahap minimal) -- diekstrak dari
Preprocessing_IndoBERT_copy.ipynb supaya bisa di-import dari notebook mana pun
(termasuk notebook uji komentar baru), tanpa perlu copy-paste ulang.

Cara pakai di notebook lain:
    import sys
    sys.path.insert(0, BASE_DIR)
    from preprocessing_indobert import preprocess_indobert, muat_dictionary_indobert

    muat_dictionary_indobert(BASE_DIR)   # panggil SEKALI di awal notebook
    hasil = preprocess_indobert("teks komentar di sini")

CATATAN PENTING (jangan dihapus): loading emoji_dict di modul ini MENGIKUTI kode asli
Preprocessing_IndoBERT_copy.ipynb, yang BELUM menerapkan perbaikan "split multi-emoji
per baris" seperti pada pipeline XGBoost (preprocessing_v2.py). Artinya baris
emoji_dict.csv yang berisi lebih dari 1 emoji dalam satu sel kemungkinan TIDAK terbaca
dengan benar di sini. Ini kemungkinan inkonsistensi antara kedua pipeline preprocessing-mu
-- pertimbangkan menerapkan perbaikan yang sama (lihat preprocessing_v2.py) supaya kedua
pipeline benar-benar sejalan, lalu latih ulang model IndoBERT kalau memang diubah.
"""
import os
import re
import pandas as pd

DOMAIN_PROTECT = {
    'bpjs', 'mbg', 'djp', 'coretax', 'spt', 'npwp', 'ktp', 'nik',
    'login', 'error', 'update', 'upload', 'download', 'website',
    'online', 'offline', 'server', 'sistem', 'aplikasi'
}

slang_dict = {}
emoji_dict = {}
_SIAP = False


def _ws(t):
    """Bersihkan whitespace berlebihan"""
    return re.sub(r' {2,}', ' ', t).strip()


def muat_dictionary_indobert(base_dir, verbose=True):
    """Panggil SEKALI di awal notebook sebelum memakai preprocess_indobert().
    base_dir = folder yang berisi subfolder 'dictionaries/' (slang_dict.csv, emoji_dict.csv),
    sesuai path yang dipakai di Preprocessing_IndoBERT_copy.ipynb."""
    global slang_dict, emoji_dict, _SIAP

    # ── Slang dict ────────────────────────────────────────────────────────────
    SLANG_FILE = os.path.join(base_dir, 'dictionaries/slang_dict.csv')
    if os.path.exists(SLANG_FILE):
        df_slang = pd.read_csv(SLANG_FILE)
        slang_raw = dict(zip(
            df_slang['slang_list'].str.lower().str.strip(),
            df_slang['baku'].str.lower().str.strip()
        ))
        slang_dict = {
            k: v for k, v in slang_raw.items()
            if k not in DOMAIN_PROTECT and len(v.split()) <= 3
        }
        if verbose:
            print(f'Slang dict: {len(slang_dict):,} kata')
    else:
        slang_dict = {}
        if verbose:
            print('Slang dict tidak ditemukan -- skip')

    # ── Emoji dict (kata deskriptif, bukan nama kelas) ──────────────────────────
    EMOJI_FILE = os.path.join(base_dir, 'dictionaries/emoji_dict.csv')
    if os.path.exists(EMOJI_FILE):
        df_em = pd.read_csv(EMOJI_FILE)
        _kls = {'keluhan': 'marah', 'pujian': 'senang', 'saran': 'harap'}
        emoji_dict = {}
        for _, row in df_em.iterrows():
            kata = _kls.get(str(row['klasifikasi']).lower().strip(), '')
            if kata:
                emoji_dict[str(row['emoji_list'])] = kata
        if verbose:
            print(f'Emoji dict: {len(emoji_dict):,} emoji -> kata deskriptif')
    else:
        emoji_dict = {
            '\U0001F621': 'marah', '\U0001F620': 'marah', '\U0001F92C': 'marah',
            '\U0001F62D': 'sedih', '\U0001F614': 'sedih', '\U0001F615': 'bingung',
            '\U0001F60A': 'senang', '\U0001F601': 'senang', '\U0001F64F': 'terima kasih',
            '\U0001F44D': 'bagus', '\U0001F44F': 'bagus', '\U0001F525': 'semangat',
            '\U0001F602': 'lucu', '\U0001F923': 'lucu',
        }
        if verbose:
            print(f'Emoji dict (builtin): {len(emoji_dict):,} emoji')

    if verbose:
        print('\nSemua kamus IndoBERT siap!')
    _SIAP = True


def preprocess_indobert(text):
    """Pipeline IndoBERT -- 5 Tahap Minimal, identik dengan
    Preprocessing_IndoBERT_copy.ipynb. Mempertahankan huruf kapital, tanda baca,
    stopword, dan konteks kalimat -- biarkan BertTokenizer yang menangani tokenisasi.
    Panggil muat_dictionary_indobert() dulu sebelum memanggil fungsi ini."""
    if not _SIAP:
        raise RuntimeError(
            "Dictionary belum dimuat! Panggil muat_dictionary_indobert(BASE_DIR) dulu "
            "sebelum memanggil preprocess_indobert()."
        )

    # P1: Normalisasi newline
    t = re.sub(r'\r\n|\r|\n', ' ', str(text))
    t = _ws(t)

    # P2: Hapus URL, mention, hashtag
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'@\w+', '', t)
    t = re.sub(r'#\w+', '', t)
    t = _ws(t)

    # P3: Konversi emoji -> kata deskriptif
    for em, kata in emoji_dict.items():
        t = t.replace(em, f' {kata} ')
    t = re.sub(r'[^\x00-\x7F\u00C0-\u024F]+', ' ', t)
    t = _ws(t)

    # P4: Normalisasi huruf berulang
    t = re.sub(r'(.)\1{2,}', r'\1\1', t)
    t = _ws(t)

    # P5: Koreksi slang (pertahankan huruf kapital asli)
    words = t.split()
    result = []
    for w in words:
        wl = w.lower()
        if wl in DOMAIN_PROTECT:
            result.append(w)
        elif wl in slang_dict:
            result.append(slang_dict[wl])
        else:
            result.append(w)
    t = ' '.join(result)
    t = _ws(t)

    return t
