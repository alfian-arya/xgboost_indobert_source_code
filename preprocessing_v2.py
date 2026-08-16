r"""
preprocessing_v2.py
Modul preprocessing v2.0 -- diekstrak dari Preprocessing_v2_Final.ipynb
supaya bisa di-import dari notebook mana pun (termasuk notebook tahapan XGBoost),
tanpa perlu copy-paste ulang kode preprocessing.

Cara pakai di notebook lain:
    import sys
    sys.path.insert(0, r'C:\Users\Lenovo\Downloads\skrips_code\xgboost_indobert_method')  # folder tempat file ini disimpan
    from preprocessing_v2 import preprocess_v2, muat_semua_dictionary

    muat_semua_dictionary(BASE_DIR)   # panggil SEKALI di awal notebook
    hasil = preprocess_v2("teks komentar di sini")
"""
import os
import re
import pandas as pd
from functools import lru_cache
from collections import Counter

# ── FIX 1: NEGASI_PROTECT ─────────────────────────────────────────────────────
NEGASI_PROTECT = {
    'tidak', 'tak', 'bukan', 'belum',
    'tanpa', 'tiada', 'jangan',
    'padahal', 'seharusnya', 'harusnya',
    'malah', 'justru', 'namun', 'tapi', 'tetapi',
    'kurang', 'susah', 'sulit', 'gagal', 'rusak',
    'lambat', 'lemot', 'masalah', 'kendala',
}

# ── FIX 2: STEM_PROTECT ────────────────────────────────────────────────────────
STEM_PROTECT = {
    'penangguhan',
    'pembayaran', 'pelayanan', 'pendaftaran',
    'penggantian', 'perbaikan', 'keterlambatan',
    'penolakan', 'pembatalan', 'perpanjangan',
    'penonaktifan', 'pengaduan', 'pengaktifan',
    'pelaporan', 'pengembalian', 'pemeriksaan', 'penagihan',
    'pembagian', 'penyaluran', 'pengadaan', 'pelaksanaan',
    'bpjs', 'mbg', 'djp', 'coretax', 'spt', 'npwp',
    'login', 'error', 'update', 'upload', 'download',
    'website', 'online', 'offline', 'server', 'sistem',
    'aplikasi', 'faskes', 'puskesmas', 'rujukan',
    'klaim', 'premi', 'iuran', 'pajak', 'akun', 'email',
    'pandawa', 'perbaiki'
}
STEM_PROTECT.update(NEGASI_PROTECT)

# ── FIX 3: NEGASI_COMBINE ──────────────────────────────────────────────────────
NEGASI_COMBINE = {'tidak', 'tak', 'bukan', 'belum', 'tanpa', 'tiada', 'jangan'}

# ── Variabel global yang diisi oleh muat_semua_dictionary() ───────────────────
_stemmer_obj = None
emoji_dict = {}
profanity_set = set()
slang_dict = {}
stopwords_id = set()
kbbi_words = set()
USE_KBBI = False
_SIAP = False


@lru_cache(maxsize=300000)
def stem_word(w):
    if w in STEM_PROTECT:
        return w
    return _stemmer_obj.stem(w)


def negation_handling(tokens):
    """FIX 3: Gabungkan kata negasi dengan kata sesudahnya.
    'tidak' + 'aktif' -> 'tidak_aktif' (token baru bermakna). Token asli 'tidak' tetap disimpan.
    Referensi: Kalaivani et al. (2023)"""
    result = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if (w in NEGASI_COMBINE
                and i + 1 < len(tokens)
                and tokens[i + 1] not in NEGASI_COMBINE
                and '_' not in tokens[i + 1]):
            result.append(f'{w}_{tokens[i+1]}')
            result.append(w)
            i += 2
        else:
            result.append(w)
            i += 1
    return result


def muat_semua_dictionary(base_dir, verbose=True):
    """Panggil SEKALI di awal notebook sebelum memakai preprocess_v2().
    base_dir = folder yang berisi emoji_dict.csv, profanity_dict.csv, slang_dict.csv,
    stopword_dict.csv, kbbi_dict.csv (sesuaikan kalau file kamu ada di subfolder
    'dictionaries/', tinggal ganti base_dir jadi os.path.join(BASE_DIR, 'dictionaries'))."""
    global _stemmer_obj, emoji_dict, profanity_set, slang_dict, stopwords_id, kbbi_words, USE_KBBI, _SIAP

    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    if verbose:
        print('Loading semua dictionary...')
        print('=' * 50)

    _factory = StemmerFactory()
    _stemmer_obj = _factory.create_stemmer()
    stem_word.cache_clear()

    # 2. Emoji dict -> kata deskriptif
    EMOJI_FILE = os.path.join(base_dir, 'emoji_dict.csv')
    emoji_dict = {}
    if os.path.exists(EMOJI_FILE):
        _df_em = pd.read_csv(EMOJI_FILE)
        _map = {'keluhan': 'marah', 'pujian': 'senang', 'saran': 'harap'}
        for _, row in _df_em.iterrows():
            kls = str(row['klasifikasi']).lower().strip()
            kata = _map.get(kls, '')
            if not kata:
                continue
            parts = [e.strip() for e in str(row['emoji_list']).split(',')]
            for emoji_char in parts:
                if not emoji_char or emoji_char == 'nan':
                    continue
                if emoji_char.isascii():
                    continue
                if emoji_char not in emoji_dict:
                    emoji_dict[emoji_char] = kata
    if verbose:
        dist = Counter(emoji_dict.values())
        print(f'\u2705 emoji_dict     : {len(emoji_dict)} emoji individu '
              f'(marah={dist["marah"]}, senang={dist["senang"]}, harap={dist["harap"]})')

    # 3. Profanity dict
    PROFANITY_FILE = os.path.join(base_dir,  'profanity_dict.csv')
    profanity_set = set()
    if os.path.exists(PROFANITY_FILE):
        _df_prof = pd.read_csv(PROFANITY_FILE)
        profanity_set = set(_df_prof.iloc[:, 0].astype(str).str.lower().str.strip())
    if verbose:
        print(f'\u2705 profanity_set  : {len(profanity_set):,} kata kasar -> [BADWORD]')

    # 4. Slang dict
    SLANG_FILE = os.path.join(base_dir, 'slang_dict.csv')
    slang_dict = {}
    if os.path.exists(SLANG_FILE):
        _df_s = pd.read_csv(SLANG_FILE)
        slang_raw = dict(zip(
            _df_s['slang_list'].str.lower().str.strip(),
            _df_s['baku'].str.lower().str.strip()
        ))
        slang_dict = {k: v for k, v in slang_raw.items()
                      if k not in STEM_PROTECT and len(v.split()) <= 3}
    if verbose:
        print(f'\u2705 slang_dict     : {len(slang_dict):,} kata slang -> baku')

    # 5. Stopword (dengan NEGASI_PROTECT)
    _sw_factory = StopWordRemoverFactory()
    stopwords_id = set(_sw_factory.get_stop_words())
    SW_FILE = os.path.join(base_dir, 'stopword_dict.csv')
    if os.path.exists(SW_FILE):
        extra_sw = set(pd.read_csv(SW_FILE).iloc[:, 0].str.lower().str.strip())
        stopwords_id.update(extra_sw)
    stopwords_id -= NEGASI_PROTECT
    if verbose:
        print(f'\u2705 stopwords_id   : {len(stopwords_id):,} kata (negasi sudah diproteksi)')

    # 6. KBBI dict
    KBBI_FILE = os.path.join(base_dir, 'kbbi_dict.csv')
    kbbi_words = set()
    USE_KBBI = False
    if os.path.exists(KBBI_FILE):
        kbbi_words = set(pd.read_csv(KBBI_FILE)['kbbi_list'].str.lower().str.strip())
        kbbi_words.update(STEM_PROTECT)
        kbbi_words.update(NEGASI_PROTECT)
        kbbi_words.add('badword')
        USE_KBBI = True
    if verbose:
        print(f'\u2705 kbbi_words     : {len(kbbi_words):,} kata (termasuk negasi, domain, badword)')
        print('\n\u2705 Semua dictionary berhasil dimuat! preprocess_v2() siap dipakai.')

    _SIAP = True


def preprocess_v2(text):
    """Pipeline preprocessing v2.0 (15 langkah P1-P15) -- identik dengan
    Preprocessing_v2_Final.ipynb. Panggil muat_semua_dictionary() dulu sebelum ini."""
    if not _SIAP:
        raise RuntimeError(
            "Dictionary belum dimuat! Panggil muat_semua_dictionary(BASE_DIR) dulu "
            "sebelum memanggil preprocess_v2()."
        )

    def ws(t):
        return re.sub(r' {2,}', ' ', t).strip()

    # P1: Normalisasi newline
    t = re.sub(r'\r\n|\r|\n', ' ', str(text))
    t = ws(t)

    # P2: Hapus URL, mention, hashtag
    t = re.sub(r'https?://\S+|www\.\S+', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    t = re.sub(r'#\w+', ' ', t)
    t = ws(t)

    # P3: Konversi emoji -> kata deskriptif
    for em, kata in emoji_dict.items():
        t = t.replace(em, f' {kata} ')

    # P4: Hapus karakter non-ASCII (sisa emoji)
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)
    t = ws(t)

    # P5: Hapus tanda baca kecuali underscore
    t = re.sub(r'[^a-zA-Z0-9\s_]', ' ', t)
    t = ws(t)

    # P6: Case folding
    t = t.lower()

    # P7: Tokenisasi
    toks = t.split()
    if not toks:
        return ''

    # P8: Normalisasi huruf berulang
    toks = [re.sub(r'(.)\1{2,}', r'\1\1', w) for w in toks]

    # P9: Ganti profanity -> 'badword'
    toks = ['badword' if w in profanity_set else w for w in toks]

    # P10: Koreksi slang
    result = []
    for w in toks:
        if w in STEM_PROTECT or w == 'badword':
            result.append(w)
        else:
            result.extend(slang_dict.get(w, w).split())
    toks = result

    # P11: Stopword removal
    toks = [w for w in toks if w not in stopwords_id]

    # P12: Negation handling
    toks = negation_handling(toks)

    # P13: Stemming
    toks = [
        w if ('_' in w or w == 'badword') else stem_word(w)
        for w in toks
    ]

    # P14: Filter KBBI
    if USE_KBBI:
        toks = [
            w for w in toks
            if '_' in w
            or w == 'badword'
            or len(w) <= 2
            or w in kbbi_words
        ]

    # P15: Filter panjang
    toks = [w for w in toks if len(w) >= 2]

    return ' '.join(toks)
