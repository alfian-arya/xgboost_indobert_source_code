"""
Aplikasi Demo Klasifikasi Sentimen Komentar Layanan Digital Pemerintah
Versi 2.0 — Semua Model Berjalan Sekaligus pada Satu Input Komentar

Jalankan: streamlit run sentiment_app_v2.py
"""

import os, re, pickle, warnings, io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR  = r'C:\Users\Lenovo\Downloads\skrips_code\xgboost_indobert_method'
MODEL_DIR = os.path.join(BASE_DIR, 'models')
BASE_DIR2 = r'C:\Users\Asus\Desktop\SKRIPSI\1. MAIN'   # fallback

# ── KONSTANTA ────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['keluhan', 'saran', 'pujian']
LABEL_MAP    = {0: 'keluhan', 1: 'saran', 2: 'pujian'}
CLASS_COLORS = {'keluhan': '#EF4444', 'saran': '#3B82F6', 'pujian': '#22C55E'}
CLASS_EMOJI  = {'keluhan': '⚠️', 'saran': '💡', 'pujian': '⭐'}
CLASS_BG     = {'keluhan': '#FEF2F2', 'saran': '#EFF6FF', 'pujian': '#F0FDF4'}

# ── DAFTAR SEMUA MODEL ───────────────────────────────────────────────────────
ALL_MODELS = [
    {
        'id'       : 'indobert_bgab',
        'name'     : 'IndoBERT',
        'scenario' : 'B-Gabungan',
        'type'     : 'indobert',
        'path'     : os.path.join(MODEL_DIR, 'indobert\indobert_B_Gabungan'),
        'acc'      : '95.37%', 'f1': '0.9331',
        'badge'    : '🏆 Terbaik',
    },
    {
        'id'       : 'indobert_agab',
        'name'     : 'IndoBERT',
        'scenario' : 'A-Gabungan',
        'type'     : 'indobert',
        'path'     : os.path.join(MODEL_DIR, 'indobert\indobert_A_Gabungan'),
        'acc'      : '94.98%', 'f1': '0.9269',
        'badge'    : '',
    },
    {
        'id'       : 'indobert_aman',
        'name'     : 'IndoBERT',
        'scenario' : 'A-Manual',
        'type'     : 'indobert',
        'path'     : os.path.join(MODEL_DIR, 'indobert\indobert_A_Manual'),
        'acc'      : '87.82%', 'f1': '0.8449',
        'badge'    : '',
    },
    {
        'id'       : 'xgb_tuned_bgab',
        'name'     : 'XGBoost (Tuned)',
        'scenario' : 'B-Gabungan',
        'type'     : 'sklearn',
        'key'      : 'XGB_v2_Tuned_B_Gabungan',
        'acc'      : '84.89%', 'f1': '0.8073',
        'badge'    : '⚡ Terbaik XGB',
    },
    {
        'id'       : 'xgb_tuned_agab',
        'name'     : 'XGBoost (Tuned)',
        'scenario' : 'A-Gabungan',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'XGB_v2_Tuned_A_Gabungan',
        'acc'      : '84.48%', 'f1': '0.8037',
        'badge'    : '',
    },
    {
        'id'       : 'xgb_tuned_aman',
        'name'     : 'XGBoost (Tuned)',
        'scenario' : 'A-Manual',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'XGB_v2_Tuned_A_Manual',
        'acc'      : '80.79%', 'f1': '0.7721',
        'badge'    : '',
    },
    {
        'id'       : 'xgb_notune_aman',
        'name'     : 'XGBoost (NoTune)',
        'scenario' : 'A-Manual',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'XGB_v2_NoTune_A_Manual',
        'acc'      : '79.75%', 'f1': '0.7632',
        'badge'    : '',
    },
    {
        'id'       : 'xgb_notune_agab',
        'name'     : 'XGBoost (NoTune)',
        'scenario' : 'A-Gabungan',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'XGB_v2_NoTune_A_Gabungan',
        'acc'      : '80.82%', 'f1': '0.7699',
        'badge'    : '',
    },
    {
        'id'       : 'xgb_notune_bgab',
        'name'     : 'XGBoost (NoTune)',
        'scenario' : 'B-Gabungan',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'XGB_v2_NoTune_B_Gabungan',
        'acc'      : '80.79%', 'f1': '0.7721',
        'badge'    : '',
    },
    {
        'id'       : 'lr_agab',
        'name'     : 'Logistic Regression',
        'scenario' : 'A-Gabungan',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'LR_v2_A_Gabungan',
        'acc'      : '86.16%', 'f1': '0.8212',
        'badge'    : '⚡ Terbaik LR',
    },
    {
        'id'       : 'lr_bgab',
        'name'     : 'Logistic Regression',
        'scenario' : 'B-Gabungan',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'LR_v2_B_Gabungan',
        'acc'      : '85.38%', 'f1': '0.8138',
        'badge'    : '',
    },
    {
        'id'       : 'lr_aman',
        'name'     : 'Logistic Regression',
        'scenario' : 'A-Manual',
        'type'     : 'sklearn',
        'path'     : os.path.join(MODEL_DIR, 'ml'),
        'key'      : 'LR_v2_A_Manual',
        'acc'      : '81.40%', 'f1': '0.7804',
        'badge'    : '',
    },
]

# ── PREPROCESSING v2.0 ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Memuat resource preprocessing v2.0...')
def load_prep_resources():
    res = {
        'stemmer'      : None,
        'stopwords'    : set(),
        'slang_dict'   : {},
        'profanity_set': set(),
        'kbbi_set'     : set(),
        'emoji_dict'   : {},
        'stem_protect' : set(),
    }
    base = BASE_DIR if os.path.isdir(BASE_DIR) else BASE_DIR2

    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        res['stemmer']   = StemmerFactory().create_stemmer()
        res['stopwords'] = set(StopWordRemoverFactory().get_stop_words())
    except Exception:
        pass

    for fname, key, kcol, vcol in [
        ('slang_dict.csv',    'slang_dict',    'slang_list',    'baku'),
        ('profanity_dict.csv','profanity_set',  'profanity_list', None),
        ('kbbi_dict.csv',     'kbbi_set',       'kbbi_list',      None),
        ('stopword_dict.csv', 'stopwords',      'stopword_list',  None),
    ]:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            continue
        try:
            df = pd.read_csv(fpath).apply(lambda x: x.astype(str).str.lower())
            if vcol:
                res[key] = dict(zip(df[kcol], df[vcol]))
            elif key == 'stopwords':
                res['stopwords'] |= set(df[kcol].unique())
            else:
                res[key] = set(df[kcol].unique())
        except Exception:
            pass

    ep = os.path.join(base, 'emoji_dict.csv')
    if os.path.exists(ep):
        try:
            df_em = pd.read_csv(ep)
            _map  = {'keluhan':'marah','pujian':'senang','saran':'harap'}
            for _, row in df_em.iterrows():
                kata = _map.get(str(row.get('klasifikasi','')).lower().strip(),'')
                if not kata:
                    continue
                for em in str(row['emoji_list']).split(','):
                    em = em.strip()
                    if em and not em.isascii() and em not in res['emoji_dict']:
                        res['emoji_dict'][em] = kata
        except Exception:
            pass

    NEGASI = {'tidak','belum','bukan','jangan','tanpa','kurang',
              'tak','nggak','ga','gak','ndak','enggak'}
    res['stopwords'] -= NEGASI

    res['stem_protect'] = {
        'penangguhan','pembayaran','pelayanan','penanganan','perbaikan',
        'kepesertaan','administrasi','pendaftaran','pembatalan','penggunaan',
        'pembuatan','penolakan','keterlambatan','pengajuan','pencairan',
        'pemotongan','permohonan','perpanjangan','pelaporan','pengembalian',
        'penambahan','pengurangan','penerbitan','pembaruan','perubahan',
        'semangat','pemerintah','penanggung','bergantung','berguna',
    }
    return res


def preprocess_xgb(text, res):
    t = str(text).replace('\n',' ').replace('\r',' ')
    t = re.sub(r'https?://\S+|www\.\S+','',t)
    t = re.sub(r'@\w+','',t)
    t = re.sub(r'#',' ',t)
    for em, kata in res['emoji_dict'].items():
        t = t.replace(em, f' {kata} ')
    t = re.sub(r'[^\x00-\x7F]+',' ',t)
    t = re.sub(r'[^a-zA-Z0-9\s_]',' ',t)
    t = t.lower()
    tokens = t.split()
    tokens = [re.sub(r'(.)\1{2,}',r'\1\1',tok) for tok in tokens]
    tokens = ['badword' if tok in res['profanity_set'] else tok for tok in tokens]
    tokens = [res['slang_dict'].get(tok,tok) for tok in tokens]
    NEGASI = {'tidak','belum','bukan','jangan','tanpa','kurang',
              'tak','nggak','ga','gak','ndak','enggak'}
    tokens = [t for t in tokens if t in NEGASI or t not in res['stopwords']]
    result, i = [], 0
    while i < len(tokens):
        if tokens[i] in NEGASI and i+1 < len(tokens):
            result += [f'{tokens[i]}_{tokens[i+1]}', tokens[i]]
            i += 2
        else:
            result.append(tokens[i]); i += 1
    tokens = result
    stemmed = []
    for tok in tokens:
        if '_' in tok or tok == 'badword': stemmed.append(tok)
        elif tok in res['stem_protect']:   stemmed.append(tok)
        elif res['stemmer']:
            try: stemmed.append(res['stemmer'].stem(tok))
            except: stemmed.append(tok)
        else: stemmed.append(tok)
    tokens = stemmed
    if res['kbbi_set']:
        tokens = [t for t in tokens
                  if '_' in t or t=='badword' or t in ['marah','senang','harap']
                  or t in res['kbbi_set']]
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


def preprocess_bert(text, res):
    t = str(text).replace('\n',' ').replace('\r',' ')
    t = re.sub(r'https?://\S+|www\.\S+','',t)
    t = re.sub(r'@\w+','',t)
    t = re.sub(r'#',' ',t)
    for em, kata in res['emoji_dict'].items():
        t = t.replace(em, f' {kata} ')
    tokens = t.split()
    tokens = [re.sub(r'(.)\1{2,}',r'\1\1',tok) for tok in tokens]
    tokens = [res['slang_dict'].get(tok.lower(),tok) for tok in tokens]
    return ' '.join(tokens)


# ── LOAD SEMUA MODEL ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Memuat semua model v2.0...')
def load_all_models():
    loaded = {}
    for m in ALL_MODELS:
        try:
            if m['type'] == 'sklearn':
                mp = os.path.join(MODEL_DIR, f"{m['key']}.pkl")
                vp = os.path.join(MODEL_DIR, f"vec_{m['key']}.pkl")
                if os.path.exists(mp) and os.path.exists(vp):
                    loaded[m['id']] = {
                        'model': pickle.load(open(mp,'rb')),
                        'vec'  : pickle.load(open(vp,'rb')),
                        'type' : 'sklearn',
                    }
            elif m['type'] == 'indobert':
                if os.path.isdir(m['path']):
                    import torch
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    tok   = AutoTokenizer.from_pretrained(m['path'])
                    model = AutoModelForSequenceClassification.from_pretrained(m['path'])
                    model.eval()
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model.to(device)
                    loaded[m['id']] = {
                        'model'    : model,
                        'tokenizer': tok,
                        'device'   : device,
                        'type'     : 'indobert',
                    }
        except Exception:
            pass
    return loaded


# ── FUNGSI PREDIKSI ───────────────────────────────────────────────────────────
def predict_one(text, model_id, loaded, res):
    if model_id not in loaded:
        return None, {}
    obj  = loaded[model_id]
    mtype = obj['type']
    try:
        if mtype == 'sklearn':
            processed = preprocess_xgb(text, res)
            if not processed.strip():
                return None, {}
            X   = obj['vec']['selector'].transform(obj['vec']['tfidf'].transform([processed]))
            lid = int(obj['model'].predict(X)[0])
            if hasattr(obj['model'], 'predict_proba'):
                probs = obj['model'].predict_proba(X)[0]
                conf  = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}
            else:
                conf = {CLASS_NAMES[lid]: 1.0,
                        CLASS_NAMES[(lid+1)%3]: 0.0,
                        CLASS_NAMES[(lid+2)%3]: 0.0}
            return LABEL_MAP[lid], conf

        elif mtype == 'indobert':
            import torch
            processed = preprocess_bert(text, res)
            if not processed.strip():
                return None, {}
            enc = obj['tokenizer'](processed, max_length=128, truncation=True,
                                   padding='max_length', return_tensors='pt')
            enc = {k: v.to(obj['device']) for k, v in enc.items()}
            with torch.no_grad():
                logits = obj['model'](**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            lid   = int(np.argmax(probs))
            conf  = {CLASS_NAMES[i]: float(probs[i]) for i in range(3)}
            return LABEL_MAP[lid], conf
    except Exception:
        return None, {}


# ── VISUALISASI ───────────────────────────────────────────────────────────────
def mini_conf_bar(conf: dict, predicted: str):
    """Bar chart confidence horizontal kecil."""
    fig, ax = plt.subplots(figsize=(4, 1.6))
    vals = [conf.get(c, 0)*100 for c in CLASS_NAMES]
    colors = [CLASS_COLORS[c] if c == predicted else '#D1D5DB' for c in CLASS_NAMES]
    bars = ax.barh(CLASS_NAMES, vals, color=colors, height=0.5)
    ax.set_xlim(0, 110)
    ax.set_xticks([])
    for bar, val in zip(bars, vals):
        ax.text(val+1, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=8)
    ax.spines[['top','right','bottom','left']].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    plt.tight_layout(pad=0.5)
    return fig


def comparison_radar(all_results):
    """Bar chart perbandingan confidence semua model."""
    valid = [(m, r) for m, r in all_results.items()
             if r['label'] and r['label'] in CLASS_NAMES]
    if len(valid) < 2:
        return None
    names = [f"{m['name']}\n{m['scenario']}" for m_id, r in valid
             for m in ALL_MODELS if m['id'] == m_id]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    fig.suptitle('Confidence Semua Model per Kelas', fontsize=11, y=1.02)
    for col_idx, cls in enumerate(CLASS_NAMES):
        vals   = [r['conf'].get(cls, 0)*100 for _, r in valid]
        colors = [CLASS_COLORS[cls] if r['label']==cls else '#D1D5DB' for _, r in valid]
        y_pos  = range(len(valid))
        label_names = []
        for m_id, _ in valid:
            for m in ALL_MODELS:
                if m['id'] == m_id:
                    label_names.append(f"{m['name']} {m['scenario']}")
                    break
        axes[col_idx].barh(label_names, vals, color=colors, height=0.6)
        axes[col_idx].set_xlim(0, 110)
        axes[col_idx].set_title(f'{CLASS_EMOJI[cls]} {cls.title()}',
                                 color=CLASS_COLORS[cls], fontsize=10)
        for i, v in enumerate(vals):
            axes[col_idx].text(v+1, i, f'{v:.1f}%', va='center', fontsize=8)
        axes[col_idx].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig


def plot_batch_dist(df: pd.DataFrame):
    """Pie + bar chart distribusi batch."""
    counts = df['prediksi'].value_counts()
    valid  = counts[counts.index.isin(CLASS_NAMES)]
    if valid.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = [CLASS_COLORS[c] for c in valid.index]
    axes[0].pie(valid.values,
                labels=[f'{c.title()}\n({v})' for c,v in zip(valid.index,valid.values)],
                colors=colors, autopct='%1.1f%%', startangle=140)
    axes[0].set_title('Distribusi Kelas Sentimen')
    axes[1].bar([c.title() for c in valid.index], valid.values, color=colors)
    axes[1].set_ylabel('Jumlah')
    axes[1].set_title('Jumlah per Kelas')
    for i, v in enumerate(valid.values):
        axes[1].text(i, v+0.2, str(v), ha='center', fontsize=11)
    plt.tight_layout()
    return fig


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='Klasifikasi Sentimen v2.0',
        page_icon='💬',
        layout='wide',
    )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='text-align:center;color:#1D4ED8;'>
        💬 Klasifikasi Sentimen Komentar Layanan Digital Pemerintah
    </h1>
    <p style='text-align:center;color:#6B7280;font-size:14px;'>
        Pipeline v2.0 · Semua Model Berjalan Sekaligus · Negation Handling + STEM_PROTECT
    </p><hr style='border:1px solid #E5E7EB;margin-top:4px;'>
    """, unsafe_allow_html=True)

    # ── Load resource ─────────────────────────────────────────────────────────
    res     = load_prep_resources()
    loaded  = load_all_models()

    n_loaded = len(loaded)
    n_total  = len(ALL_MODELS)
    if n_loaded == 0:
        st.error(f'⚠️ Tidak ada model yang berhasil dimuat dari `{MODEL_DIR}`. '
                 'Pastikan file .pkl tersedia.')
        st.stop()

    # Sidebar info
    st.sidebar.title('📋 Status Model')
    for m in ALL_MODELS:
        ok  = m['id'] in loaded
        ico = '✅' if ok else '⬜'
        badge = f' {m["badge"]}' if m['badge'] else ''
        st.sidebar.markdown(
            f'{ico} **{m["name"]}** – {m["scenario"]}{badge}  \n'
            f'<span style="color:#9CA3AF;font-size:11px;">'
            f'Acc {m["acc"]} · F1 {m["f1"]}</span>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown('---')
    st.sidebar.markdown(
        f'**{n_loaded}/{n_total}** model dimuat  \n'
        f'{"✅ Siap" if n_loaded > 0 else "❌ Tidak ada model"}'
    )

    # ── Tab ───────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(['📝 Klasifikasi Teks (Semua Model)', '📂 Upload File CSV'])

    # ══════════════════════════════════════════════════════════════════════════
    #   TAB 1: Single Input → Semua Model Sekaligus
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader('Masukkan Komentar')
        user_text = st.text_area(
            'Ketik atau tempelkan komentar di sini:',
            height=120,
            placeholder='Contoh: BPJS saya tidak aktif padahal sudah bayar tiap bulan...',
        )

        col_btn, col_info, _ = st.columns([1, 2, 3])
        with col_btn:
            run_btn = st.button('🔍 Klasifikasi Semua Model', type='primary',
                                use_container_width=True)
        with col_info:
            st.markdown(
                f'<span style="color:#6B7280;font-size:13px;">'
                f'Akan menjalankan <b>{n_loaded}</b> model secara bersamaan</span>',
                unsafe_allow_html=True,
            )

        if run_btn and user_text.strip():
            with st.spinner(f'Menjalankan {n_loaded} model...'):
                all_results = {}
                for m in ALL_MODELS:
                    if m['id'] not in loaded:
                        continue
                    label, conf = predict_one(user_text, m['id'], loaded, res)
                    all_results[m['id']] = {'label': label, 'conf': conf}

            # ── Ringkasan konsensus ─────────────────────────────────────────
            valid_preds = [r['label'] for r in all_results.values()
                           if r['label'] in CLASS_NAMES]
            if valid_preds:
                from collections import Counter
                consensus = Counter(valid_preds).most_common(1)[0][0]
                agree_pct = valid_preds.count(consensus) / len(valid_preds) * 100

                st.markdown('---')
                st.markdown('### 🗳️ Konsensus Semua Model')
                col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
                with col_c1:
                    c = CLASS_COLORS[consensus]
                    st.markdown(f"""
                    <div style='background:{c}18;border:2px solid {c};
                                border-radius:12px;padding:18px;text-align:center;'>
                        <div style='font-size:2.5rem;'>{CLASS_EMOJI[consensus]}</div>
                        <div style='color:{c};font-size:1.3rem;font-weight:700;
                                    text-transform:uppercase;'>{consensus}</div>
                        <div style='color:#6B7280;font-size:0.85rem;'>Prediksi Konsensus</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c2:
                    st.metric('Model Setuju', f'{valid_preds.count(consensus)}/{len(valid_preds)}',
                              f'{agree_pct:.0f}% konsensus')
                with col_c3:
                    # Mini pie konsensus
                    cnt = Counter(valid_preds)
                    fig_pie, ax_pie = plt.subplots(figsize=(3.5, 3.5))
                    colors_pie = [CLASS_COLORS.get(c,'#9CA3AF') for c in cnt.keys()]
                    ax_pie.pie(cnt.values(),
                               labels=[f'{c.title()} ({v})' for c,v in cnt.items()],
                               colors=colors_pie, autopct='%1.0f%%', startangle=140)
                    ax_pie.set_title('Suara Model', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig_pie, use_container_width=True)
                    plt.close()

            # ── Grid hasil per model ────────────────────────────────────────
            st.markdown('---')
            st.markdown('### 🔬 Hasil Per Model')

            # Susun dalam grid 3 kolom
            loaded_models = [m for m in ALL_MODELS if m['id'] in all_results]
            n_cols = 3
            rows   = [loaded_models[i:i+n_cols]
                      for i in range(0, len(loaded_models), n_cols)]

            # Tampilkan setiap model dalam container native Streamlit
            for m in loaded_models:
                r        = all_results[m['id']]
                label    = r['label'] or 'N/A'
                conf     = r['conf']
                c        = CLASS_COLORS.get(label, '#9CA3AF')
                emoji    = CLASS_EMOJI.get(label, '❓')
                max_conf = max(conf.values()) * 100 if conf else 0

                with st.container(border=True):
                    # Baris atas: nama model + badge + prediksi
                    col_a, col_b, col_c = st.columns([3, 2, 2])

                    with col_a:
                        badge = f' `{m["badge"]}`' if m['badge'] else ''
                        st.markdown(f'**{m["name"]}** — {m["scenario"]}{badge}')
                        st.caption(f'Akurasi: {m["acc"]} · Macro F1: {m["f1"]}')

                    with col_b:
                        st.markdown(
                            f'<div style="text-align:center;">' 
                            f'<span style="font-size:1.8rem;">{emoji}</span><br>' 
                            f'<span style="color:{c};font-weight:700;font-size:1rem;' 
                            f'text-transform:uppercase;">{label}</span><br>' 
                            f'<span style="color:#6B7280;font-size:11px;">{max_conf:.1f}% confidence</span>' 
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with col_c:
                        # Progress bar per kelas
                        for cls in CLASS_NAMES:
                            pct = conf.get(cls, 0)
                            clr = CLASS_COLORS[cls]
                            bold = '**' if cls == label else ''
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">' 
                                f'<span style="width:60px;font-size:11px;color:#374151;">{bold}{cls}{bold}</span>' 
                                f'<div style="flex:1;background:#E5E7EB;border-radius:4px;height:8px;">' 
                                f'<div style="width:{pct*100:.1f}%;background:{clr};height:8px;border-radius:4px;"></div>' 
                                f'</div>' 
                                f'<span style="font-size:10px;color:#6B7280;width:38px;text-align:right;">{pct*100:.1f}%</span>' 
                                f'</div>',
                                unsafe_allow_html=True
                            )

            # ── Tabel perbandingan ──────────────────────────────────────────
            st.markdown('---')
            st.markdown('### 📊 Tabel Perbandingan Lengkap')
            table_rows = []
            for m in loaded_models:
                r     = all_results[m['id']]
                label = r['label'] or 'N/A'
                conf  = r['conf']
                table_rows.append({
                    'Model'                    : m['name'],
                    'Skenario'                 : m['scenario'],
                    'Prediksi'                 : label.upper() if label != 'N/A' else 'N/A',
                    'Conf. Keluhan (%)'        : f"{conf.get('keluhan',0)*100:.1f}",
                    'Conf. Saran (%)'          : f"{conf.get('saran',0)*100:.1f}",
                    'Conf. Pujian (%)'         : f"{conf.get('pujian',0)*100:.1f}",
                    'Max Confidence (%)'       : f"{max(conf.values())*100:.1f}" if conf else '–',
                    'Accuracy'                 : m['acc'],
                    'Macro F1'                 : m['f1'],
                })
            df_tbl = pd.DataFrame(table_rows)

            def color_pred(val):
                val_l = val.lower()
                if val_l in CLASS_COLORS:
                    return (f'background-color:{CLASS_COLORS[val_l]}20;'
                            f'color:{CLASS_COLORS[val_l]};font-weight:bold;')
                return ''

            st.dataframe(
                df_tbl.style.applymap(color_pred, subset=['Prediksi']),
                use_container_width=True,
                hide_index=True,
            )

            # Download tabel
            csv_tbl = df_tbl.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                '📥 Download Tabel Perbandingan (CSV)',
                data      = csv_tbl.encode('utf-8-sig'),
                file_name = 'perbandingan_model.csv',
                mime      = 'text/csv',
            )

            # ── Grafik perbandingan ─────────────────────────────────────────
            fig_cmp = comparison_radar(all_results)
            if fig_cmp:
                st.markdown('---')
                st.markdown('### 📈 Grafik Confidence Semua Model per Kelas')
                st.pyplot(fig_cmp, use_container_width=True)
                plt.close()

            # ── Preprocessing preview ───────────────────────────────────────
            with st.expander('🔬 Lihat Hasil Preprocessing v2.0'):
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.markdown('**Teks Asli:**')
                    st.code(user_text, language=None)
                with col_p2:
                    proc_xgb  = preprocess_xgb(user_text, res)
                    proc_bert = preprocess_bert(user_text, res)
                    st.markdown('**Output Pipeline XGBoost/LR (15 tahap):**')
                    st.code(proc_xgb or '(kosong setelah filtering)', language=None)
                    neg_tok = [t for t in proc_xgb.split() if '_' in t and '__' not in t]
                    if neg_tok:
                        st.success(f'🔗 Token negasi: `{", ".join(neg_tok[:8])}`')
                    st.markdown('**Output Pipeline IndoBERT (5 tahap):**')
                    st.code(proc_bert[:300], language=None)

        elif run_btn:
            st.warning('⚠️ Silakan masukkan komentar terlebih dahulu.')

    # ══════════════════════════════════════════════════════════════════════════
    #   TAB 2: CSV Batch (pilih satu model untuk efisiensi)
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader('Upload File CSV untuk Klasifikasi Batch')
        st.info('💡 Untuk batch CSV, pilih **satu model** agar proses lebih cepat.')

        # Pilih model untuk batch
        model_opts = {
            f'{m["name"]} — {m["scenario"]} (F1={m["f1"]})': m['id']
            for m in ALL_MODELS if m['id'] in loaded
        }
        if not model_opts:
            st.error('Tidak ada model yang tersedia.')
        else:
            sel_model_name = st.selectbox(
                '⚙️ Pilih Model untuk Batch:',
                list(model_opts.keys()),
                index=0,
            )
            sel_model_id = model_opts[sel_model_name]
            sel_mtype    = next(m['type'] for m in ALL_MODELS if m['id'] == sel_model_id)

            uploaded = st.file_uploader('Pilih file CSV:', type=['csv'])

            if uploaded:
                df_raw = None
                for enc in ['utf-8','utf-8-sig','latin1','cp1252']:
                    try:
                        uploaded.seek(0)
                        df_raw = pd.read_csv(uploaded, encoding=enc)
                        break
                    except Exception:
                        continue

                if df_raw is None:
                    st.error('Gagal membaca CSV.')
                else:
                    st.success(f'✅ {len(df_raw):,} baris dimuat — {len(df_raw.columns)} kolom')
                    st.dataframe(df_raw.head(3), use_container_width=True)

                    # Pilih kolom teks
                    candidates = [c for c in df_raw.columns
                                  if any(k in c.lower()
                                         for k in ['text','teks','komentar','comment','isi'])]
                    default_col = candidates[0] if candidates else df_raw.columns[0]
                    text_col = st.selectbox(
                        '📌 Kolom teks komentar:',
                        df_raw.columns.tolist(),
                        index=df_raw.columns.tolist().index(default_col),
                    )

                    col_n, col_btn2 = st.columns([2, 1])
                    with col_n:
                        max_rows = st.number_input('Maks. baris:', 100, 50000, 1000, step=100)
                    with col_btn2:
                        batch_btn = st.button('🚀 Mulai Klasifikasi Batch',
                                              type='primary', use_container_width=True)

                    if batch_btn:
                        texts = df_raw[text_col].fillna('').astype(str).tolist()[:int(max_rows)]
                        prog  = st.progress(0, text='Memproses...')
                        rows  = []
                        n     = len(texts)

                        for i, txt in enumerate(texts):
                            label, conf = predict_one(txt, sel_model_id, loaded, res)
                            rows.append({
                                'prediksi'     : label or 'tidak_valid',
                                'conf_keluhan' : round(conf.get('keluhan',0)*100,2),
                                'conf_saran'   : round(conf.get('saran',  0)*100,2),
                                'conf_pujian'  : round(conf.get('pujian', 0)*100,2),
                            })
                            if (i+1) % max(1,n//50) == 0 or i == n-1:
                                prog.progress((i+1)/n, text=f'{i+1}/{n} komentar')
                        prog.empty()

                        df_out = pd.concat([
                            df_raw.iloc[:n].reset_index(drop=True),
                            pd.DataFrame(rows)
                        ], axis=1)

                        st.success(f'✅ {n:,} komentar selesai diklasifikasikan.')

                        # Distribusi
                        fig_d = plot_batch_dist(df_out)
                        if fig_d:
                            st.subheader('📊 Distribusi Hasil')
                            st.pyplot(fig_d, use_container_width=True)
                            plt.close()

                        # Metrik
                        valid_df = df_out[df_out['prediksi'].isin(CLASS_NAMES)]
                        cnt = valid_df['prediksi'].value_counts()
                        cols_m = st.columns(4)
                        for i, cls in enumerate(CLASS_NAMES):
                            v = cnt.get(cls, 0)
                            cols_m[i].metric(f'{CLASS_EMOJI[cls]} {cls.title()}',
                                             f'{v:,}', f'{v/n*100:.1f}%')
                        cols_m[3].metric('Total Valid', f'{len(valid_df):,}',
                                         f'{len(valid_df)/n*100:.1f}%')

                        # Preview
                        st.subheader('🔍 Preview 20 Baris')
                        st.dataframe(df_out.head(20), use_container_width=True)

                        # Download
                        csv_dl = df_out.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            '📥 Download Hasil (CSV)',
                            data      = csv_dl.encode('utf-8-sig'),
                            file_name = f'batch_{sel_model_id}.csv',
                            mime      = 'text/csv',
                            use_container_width=True,
                        )

    # Footer
    st.markdown('---')
    st.markdown(
        '<p style="text-align:center;color:#9CA3AF;font-size:11px;">'
        'Klasifikasi Sentimen Komentar Layanan Digital Pemerintah Indonesia · '
        'Pipeline v2.0 · Teknik Informatika Universitas Hasanuddin 2026'
        '</p>', unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
