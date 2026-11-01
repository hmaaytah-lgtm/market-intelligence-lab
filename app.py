"""Market Intelligence Lab V2 — automatic, research-only market monitoring."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title='Market Intelligence Lab V2', page_icon='📊', layout='wide')
st.markdown('''<style>div[data-testid="stMetric"]{padding:12px;border:1px solid #ddd;border-radius:12px} .block-container{padding-top:1.5rem}</style>''', unsafe_allow_html=True)
st.title('📊 Market Intelligence Lab')
st.caption('النسخة الثانية · تحليل تلقائي عند فتح الصفحة · بدون تداول حقيقي')
st.warning('هذه أداة بحث وتجربة فقط. لا توجد توقعات مضمونة أو أوامر شراء أو ربط بوسيط.')

SYMBOLS = {'الذهب — GLD': 'GLD', 'S&P 500 — SPY': 'SPY'}
FEATURES = ['ret_1','ret_5','ret_20','vol_10','vol_20','ma_gap_10','ma_gap_50']
HORIZON = 5
THRESHOLD = 0.60
INITIAL_JOD = 300.0
FEE_BPS = 10

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbol):
    raw = yf.download(symbol, period='8y', interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError('مصدر الأسعار لم يرجع بيانات')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    s = raw['Close'].dropna().astype(float)
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep='last')].sort_index()

def features_for(prices):
    r = prices.pct_change()
    f = pd.DataFrame(index=prices.index)
    f['ret_1'] = r
    f['ret_5'] = prices.pct_change(5)
    f['ret_20'] = prices.pct_change(20)
    f['vol_10'] = r.rolling(10).std()
    f['vol_20'] = r.rolling(20).std()
    f['ma_gap_10'] = prices / prices.rolling(10).mean() - 1
    f['ma_gap_50'] = prices / prices.rolling(50).mean() - 1
    future = prices.shift(-HORIZON) / prices - 1
    f['future_return'] = future
    f['target'] = (future > 0).astype(float).where(future.notna())
    return f.replace([np.inf,-np.inf],np.nan)

def run_test(prices):
    f = features_for(prices)
    clean = f.dropna(subset=FEATURES)
    labeled = clean.dropna(subset=['target','future_return'])
    min_train = 504
    if len(labeled) < min_train + 100:
        raise ValueError('لا توجد بيانات تاريخية كافية')
    rows = []
    # Non-overlapping outcomes; exclude the most recent HORIZON labels from training.
    for pos in range(min_train + HORIZON, len(labeled), HORIZON):
        train = labeled.iloc[:pos-HORIZON]
        test = labeled.iloc[pos]
        if train.target.nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
        model.fit(train[FEATURES], train.target.astype(int))
        p = float(model.predict_proba(test[FEATURES].to_frame().T)[0,1])
        rows.append({'date':labeled.index[pos], 'p_up':p, 'actual_up':int(test.target),
                     'future_return':float(test.future_return)})
    back = pd.DataFrame(rows).set_index('date')
    # Use index positions (trading sessions) to exclude labels whose outcomes are not known.
    latest = clean.iloc[-1]
    train_current = labeled.loc[labeled.index <= clean.index[-(HORIZON+1)]]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
    model.fit(train_current[FEATURES], train_current.target.astype(int))
    probability = float(model.predict_proba(latest[FEATURES].to_frame().T)[0,1])
    return back, probability

def paper_test(back, enabled):
    b = back.copy()
    b['invested'] = ((b.p_up >= THRESHOLD) & enabled).astype(int)
    # Approximate next-open execution is not available from close-only series: do not present as executable P&L.
    # Apply round-trip friction and conservative position sizing, still an illustrative close-to-close simulation.
    b['period_return'] = 0.25*b.invested*(b.future_return-2*FEE_BPS/10000)
    balance, peak, halted = INITIAL_JOD, INITIAL_JOD, False
    equities, flags = [], []
    for ret in b.period_return:
        if not halted:
            balance *= 1+ret
            peak = max(peak,balance)
            if balance/peak-1 <= -0.10:
                halted = True
        equities.append(balance)
        flags.append(halted)
    b['equity'] = equities
    b['halted'] = flags
    return b

with st.sidebar:
    st.header('إعدادات المتابعة')
    st.write('السوقان يتم تحليلهما تلقائياً؛ لا حاجة لإدخال بيانات.')
    st.caption('التوقع: 5 جلسات تداول · حد إشارة تجريبية: 60% · أقصى تخصيص افتراضي: 25% · لا رافعة مالية')
    if st.button('🔄 تحديث الأسعار الآن'):
        load_prices.clear()
        st.rerun()

st.subheader('ملخص السوقين')
st.caption('الأسعار بالدولار لكل سهم من الصندوق المتداول. البيانات قد تكون متأخرة، وليست أسعاراً لحظية.')
results = {}
for title, ticker in SYMBOLS.items():
    try:
        with st.spinner(f'جاري تحليل {title}...'):
            prices = load_prices(ticker)
            back, probability = run_test(prices)
        hit = accuracy_score(back.actual_up, (back.p_up >= 0.5).astype(int))
        baseline = max(back.actual_up.mean(), 1-back.actual_up.mean())
        brier = brier_score_loss(back.actual_up, back.p_up)
        # Conservative gate: don't claim validated edge merely from a high latest probability.
        proven = bool(hit > baseline and brier < back.actual_up.mean()*(1-back.actual_up.mean()))
        stale = (pd.Timestamp.now(tz='UTC').tz_localize(None).normalize()-prices.index[-1].normalize()).days > 5
        signal = 'مراقبة فقط' if stale else ('دخول افتراضي مشروط' if proven and probability >= THRESHOLD else 'انتظار / مراقبة')
        results[ticker] = dict(title=title, prices=prices, back=back, p=probability, hit=hit,
                               baseline=baseline, brier=brier, proven=proven, stale=stale, signal=signal)
    except Exception as exc:
        results[ticker] = dict(error=str(exc), title=title)

for title, ticker in SYMBOLS.items():
    r = results[ticker]
    st.markdown(f'### {title}')
    if 'error' in r:
        st.error(f'تعذر تحديث {ticker}: {r["error"]}. جرّبي تحديث الصفحة لاحقاً.')
        continue
    prices = r['prices']
    a,b,c = st.columns(3)
    a.metric('آخر سعر إغلاق',f'${prices.iloc[-1]:,.2f}')
    b.metric('احتمال الصعود خلال 5 جلسات',f'{r["p"]:.1%}')
    c.metric('حالة المتابعة',r['signal'])
    st.caption(f'تاريخ السعر: {prices.index[-1].date()} · عدد التوقعات التاريخية: {len(r["back"])} · مصدر البيانات: Yahoo Finance عبر yfinance')
    if r['stale']:
        st.error('البيانات قديمة نسبياً؛ لا نعتمد أي إشارة حتى تتحدث الأسعار.')
    elif not r['proven']:
        st.info('الانتظار أفضل في هذا الاختبار: النموذج لم يثبت تفوقاً كافياً على خط الأساس في الدقة وجودة الاحتمالات.')
    elif r['p'] < THRESHOLD:
        st.info('احتمال الصعود أقل من الحد التجريبي المحدد (60%). لا توجد إشارة دخول.')
    else:
        st.warning('إشارة محاكاة مشروطة فقط؛ ليست توصية شراء أو ضمان ربح.')
    with st.expander('التفاصيل: لماذا ظهرت هذه النتيجة؟', expanded=False):
        st.write(f'دقة النموذج التاريخية: **{r["hit"]:.1%}** · خط الأساس: **{r["baseline"]:.1%}** · Brier: **{r["brier"]:.3f}** (الأقل أفضل).')
        st.write('يعتمد النموذج على تغيرات الأسعار السابقة، التقلب، وبعد السعر عن متوسطاته. نسبة الصعود تقدير إحصائي غير معاير وليست نسبة نجاح مؤكدة.')
        st.write('الخطر: حركة الأسعار قد تختلف عن الماضي، وقد تتأخر البيانات أو تتعطل. GLD وSPY لا يمثلان سعر الذهب الفوري أو مؤشر S&P 500 مباشرة.')
    fig = go.Figure(go.Scatter(x=prices.index[-260:], y=prices.iloc[-260:], mode='lines', name=ticker))
    fig.update_layout(height=260, margin=dict(l=8,r=8,t=8,b=8), xaxis_title='', yaxis_title='USD / share')
    st.plotly_chart(fig, use_container_width=True, key=f'price_{ticker}')
    st.divider()

st.subheader('🧪 المحفظة الافتراضية — 300 دينار')
st.caption('كل سوق يُختبر بصورة مستقلة بنفس الرصيد الافتراضي للمقارنة؛ لا يتم جمع الرصيدين أو اعتبارهما محفظة مشتركة.')
for ticker in ['GLD','SPY']:
    r = results[ticker]
    if 'error' in r:
        continue
    # Historical gate is evaluated on full backtest; using it for past trading decisions would leak future information.
    # Hence keep simulated positions disabled in V2 until a truly time-local gate and execution model are implemented.
    paper = paper_test(r['back'], enabled=False)
    with st.expander(f'{r["title"]} — نتائج الاختبار', expanded=False):
        st.metric('الرصيد التجريبي المحايد',f'{paper.equity.iloc[-1]:,.2f} د.أ')
        st.info('لم تُنفّذ صفقات في المحفظة الجديدة: تم تعليق الدخول إلى حين اختبار إشارات متتابعة بلا تسرب معلومات، مع تنفيذ في جلسة لاحقة وتكاليف واقعية. نتائج النسخة الأولى لا تُعد أداءً قابلاً للتنفيذ.')
        report = r['back'].copy()
        report.index.name='date'
        st.dataframe(report.tail(25).style.format({'p_up':'{:.1%}','future_return':'{:.2%}'}),use_container_width=True)
        st.download_button(f'تنزيل نتائج {ticker}',report.to_csv().encode('utf-8'),file_name=f'{ticker}_v2_backtest.csv',mime='text/csv',key=f'download_{ticker}')

st.caption(f'آخر فتح للتطبيق (UTC): {datetime.now(timezone.utc):%Y-%m-%d %H:%M} · تحديث البيانات المخزنة كل ساعة تقريباً. لا توجد مراقبة في الخلفية أو إشعارات تلقائية في هذه النسخة.')
