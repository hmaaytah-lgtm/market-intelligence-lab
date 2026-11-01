"""Market Intelligence Lab — educational research and paper-only dashboard."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss

st.set_page_config(page_title='Market Intelligence Lab', page_icon='📊', layout='wide')
st.title('📊 Market Intelligence Lab')
st.caption('Gold (GLD) + S&P 500 (SPY) · Research only · No brokerage connection · No real trades')
st.warning('التوقعات احتمالية وليست أرباحاً مضمونة. جميع الصفقات هنا افتراضية، والنتائج التاريخية لا تضمن المستقبل.')

with st.sidebar:
    st.header('إعدادات التجربة')
    ticker = st.selectbox('السوق', ['GLD', 'SPY'])
    years = st.slider('سنوات البيانات التاريخية', 3, 12, 7)
    horizon = st.selectbox('أفق التوقع (أيام تداول)', [1, 5, 10], index=1)
    threshold = st.slider('أقل احتمال لفتح مركز افتراضي', 0.50, 0.80, 0.60, 0.01)
    fee_bps = st.slider('تكلفة تداول افتراضية لكل جانب (نقاط أساس)', 0, 100, 10, 5)
    initial_jod = st.number_input('الرصيد الافتراضي (دينار)', min_value=100, value=300, step=50)
    st.caption('تخصيص أقصى 25% لكل مركز، دون رافعة مالية. المحاكاة Long-only؛ لا بيع على المكشوف.')

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(symbol, period):
    raw = yf.download(symbol, period=f'{period}y', interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError('No price data returned. Please retry later.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    if 'Close' not in raw:
        raise ValueError('Close prices unavailable.')
    prices = raw['Close'].dropna().astype(float)
    prices.index = pd.to_datetime(prices.index).tz_localize(None) if prices.index.tz is not None else pd.to_datetime(prices.index)
    return prices

def make_features(prices, h):
    ret = prices.pct_change()
    features = pd.DataFrame(index=prices.index)
    features['ret_1'] = ret
    features['ret_5'] = prices.pct_change(5)
    features['ret_20'] = prices.pct_change(20)
    features['vol_10'] = ret.rolling(10).std()
    features['vol_20'] = ret.rolling(20).std()
    features['ma_gap_10'] = prices / prices.rolling(10).mean() - 1
    features['ma_gap_50'] = prices / prices.rolling(50).mean() - 1
    future_return = prices.shift(-h) / prices - 1
    features['target'] = np.where(future_return.notna(), (future_return > 0).astype(float), np.nan)
    features['future_return'] = future_return
    features['close'] = prices
    return features.replace([np.inf, -np.inf], np.nan)

def walk_forward(df, h, min_train=504):
    """Non-overlapping decision dates; expanding training ends before test outcome window."""
    cols = ['ret_1','ret_5','ret_20','vol_10','vol_20','ma_gap_10','ma_gap_50']
    usable = df.dropna(subset=cols)
    labeled = usable.dropna(subset=['target','future_return'])
    if len(labeled) < min_train + 60:
        raise ValueError('Insufficient history for a meaningful walk-forward test.')
    rows = []
    for pos in range(min_train + h, len(labeled), h):
        # Training labels must have matured before the prediction date.
        train = labeled.iloc[:pos-h]
        test = labeled.iloc[pos]
        if train['target'].nunique() < 2:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
        model.fit(train[cols], train['target'].astype(int))
        probability = float(model.predict_proba(test[cols].to_frame().T)[0, 1])
        rows.append({'date':labeled.index[pos], 'p_up':probability,
                     'actual_up':int(test['target']), 'future_return':float(test['future_return'])})
    results = pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()
    # Latest forecast is made using only already-matured labels.
    current = usable.iloc[-1]
    mature = labeled.loc[labeled.index <= current.name - pd.Timedelta(days=h*2+4)]
    if len(mature) < min_train or mature['target'].nunique() < 2:
        return results, None, current.name
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
    model.fit(mature[cols], mature['target'].astype(int))
    p_latest = float(model.predict_proba(current[cols].to_frame().T)[0,1])
    return results, p_latest, current.name

def simulate(results, threshold, cost_bps, initial):
    """One non-overlapping h-day position per signal; otherwise remain in cash."""
    r = results.copy()
    r['invested'] = (r['p_up'] >= threshold).astype(int)
    r['gross_portfolio_return'] = 0.25 * r['invested'] * r['future_return']
    r['net_portfolio_return'] = r['gross_portfolio_return'] - r['invested'] * 0.25 * 2 * cost_bps / 10000
    r['equity'] = initial * (1+r['net_portfolio_return']).cumprod()
    r['peak'] = np.maximum(initial, r['equity'].cummax())
    r['drawdown'] = r['equity']/r['peak']-1
    # Stop opening simulated positions once the 10% drawdown threshold is breached.
    stopped = False
    balance = initial
    peak = initial
    balances, halted = [], []
    for _, row in r.iterrows():
        if stopped:
            balances.append(balance)
            halted.append(True)
            continue
        balance *= (1+row['net_portfolio_return'])
        peak = max(peak, balance)
        if balance/peak - 1 <= -0.10:
            stopped = True
        balances.append(balance)
        halted.append(stopped)
    r['equity'] = balances
    r['halted'] = halted
    r['drawdown'] = r['equity']/np.maximum.accumulate(np.r_[initial, r['equity'].to_numpy()])[1:]-1
    return r

try:
    with st.spinner('جاري تحميل البيانات وتشغيل الاختبار الزمني...'):
        prices = load_prices(ticker, years)
        dataset = make_features(prices, horizon)
        test, current_prob, forecast_date = walk_forward(dataset, horizon)
    if test.empty:
        st.error('لم تتوفر نتائج اختبار كافية. جرّبي فترة أطول.')
        st.stop()
    paper = simulate(test, threshold, fee_bps, initial_jod)
    hit = accuracy_score(test['actual_up'], (test['p_up']>=0.5).astype(int))
    baseline = max(test['actual_up'].mean(), 1-test['actual_up'].mean())
    brier = brier_score_loss(test['actual_up'], test['p_up'])
    active = int(paper['invested'].sum())
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('احتمال صعود آخر قراءة', f'{current_prob:.1%}' if current_prob is not None else 'غير متاح')
    c2.metric('دقة الاتجاه (اختبار تاريخي)', f'{hit:.1%}')
    c3.metric('دقة خط أساس الأغلبية', f'{baseline:.1%}')
    c4.metric('الرصيد الافتراضي النهائي', f'{paper["equity"].iloc[-1]:,.2f} د.أ')
    st.caption(f'آخر بيانات: {forecast_date.date()} | عدد التوقعات المختبرة: {len(test)} | عدد إشارات الدخول: {active} | Brier score (أقل أفضل): {brier:.3f}')
    if hit <= baseline:
        st.info('دقة النموذج لا تتفوق على خط الأساس البسيط في هذا الاختبار. لا تعتبر الإشارة ميزة تنبؤية مثبتة.')
    if paper['halted'].any():
        st.error('تم بلوغ حد التراجع الافتراضي 10%؛ أوقفت المحاكاة فتح مراكز جديدة.')
    st.subheader('سعر الأصل')
    fig = go.Figure(go.Scatter(x=prices.index, y=prices, mode='lines', name=ticker))
    fig.update_layout(height=320, xaxis_title='', yaxis_title='USD per share', margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('محفظة افتراضية — بدون أموال حقيقية')
    fig2 = go.Figure(go.Scatter(x=paper.index, y=paper['equity'], mode='lines', name='رصيد افتراضي'))
    fig2.add_hline(y=initial_jod, line_dash='dot')
    fig2.update_layout(height=280, yaxis_title='JOD (simulated)', margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.write(f'أكبر تراجع افتراضي: **{paper["drawdown"].min():.2%}** · مجموع العائد الافتراضي: **{paper["equity"].iloc[-1]/initial_jod-1:.2%}**')
    st.subheader('سجل التوقعات التاريخية')
    view = paper[['p_up','actual_up','future_return','invested','equity','halted']].tail(35).copy()
    view.index = view.index.strftime('%Y-%m-%d')
    st.dataframe(view.style.format({'p_up':'{:.1%}','future_return':'{:.2%}','equity':'{:,.2f}'}), use_container_width=True)
    st.download_button('تنزيل جميع نتائج الاختبار CSV', paper.to_csv().encode('utf-8'), file_name=f'{ticker}_paper_results.csv', mime='text/csv')
    st.info('الاختبار التاريخي يستعمل فترات قرار غير متداخلة، لكنه نموذج مبسط: التنفيذ عند سعر الإغلاق نفسه افتراضي وقد يكون متفائلاً؛ لا يشمل الضرائب أو فروق العملة أو جميع تكاليف الوسيط. الاحتمالات غير معايرة بعد. لا يوجد تداول مباشر أو إرسال أوامر.')
except Exception as exc:
    st.error(f'تعذر تشغيل التحليل: {exc}')
    st.info('تأكدي من اتصال الإنترنت وتوفر بيانات Yahoo Finance. قد يتعطل مصدر البيانات مؤقتاً.')
