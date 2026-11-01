"""V3 research signal engine. No broker connections or real orders."""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

SYMBOLS = {'GLD': 'الذهب — GLD', 'SPY': 'S&P 500 — SPY'}
FEATURES = ['ret_1', 'ret_5', 'ret_20', 'vol_10', 'vol_20', 'gap_10', 'gap_50']
HORIZON = 5
MIN_TRAIN = 504
THRESHOLD = 0.60

def prices_for(symbol):
    raw = yf.download(symbol, period='8y', interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise ValueError('لم تصل بيانات الأسعار')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    close = raw['Close'].dropna().astype(float)
    close.index = pd.to_datetime(close.index).tz_localize(None) if close.index.tz is not None else pd.to_datetime(close.index)
    return close[~close.index.duplicated(keep='last')].sort_index()

def features(prices):
    r = prices.pct_change()
    f = pd.DataFrame(index=prices.index)
    f['ret_1'] = r
    f['ret_5'] = prices.pct_change(5)
    f['ret_20'] = prices.pct_change(20)
    f['vol_10'] = r.rolling(10).std()
    f['vol_20'] = r.rolling(20).std()
    f['gap_10'] = prices / prices.rolling(10).mean() - 1
    f['gap_50'] = prices / prices.rolling(50).mean() - 1
    future = prices.shift(-HORIZON) / prices - 1
    f['target'] = (future > 0).astype(int).where(future.notna())
    f['future_return'] = future
    return f.replace([np.inf, -np.inf], np.nan)

def _model(train):
    if len(train) < MIN_TRAIN or train['target'].nunique() < 2:
        raise ValueError('البيانات التدريبية غير كافية')
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600))
    model.fit(train[FEATURES], train['target'].astype(int))
    return model

def analyze(prices):
    f = features(prices)
    usable = f.dropna(subset=FEATURES)
    labeled = usable.dropna(subset=['target', 'future_return'])
    if len(labeled) < MIN_TRAIN + 110:
        raise ValueError('لا توجد بيانات تاريخية كافية')
    # Each test prediction only trains on labels already resolved by the test date.
    rows = []
    for pos in range(MIN_TRAIN + HORIZON, len(usable) - HORIZON, HORIZON):
        train = usable.iloc[:pos-HORIZON].dropna(subset=['target'])
        test = usable.iloc[pos]
        if len(train) < MIN_TRAIN:
            continue
        p = float(_model(train).predict_proba(test[FEATURES].to_frame().T)[0, 1])
        rows.append({'date': usable.index[pos], 'p_up': p, 'actual_up': int(test['target']), 'future_return': float(test['future_return'])})
    back = pd.DataFrame(rows).set_index('date')
    # Historical performance is exploratory; do not use full-history test scores to approve past trades.
    current = usable.iloc[-1]
    training = usable.iloc[:-HORIZON].dropna(subset=['target'])
    probability = float(_model(training).predict_proba(current[FEATURES].to_frame().T)[0, 1])
    accuracy = accuracy_score(back.actual_up, (back.p_up >= .5).astype(int))
    base = max(back.actual_up.mean(), 1-back.actual_up.mean())
    brier = brier_score_loss(back.actual_up, back.p_up)
    stale = (datetime.now(timezone.utc).date() - prices.index[-1].date()).days > 5
    # Cautious informational signal only; validation is not a guarantee.
    status = 'بيانات قديمة' if stale else ('فرصة للمراقبة فقط' if probability >= THRESHOLD else 'انتظار')
    return {'probability': probability, 'back': back, 'accuracy': accuracy, 'baseline': base,
            'brier': brier, 'status': status, 'stale': stale, 'date': prices.index[-1].date(),
            'close': float(prices.iloc[-1]), 'volatility': float(usable.iloc[-1]['vol_20'])}
