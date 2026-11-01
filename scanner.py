"""Scheduled, append-only forward prediction journal (not a trade executor)."""
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from engine import SYMBOLS, prices_for, analyze

PATH = Path(__file__).parent / 'signals.csv'
COLUMNS = ['recorded_at_utc','symbol','price_date','close_usd','p_up_5d','status','data_source']

def main():
    if PATH.exists():
        data = pd.read_csv(PATH, dtype={'price_date':str,'symbol':str})
    else:
        data = pd.DataFrame(columns=COLUMNS)
    new = []
    for ticker in SYMBOLS:
        try:
            r = analyze(prices_for(ticker))
            key_date = str(r['date'])
            if r['stale'] or ((data.symbol == ticker) & (data.price_date == key_date)).any():
                continue
            new.append({'recorded_at_utc':datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        'symbol':ticker,'price_date':key_date,'close_usd':round(r['close'],4),
                        'p_up_5d':round(r['probability'],6),'status':r['status'],'data_source':'Yahoo Finance via yfinance'})
        except Exception as e:
            print(f'{ticker}: {e}')
    if new:
        pd.concat([data,pd.DataFrame(new)],ignore_index=True).to_csv(PATH,index=False)
        print(f'Added {len(new)} predictions')
    else:
        print('No new predictions')

if __name__ == '__main__':
    main()
