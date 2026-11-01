"""Market Intelligence Lab V3: Arabic-first informational signals and forward journal."""
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from engine import SYMBOLS, prices_for, analyze, HORIZON

st.set_page_config(page_title='Market Intelligence Lab V3', page_icon='📈', layout='wide')
st.markdown('<style>.block-container{padding-top:1.2rem}div[data-testid="stMetric"]{border:1px solid #ddd;border-radius:12px;padding:12px}</style>', unsafe_allow_html=True)
st.title('📈 Market Intelligence Lab V3')
st.caption('مراقبة تلقائية عند فتح الصفحة · إشارات بحثية · اختبار افتراضي · بدون تداول حقيقي')
st.warning('هذه أداة بحث وتجربة وليست توصيات استثمارية. الاحتمالات تقديرية وغير مضمونة. لا توجد أوامر شراء أو ربط بوسيط.')

@st.cache_data(ttl=3600, show_spinner=False)
def cached_prices(ticker):
    return prices_for(ticker)

with st.sidebar:
    st.header('الإعدادات')
    capital = st.select_slider('الرصيد الافتراضي بالدينار', options=[100,150,200,250,300], value=200)
    st.caption('أفق التوقع: 5 جلسات تداول. لا رافعة مالية. لا يوجد تنفيذ تلقائي.')
    if st.button('🔄 تحديث الأسعار الآن'):
        cached_prices.clear()
        st.rerun()

st.subheader('ملخص السوقين')
st.caption('GLD وSPY صندوقان متداولان بالدولار، وليسا الذهب الفوري أو مؤشر S&P 500 نفسه. البيانات قد تكون متأخرة.')
for ticker, title in SYMBOLS.items():
    st.markdown(f'### {title}')
    try:
        with st.spinner(f'تحليل {ticker}...'):
            prices = cached_prices(ticker)
            r = analyze(prices)
        a,b,c = st.columns(3)
        a.metric('آخر إغلاق بالدولار', f'${r["close"]:,.2f}')
        b.metric('تقدير الصعود خلال 5 جلسات', f'{r["probability"]:.1%}')
        c.metric('الحالة', r['status'])
        st.caption(f'تاريخ السعر: {r["date"]} · بيانات Yahoo Finance عبر yfinance · ليست أسعاراً لحظية')
        if r['stale']:
            st.error('البيانات قديمة: لا نعتمد الإشارة.')
        elif r['status'] == 'انتظار':
            st.info('لا توجد إشارة تستوفي الحد التجريبي (60%). الأفضل الانتظار.')
        else:
            st.info('إشارة للمراقبة والاختبار فقط. لم يتم فتح صفقة افتراضية أو حقيقية.')
        with st.expander('لماذا ظهرت النتيجة؟'):
            st.write(f'دقة التوقعات التاريخية: {r["accuracy"]:.1%} · خط الأساس: {r["baseline"]:.1%} · مقياس Brier: {r["brier"]:.3f} (الأقل أفضل).')
            st.write('النموذج يعتمد على العائد السابق والتقلب وبعد السعر عن المتوسطات. النسبة ليست احتمالية مؤكدة أو معايرة.')
            st.write('مهم: الاختبار التاريخي لا يثبت ربحية الصفقات؛ التنفيذ الفعلي يتأثر بالفروقات السعرية والرسوم وفجوات الافتتاح.')
        fig = go.Figure(go.Scatter(x=prices.index[-260:],y=prices.iloc[-260:],mode='lines',name=ticker))
        fig.update_layout(height=260, margin=dict(l=8,r=8,t=8,b=8), yaxis_title='USD / share')
        st.plotly_chart(fig,use_container_width=True,key=ticker)
        with st.expander('التوقعات التاريخية — للمراجعة'):
            back = r['back'].copy()
            st.dataframe(back.tail(30).style.format({'p_up':'{:.1%}','future_return':'{:.2%}'}),use_container_width=True)
            st.download_button('تنزيل CSV',back.to_csv().encode(),file_name=f'{ticker}_v3_history.csv',key=f'dl_{ticker}')
    except Exception as e:
        st.error(f'تعذر تحليل {ticker}: {e}')
    st.divider()

st.subheader('🧪 سجل الإشارات المستقبلية')
st.caption('السجل يُنشأ عبر مهمة GitHub Actions المجدولة، وليس من فتح الصفحة فقط. سجلاته مؤرخة وغير قابلة لإعادة تفسير التوقع الأصلي بعد النتيجة.')
path = Path(__file__).parent / 'signals.csv'
if path.exists():
    try:
        journal = pd.read_csv(path)
        if journal.empty:
            st.info('السجل فارغ حالياً. أول تشغيل ناجح للمهمة المجدولة سيبدأ تسجيل الإشارات.')
        else:
            st.dataframe(journal.sort_values('recorded_at_utc',ascending=False),use_container_width=True,hide_index=True)
            st.download_button('تنزيل سجل الإشارات',journal.to_csv(index=False).encode('utf-8-sig'),file_name='v3_signals.csv')
    except Exception as e:
        st.error(f'تعذر قراءة السجل: {e}')
else:
    st.info('لم يبدأ تسجيل الإشارات بعد. يلزم تفعيل GitHub Actions حسب التعليمات.')
st.metric('الرصيد الافتراضي المرجعي', f'{capital:.0f} د.أ')
st.caption('لا تُحتسب أرباح أو خسائر افتراضية حتى نضيف نموذج تنفيذ لاحق بأسعار جلسة تالية وتكاليف قابلة للتحقق. لا يوجد تداول بالخلفية أو إشعارات هاتفية.')
