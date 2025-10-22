import streamlit as st
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import requests # NewsAPI 호출용
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os # 비밀(Secrets) 접근용

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="퀀트 리스크 & 스크리너", layout="wide")
st.title("📈 퀀트 리스크 대시보드 & S&P 500 스크리너")

# --- 2. 메인 탭(Tabs) 생성 ---
tab1, tab2 = st.tabs(["실시간 시장 리스크", "📈 S&P 500 퀀트 스크리너"])


# ==============================================================================
# --- 탭 1: 실시간 시장 리스크 ---
# ==============================================================================
with tab1:
    st.header("실시간 시장 리스크 (Macro)")
    
    # --- (캐시!) 모델과 스케일러는 앱 실행 시 한 번만 로드합니다. ---
    @st.cache_resource
    def load_models():
        try:
            model = joblib.load('crash_model_v1.pkl')
            scaler = joblib.load('scaler_v1.pkl')
            
            # FinBERT 모델 로드 (yiyanghkust/finbert-tone)
            finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            
            return model, scaler, finbert_tokenizer, finbert_model
        except FileNotFoundError:
            st.error("오류: 'crash_model_v1.pkl' 또는 'scaler_v1.pkl' 파일을 찾을 수 없습니다.")
            st.error("Codespace에 이 파일들을 업로드했는지 확인하세요.")
            return None, None, None, None
        except Exception as e:
            st.error(f"모델 로드 중 오류 발생: {e}")
            return None, None, None, None

    model, scaler, finbert_tokenizer, finbert_model = load_models()

    # --- FinBERT 감성 분석 함수 ---
    @st.cache_data
    def analyze_sentiment(headlines):
        if not headlines:
            return 0.0
        device = torch.device("cpu")
        finbert_model.to(device)
        inputs = finbert_tokenizer(headlines, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = inputs.to(device) 
        with torch.no_grad():
            outputs = finbert_model(**inputs)
        probabilities = F.softmax(outputs.logits, dim=1)
        negative_scores = probabilities[:, 0].cpu().numpy()
        return np.mean(negative_scores)

    # --- 실시간 데이터 수집 함수 (API 호출) ---
    @st.cache_data(ttl=600)
    def get_market_data():
        """VIX와 10Y-3M 금리 스프레드의 최신 데이터를 가져옵니다. (더욱 안전한 버전)"""
        try:
            vix_series = yf.download('^VIX', period='10d')['Close']
            vix = float(vix_series.dropna().iloc[-1])
            
            tnx_series = yf.download('^TNX', period='10d')['Close']
            irx_series = yf.download('^IRX', period='10d')['Close']
            tnx = float(tnx_series.dropna().iloc[-1])
            irx = float(irx_series.dropna().iloc[-1])
            
            spread = tnx - irx
            return vix, spread
        except Exception as e:
            st.error(f"[탭 1] yfinance 데이터 수집 오류: {e}")
            return None, None

    @st.cache_data(ttl=600)
    def get_news_headlines():
        """NewsAPI로 최신 금융 헤드라인을 가져옵니다."""
        try:
            API_KEY = os.environ.get('NEWS_API_KEY') # (Secret 이름 확인!)
            if not API_KEY:
                st.error("NewsAPI 키가 설정되지 않았습니다. (Codespace Secrets: NEWS_API_KEY)")
                return []
            url = (f'https://newsapi.org/v2/top-headlines?'
                   'sources=bloomberg,financial-post,the-wall-street-journal,reuters'
                   '&language=en'
                   f'&apiKey={API_KEY}')
            response = requests.get(url)
            data = response.json()
            if data['status'] == 'ok':
                headlines = [article['title'] for article in data['articles']]
                return headlines
            else:
                st.error(f"NewsAPI 오류: {data.get('message', '알 수 없는 오류')}")
                return []
        except Exception as e:
            st.error(f"NewsAPI 요청 오류: {e}")
            return []

    # --- 메인 대시보드 로직 (탭 1) ---
    if model and scaler and finbert_model:
        vix_latest, spread_latest = get_market_data()
        headlines_latest = get_news_headlines()
        
        if vix_latest is not None and spread_latest is not None:
            input_data = np.array([[vix_latest, spread_latest]])
            input_scaled = scaler.transform(input_data)
            crash_probability = model.predict_proba(input_scaled)[0][1]
            crash_prob_percent = crash_probability * 100
            
            negative_sentiment = analyze_sentiment(headlines_latest)
            negative_sentiment_percent = negative_sentiment * 100

            st.subheader("실시간 리스크 지표")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="모델 기반 폭락 확률 (VIX, 금리)",
                    value=f"{crash_prob_percent:.2f} %",
                    delta=f"{crash_prob_percent - 30:.2f} % (vs. 기준 30%)",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    label="실시간 뉴스 부정 감성 (FinBERT)",
                    value=f"{negative_sentiment_percent:.2f} %",
                    delta=f"{negative_sentiment_percent - 50:.2f} % (vs. 기준 50%)",
                    delta_color="inverse"
                )

            st.subheader("최신 입력 데이터")
            col1, col2 = st.columns(2)
            col1.info(f"**최신 VIX:** {vix_latest:.2f}")
            col2.info(f"**최신 10Y-3M 스프레드:** {spread_latest:.2f}")

            st.subheader("최신 분석 헤드라인 (NewsAPI)")
            if headlines_latest:
                st.dataframe(pd.DataFrame(headlines_latest, columns=["Headline"]), use_container_width=True)
            else:
                st.warning("헤드라인을 가져오지 못했습니다.")
    else:
        st.error("모델 로드에 실패하여 [탭 1]을 실행할 수 없습니다.")


# ==============================================================================
# --- 탭 2: S&P 500 퀀트 스크리너 ---
# (*** 이 부분이 "Top 50" 기능으로 업그레이드되었습니다 ***)
# ==============================================================================
with tab2:
    st.header("S&P 500 퀀트 스크리너 (Micro)")
    st.write('S&P 500 기업 중 "기본기(Fundamentals)가 좋은" 주식을 필터링하고, "현재 뉴스 감성"을 분석합니다.')
    st.write('**신규 기능:** 섹터별 필터링 후, 지정한 기준의 **Top 50** 기업만 표시합니다.')

    # --- (FinBERT 모델 로드 - 탭 1에서 로드된 것을 재사용) ---
    if not finbert_model or not finbert_tokenizer:
        st.error("FinBERT 모델이 로드되지 않았습니다. [실시간 시장 리스크] 탭을 먼저 방문해주세요.")
        st.stop()

    # --- 스크리너 함수 (탭 2 전용) ---
    
    @st.cache_data(ttl=86400) # 24시간 캐시
    def get_sp500_tickers():
        """위키피디아에서 S&P 500 티커 목록을 스크래핑합니다. (User-Agent 수정됨)"""
        url = 'https://en.wikipedia.org/wiki/List_of_S&P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status() # 오류가 있으면 여기서 멈춤
        table = pd.read_html(response.text)
        tickers = table[0]['Symbol'].tolist()
        return [ticker.replace('.', '-') for ticker in tickers]

    @st.cache_data(ttl=86400) # 24시간 캐시
    def get_fundamental_data(tickers):
        """티커 리스트를 받아 yfinance.info로 기본기 데이터를 가져옵니다."""
        data = []
        progress_bar = st.progress(0, text="S&P 500 재무 데이터 로드 중... (1/500)")
        
        total_tickers = len(tickers)
        for i, ticker in enumerate(tickers):
            try:
                tk = yf.Ticker(ticker)
                info = tk.info
                data.append({
                    'Ticker': ticker,
                    'Company Name': info.get('shortName'),
                    'Sector': info.get('sector'),
                    'Market Cap (Billions)': info.get('marketCap', 0) / 1e9,
                    'P/E (Forward)': info.get('forwardPE'),
                    'P/B (Price to Book)': info.get('priceToBook'),
                    'ROE (Return on Equity)': info.get('returnOnEquity'),
                    'Revenue Growth (YoY)': info.get('revenueGrowth'),
                    'Debt/Equity': info.get('debtToEquity')
                })
            except Exception as e:
                pass 
            progress_bar.progress(
                (i + 1) / total_tickers, 
                text=f"S&P 500 재무 데이터 로드 중... ({ticker}: {i+1}/{total_tickers})"
            )
        progress_bar.empty() 
        # (중요!) Sector가 없는 데이터는 제거
        return pd.DataFrame(data).dropna(subset=['P/E (Forward)', 'P/B (Price to Book)', 'Company Name', 'Sector'])

    # --- (신규!) Phase 2: 개별 주식 뉴스 감성 분석 함수 ---
    @st.cache_data(ttl=3600) # 1시간(3600초) 캐시
    def get_news_for_stock(company_name):
        """NewsAPI로 특정 회사의 헤드라인을 가져옵니다."""
        try:
            API_KEY = os.environ.get('NEWS_API_KEY') # (Secret 이름 확인!)
            if not API_KEY:
                st.error("NewsAPI 키가 설정되지 않았습니다.")
                return []
            url = (f'https://newsapi.org/v2/everything?'
                   f'q="{company_name}"' 
                   '&language=en&pageSize=20'
                   f'&apiKey={API_KEY}')
            response = requests.get(url)
            data = response.json()
            if data['status'] == 'ok':
                return [article['title'] for article in data['articles']]
            else:
                return []
        except Exception as e:
            st.warning(f"{company_name} 뉴스 검색 오류: {e}")
            return []

    @st.cache_data(ttl=3600) # 1시간(3600초) 캐시
    def get_sentiment_for_stocks(df_filtered):
        """필터링된 DataFrame을 받아 개별 뉴스 감성을 분석합니다. (iterrows 수정)"""
        sentiment_scores = []
        
        progress_bar = st.progress(0, text="뉴스 감성 분석 중... (1/?)")
        total_stocks = len(df_filtered)
        
        # --- (수정됨!) .itertuples() 대신 .iterrows() 사용 ---
        for i, (index, row) in enumerate(df_filtered.iterrows()):
            
            # .iterrows()는 row가 Series 객체이므로, 딕셔너리처럼 []로 접근합니다.
            # 이름 변환(mangling)이 없어 안전합니다.
            company_name = row['Company Name'] 
            
            progress_bar.progress(
                (i + 1) / total_stocks, 
                text=f"뉴스 감성 분석 중... ({company_name}: {i+1}/{total_stocks})"
            )
            
            headlines = get_news_for_stock(company_name)
            if headlines:
                score = analyze_sentiment(headlines)
                sentiment_scores.append(score)
            else:
                sentiment_scores.append(np.nan) 
                
        progress_bar.empty()
        # (결과를 할당할 때 .copy()를 사용해야 경고가 뜨지 않습니다)
        df_result = df_filtered.copy()
        df_result['Negative Score'] = sentiment_scores
        return df_result.dropna(subset=['Negative Score'])

    # --- (신규!) 섹터별 기본 파라미터 딕셔너리 ---
    DEFAULT_SECTOR_PARAMS = {
        'Technology':       {'pe_max': 40.0, 'pb_max': 10.0, 'roe_min': 0.15, 'revg_min': 0.10},
        'Healthcare':       {'pe_max': 25.0, 'pb_max': 7.0,  'roe_min': 0.10, 'revg_min': 0.05},
        'Financials':       {'pe_max': 20.0, 'pb_max': 2.0,  'roe_min': 0.08, 'revg_min': 0.03},
        'Consumer Cyclical':{'pe_max': 25.0, 'pb_max': 5.0,  'roe_min': 0.12, 'revg_min': 0.05},
        'Consumer Defensive':{'pe_max': 20.0, 'pb_max': 3.5,  'roe_min': 0.10, 'revg_min': 0.03},
        'Industrials':      {'pe_max': 22.0, 'pb_max': 4.0,  'roe_min': 0.10, 'revg_min': 0.05},
        'Real Estate':      {'pe_max': 25.0, 'pb_max': 3.0,  'roe_min': 0.05, 'revg_min': 0.03},
        'Utilities':        {'pe_max': 20.0, 'pb_max': 2.5,  'roe_min': 0.08, 'revg_min': 0.02},
        'Communication Services': {'pe_max': 28.0, 'pb_max': 5.0, 'roe_min': 0.10, 'revg_min': 0.05},
        'Energy':           {'pe_max': 15.0, 'pb_max': 2.5,  'roe_min': 0.05, 'revg_min': 0.03},
        'Basic Materials':  {'pe_max': 20.0, 'pb_max': 3.0,  'roe_min': 0.08, 'revg_min': 0.03},
        'Other':            {'pe_max': 25.0, 'pb_max': 5.0,  'roe_min': 0.10, 'revg_min': 0.05}
    }
    
    # --- 스크리너 UI (탭 2) ---
    
    if 'df_fundamentals' not in st.session_state:
        st.info("S&P 500 전체 기업의 재무 데이터를 로드해야 합니다.")
        if st.button("S&P 500 데이터 로드하기 (최초 1회 5~10분 소요)", key='load_sp500'):
            with st.spinner("Wikipedia에서 S&P 500 목록 스크래핑 중..."):
                tickers = get_sp500_tickers()
            df_fundamentals = get_fundamental_data(tickers)
            st.session_state['df_fundamentals'] = df_fundamentals
            
            # (신규!) 섹터 파라미터를 세션 상태에 초기화
            st.session_state['sector_params'] = DEFAULT_SECTOR_PARAMS.copy()
            st.rerun() 
    
    if 'df_fundamentals' in st.session_state:
        df_full = st.session_state['df_fundamentals']
        
        # (신규!) 섹터 파라미터 설정 UI (사이드바 대신 메인 탭에)
        st.sidebar.info("섹터별 상세 필터는 '퀀트 스크리너' 탭 메인 화면에서 설정하세요.")
        
        with st.expander("📈 (Phase 1) 섹터별 펀더멘탈 필터 설정"):
            # (데이터에 있는 섹터 목록 + 'Other' 추가)
            available_sectors = list(df_full['Sector'].unique())
            if 'Other' not in available_sectors:
                available_sectors.append('Other')
                
            # (st.tabs를 사용해 섹터별로 UI 분리)
            sector_tabs = st.tabs(available_sectors)
            
            for i, sector_name in enumerate(available_sectors):
                with sector_tabs[i]:
                    # (세션 상태에 저장된 파라미터 값을 가져옴)
                    if sector_name not in st.session_state.sector_params:
                        st.session_state.sector_params[sector_name] = DEFAULT_SECTOR_PARAMS['Other'] # 기본값
                        
                    params = st.session_state.sector_params[sector_name]
                    
                    c1, c2 = st.columns(2)
                    c3, c4 = st.columns(2)
                    
                    # (각 슬라이더가 st.session_state를 직접 수정하도록 key 설정)
                    params['pe_max'] = c1.slider('최대 P/E', 5.0, 100.0, params['pe_max'], 1.0, key=f'pe_{sector_name}')
                    params['pb_max'] = c2.slider('최대 P/B', 0.1, 20.0, params['pb_max'], 0.1, key=f'pb_{sector_name}')
                    params['roe_min'] = c3.slider('최소 ROE', 0.0, 1.0, params['roe_min'], 0.01, format="%.2f", key=f'roe_{sector_name}')
                    params['revg_min'] = c4.slider('최소 매출 성장률', 0.0, 1.0, params['revg_min'], 0.01, format="%.2f", key=f'revg_{sector_name}')

        # --- (신규!) 필터링 로직 (섹터별 적용) ---
        filtered_dfs = []
        for sector_name, params in st.session_state.sector_params.items():
            sector_df = df_full[df_full['Sector'] == sector_name]
            
            if not sector_df.empty:
                sector_filtered = sector_df[
                    (sector_df['P/E (Forward)'] <= params['pe_max']) &
                    (sector_df['P/B (Price to Book)'] <= params['pb_max']) &
                    (sector_df['ROE (Return on Equity)'] >= params['roe_min']) &
                    (sector_df['Revenue Growth (YoY)'] >= params['revg_min'])
                ]
                filtered_dfs.append(sector_filtered)

        df_filtered = pd.concat(filtered_dfs).sort_index()
        
        # --- (신규!) Top 50 필터링 및 결과 표시 ---
        st.subheader(f"필터링된 '우량주' 후보: ({len(df_filtered)} / {len(df_full)}개)")
        
        if not df_filtered.empty:
            # --- (신규!) Top 50 정렬 기준 선택 ---
            sort_by = st.selectbox(
                "Top 50 정렬 기준 선택",
                ('P/E (Forward) (낮은 순)', 
                 'Market Cap (Billions) (높은 순)', 
                 'ROE (Return on Equity) (높은 순)'),
                key='sort_by_top50'
            )

            if sort_by == 'P/E (Forward) (낮은 순)':
                df_sorted = df_filtered.sort_values(by='P/E (Forward)', ascending=True)
            elif sort_by == 'Market Cap (Billions) (높은 순)':
                df_sorted = df_filtered.sort_values(by='Market Cap (Billions)', ascending=False)
            else: # ROE
                df_sorted = df_filtered.sort_values(by='ROE (Return on Equity)', ascending=False)
            
            df_top50 = df_sorted.head(50)
            
            st.dataframe(df_top50, use_container_width=True)
            
            st.divider()
            st.header("📰 (Phase 2) 뉴스 감성 분석")
            
            # --- (신규!) Phase 2 실행 버튼 (df_top50을 사용) ---
            if st.button(f"필터링된 Top {len(df_top50)}개 주식 뉴스 감성 분석하기", key='run_sentiment'):
                if len(df_top50) == 0:
                    st.error("분석할 주식이 없습니다. Phase 1 필터를 조정하세요.")
                else:
                    # (get_sentiment_for_stocks 함수가 df_top50을 받도록 수정)
                    df_with_sentiment = get_sentiment_for_stocks(df_top50.copy())
                    st.session_state['df_sentiment'] = df_with_sentiment
        else:
            st.warning("Phase 1 필터 조건에 맞는 주식이 없습니다.")


        # --- Phase 3: 최종 결과 표시 ---
        if 'df_sentiment' in st.session_state:
            st.header("🎯 (Phase 3) 저평가 기회 목록 (우량주 + 나쁜 뉴스)")
            st.write("기본기는 좋지만(Phase 1), 현재 단기적으로 부정적인 뉴스(Phase 2)가 많은 주식입니다.")
            
            df_final = st.session_state['df_sentiment']
            st.dataframe(
                df_final.sort_values(by='Negative Score', ascending=False), 
                use_container_width=True,
                column_config={
                    "Negative Score": st.column_config.ProgressColumn(
                        "부정 뉴스 점수",
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0,
                    )
                }
            )

        if st.sidebar.button("캐시 지우고 모든 데이터 다시 로드", key='clear_cache_sp500'):
            st.cache_data.clear() # 모든 캐시 지우기
            # 세션 상태도 모두 삭제
            for key in list(st.session_state.keys()):
                if key in ['df_fundamentals', 'df_sentiment', 'sector_params']:
                    del st.session_state[key]
            st.rerun()