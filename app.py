import lightgbm as lgb
import numpy as np
import pandas as pd
from serpapi import GoogleSearch
import streamlit as st

# 1. 페이지 설정 (반드시 최상단에 위치해야 함)
st.set_page_config(page_title="유휴 공간 AI 시뮬레이터", page_icon="🏙️", layout="wide")


# ---- Helper functions -----------------------------------------------------
def load_dummy_data():
  rng = np.random.default_rng(42)
  n = 200
  data = pd.DataFrame({
      "year_built": rng.integers(1990, 2023, size=n),
      "floors": rng.integers(1, 6, size=n),
      "foot_traffic": rng.integers(100, 5000, size=n),
      "latitude": rng.uniform(35.5, 36.5, size=n),
      "longitude": rng.uniform(128.5, 129.5, size=n),
      "industry": rng.choice(
          ["로컬 카페", "피트니스/헬스", "독립서점/공방", "의원/클리닉", "공유오피스"], size=n
      ),
  })
  age = 2026 - data["year_built"]
  surv_score = 0.6 * (
      data["foot_traffic"] / data["foot_traffic"].max()
  ) + 0.4 * (1 - age / age.max())
  data["survive"] = (surv_score > 0.5).astype(int)
  data["profit"] = surv_score * 100000 - age * 2000
  return data


@st.cache_data
def train_models(df: pd.DataFrame):
  features = ["year_built", "floors", "foot_traffic", "latitude", "longitude"]
  X = df[features]
  y_cls = df["survive"]
  y_reg = df["profit"]

  cls = lgb.LGBMClassifier(random_state=0, verbose=-1)
  cls.fit(X, y_cls)

  reg = lgb.LGBMRegressor(random_state=0, verbose=-1)
  reg.fit(X, y_reg)
  return cls, reg, features


def predict_survival(cls, reg, features, input_dict):
  df_input = pd.DataFrame([input_dict])
  prob = cls.predict_proba(df_input[features])[0, 1]
  profit = reg.predict(df_input[features])[0]
  return prob, profit


def get_top_industries(df, top_n=3):
  rates = df.groupby("industry")["survive"].mean().sort_values(ascending=False)
  return list(rates.head(top_n).index)


def fetch_local_news(region: str, api_key: str, top_n: int = 3):
  search = GoogleSearch({
      "engine": "google",
      "tbm": "nws",
      "q": f"{region} 상권",
      "hl": "ko",
      "gl": "kr",
      "api_key": api_key,
      "num": top_n,
  })
  try:
    result = search.get_dict()
    articles = result.get("news_results", [])[:top_n]
    if not articles:
      return [{"title": "관련 최신 뉴스 검색 결과가 없습니다.", "link": "#"}]
    return [{"title": a.get("title"), "link": a.get("link")} for a in articles]
  except Exception as e:
    return [{"title": f"검색 연동 오류: {e}", "link": "#"}]


# ---- Load data & train models ---------------------------------------------
df_dummy = load_dummy_data()
cls_model, reg_model, feature_cols = train_models(df_dummy)
industry_options = ["로컬 카페", "피트니스/헬스", "독립서점/공방", "의원/클리닉", "공유오피스"]

# ---- Streamlit UI ---------------------------------------------------------
st.title("🏙️ 지방 중소도시 유휴 공간 최적 업종 매칭 및 생존율 예측 AI 시뮬레이터")
st.markdown(
    "공공 건축물·상권 데이터 기반 **LightGBM 머신러닝 예측**과 **SerpApi 실시간 상권"
    " 트렌드**를 결합한 시뮬레이터입니다."
)
st.markdown("---")

with st.sidebar:
  st.header("🗺️ 지역·매물 선택")
  region = st.selectbox(
      "시·군 선택", ["경북 영주시", "경북 포항시", "경북 안동시", "전남 광양시"], index=0
  )

  st.subheader("📍 매물 위치")
  lat = st.number_input(
      "위도 (Latitude)", min_value=-90.0, max_value=90.0, value=36.8058, step=0.0001
  )
  lon = st.number_input(
      "경도 (Longitude)",
      min_value=-180.0,
      max_value=180.0,
      value=128.6245,
      step=0.0001,
  )
  st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

  st.subheader("🏢 건물 스펙")
  year_built = st.slider("준공연도", 1990, 2026, 2010)
  floors = st.selectbox("층수", list(range(1, 11)), index=0)
  foot_traffic = st.number_input(
      "유동인구 (일 평균)", min_value=0, max_value=100000, value=2000, step=100
  )

  st.header("📊 분석 옵션")
  industry = st.multiselect(
      "입점 희망 업종 (복수 선택 가능)", industry_options, default=["로컬 카페"]
  )
  run_button = st.button("🚀 AI 분석 실행", type="primary")

if run_button:
  input_data = {
      "year_built": year_built,
      "floors": floors,
      "foot_traffic": foot_traffic,
      "latitude": lat,
      "longitude": lon,
  }
  surv_prob, profit_est = predict_survival(
      cls_model, reg_model, feature_cols, input_data
  )
  top_industries = get_top_industries(df_dummy, top_n=3)

  col1, col2 = st.columns(2)
  with col1:
    st.subheader("🔮 AI 예측 분석 결과")
    st.metric(label="1년 뒤 예상 생존 확률", value=f"{surv_prob*100:.1f}%")
    st.metric(label="예상 월 기대 수익 지수", value=f"{max(profit_est, 0):,.0f} 원")
    st.markdown("**🏆 해당 입지 추천 업종 TOP 3**")
    for i, ind in enumerate(top_industries, 1):
      st.write(f"{i}. **{ind}**")

  with col2:
    st.subheader(f"📰 {region} 최신 상권 뉴스 (SerpApi)")
    api_key = (
        "b3de0e37bcd4678ac9fc53f442297b4df7a76d0512e2a0c7e65fa89853004f08"
    )
    news = fetch_local_news(region, api_key)
    for n in news:
      st.markdown(f"- [{n['title']}]({n['link']})")
else:
  st.info(
      "👈 좌측 사이드바에서 매물 정보를 입력하고 **‘🚀 AI 분석 실행’** 버튼을 눌러 결과를"
      " 확인하세요."
  )
