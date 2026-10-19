"""따릉이 시간대별 수요 예측 + 재배치 운영 제안 (Streamlit). 로직은 core.py."""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import core

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="따릉이 수요 예측·재배치", page_icon="🚲", layout="wide")


@st.cache_data
def load():
    w = DATA / "weather.csv"
    return core.load_hourly(DATA / "hourly_rentals.csv", w if w.exists() else None)


@st.cache_data(ttl=3600)
def _forecast():
    return core.fetch_forecast().set_index("datetime")


def load_forecast():
    """예보를 1시간 캐시. 실패(None)는 캐시하지 않아 다음 실행에서 다시 시도한다."""
    try:
        return _forecast()
    except Exception:
        return None


df = load()
fc = load_forecast()
has_weather = "temp" in df.columns
test_days = st.sidebar.slider("검증 기간 (최근 N일)", 7, 56, 28)
train, test = core.split_by_date(df, test_days)
q25, q75 = train["rentals"].quantile([0.25, 0.75])
ratio = {k: st.sidebar.slider(f"재배치 비율 · 수요 {k}", 0, 40, int(v * 100), format="%d%%") / 100
         for k, v in core.REBALANCE_RATIO.items()}
st.sidebar.caption("재배치 수량 = 예측 대여 × 등급별 비율 (올림)")

st.title("🚲 따릉이 시간대별 수요 예측 · 재배치 제안")
st.caption(f"학습 {train.datetime.min():%Y-%m-%d} ~ {train.datetime.max():%Y-%m-%d} · "
           f"검증 {test.datetime.min():%Y-%m-%d} ~ {test.datetime.max():%Y-%m-%d} (날짜 기준 분할)")

if not has_weather:
    st.warning("기온·강수 데이터(data/weather.csv: datetime, temp, rain)가 아직 없어 "
               "1단계(요일·시간대 평균 규칙)로 동작합니다.")

tab1, tab2, tab3 = st.tabs(["① 수요 예측·재배치", "② 날씨별 수요 변화", "③ 모델 검증"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    d = c1.date_input("날짜", (fc.index.min() if fc is not None else test.datetime.max() + pd.Timedelta(days=1)).date(),
                      key="pred_date")
    h = c2.slider("시간대", 0, 23, 8)
    use_fc = st.checkbox("기상 예보 자동 사용 (Open-Meteo, 서울)", value=fc is not None and has_weather,
                         disabled=fc is None or not has_weather)
    if fc is None and has_weather:
        st.warning("예보를 불러오지 못해 기온·강수를 직접 입력해야 합니다.")
    when = pd.Timestamp(d) + pd.Timedelta(hours=h)
    auto = use_fc and when in fc.index
    if use_fc and not auto:
        st.warning(f"예보 범위({fc.index.min():%m-%d} ~ {fc.index.max():%m-%d})를 벗어나 직접 입력 값을 사용합니다.")
    if auto:
        temp, rain = float(fc.at[when, "temp"]), float(fc.at[when, "rain"])
        c3.metric("예보 기온 (℃)", f"{temp:.1f}")
        c4.metric("예보 강수량 (mm/h)", f"{rain:.1f}")
    else:
        temp = c3.slider("예상 기온 (℃)", -15.0, 40.0, 20.0, 0.5, disabled=not has_weather)
        rain = c4.slider("예상 강수량 (mm/h)", 0.0, 50.0, 0.0, 0.5, disabled=not has_weather)


    def wx_at(t):
        """시각 t의 날씨 입력: 예보가 있으면 예보, 아니면 화면의 값."""
        if not has_weather:
            return {}
        if use_fc and t in fc.index:
            return dict(temp=float(fc.at[t, "temp"]), rain=float(fc.at[t, "rain"]))
        return dict(temp=temp, rain=rain)


    wx = wx_at(when)
    pred = core.predict_demand(train, when, **wx)
    plan = core.rebalance_plan(pred, q25, q75, ratio)
    m1, m2, m3 = st.columns(3)
    m1.metric("예측 대여 건수", f"{pred:,.0f} 건")
    m2.metric("수요 등급", plan["level"])
    m3.metric("재배치 수량", f"{plan['bikes']:,} 대")
    st.info(plan["action"])

    day = pd.date_range(pd.Timestamp(d), periods=24, freq="h")
    rows = [(t.hour, core.predict_demand(train, t, **wx_at(t))) for t in day]
    plans = [core.rebalance_plan(p, q25, q75, ratio) for _, p in rows]
    day_df = pd.DataFrame({"시간": [r[0] for r in rows], "예측 대여": [r[1] for r in rows],
                           "등급": [p["level"] for p in plans], "재배치(대)": [p["bikes"] for p in plans]})
    st.plotly_chart(px.bar(day_df, x="시간", y="예측 대여", color="등급",
                           color_discrete_map={"높음": "#d62728", "보통": "#bbbbbb", "낮음": "#2ca02c"},
                           title=f"{d} 시간대별 예측과 등급" + (" (예보 반영)" if auto else f" (기온 {temp}℃, 강수 {rain}mm)" if has_weather else "")), width="stretch")
    st.dataframe(day_df.round(0), width="stretch", hide_index=True)
    st.caption(f"하루 재배치 합계: {day_df['재배치(대)'].sum():,} 대")

with tab2:
    if not has_weather:
        st.info("weather.csv 를 추가하면 기온·강수별 수요 변화 그래프가 표시됩니다.")
    else:
        w = df.dropna(subset=["temp", "rain"])
        w = w.assign(기온구간=pd.cut(w["temp"], range(-20, 45, 5)).astype(str),
                     강수=pd.cut(w["rain"], [-0.1, 0, 1, 5, 20, 1000], labels=["0", "0~1", "1~5", "5~20", "20+"]))
        a, b = st.columns(2)
        a.plotly_chart(px.bar(w.groupby("기온구간", observed=True)["rentals"].mean().reset_index(),
                              x="기온구간", y="rentals", title="🌡️기온별(℃) 평균 시간당 대여"), width="stretch")
        b.plotly_chart(px.bar(w.groupby("강수", observed=True)["rentals"].mean().reset_index(),
                              x="강수", y="rentals", title="☂️강수량(mm)별 평균 시간당 대여"), width="stretch")

with tab3:
    t = test.dropna(subset=["temp", "rain"]) if has_weather else test
    pr = core.predict_rule(train, t["datetime"])
    rows = {"1단계 규칙 (요일·시간대 평균)": core.evaluate(t["rentals"], pr)}
    cmp = pd.DataFrame({"datetime": t["datetime"], "실제": t["rentals"], "규칙": pr})
    if has_weather:
        pm = core.predict_demand(train, t["datetime"], t["temp"].values, t["rain"].values)
        rows["2단계 모델 (+기온·강수)"] = core.evaluate(t["rentals"], pm)
        cmp["모델"] = pm
    st.dataframe(pd.DataFrame(rows).T.round({"MAE": 0, "R2": 3}), width="stretch")
    st.plotly_chart(px.line(cmp.melt("datetime"), x="datetime", y="value", color="variable",
                            title="검증 구간 실제 vs 예측"), width="stretch")
