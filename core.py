"""수요 예측 + 운영 전략 로직. 화면(app.py)은 이 파일의 함수만 부른다.

1단계: predict_demand 는 '같은 요일·시간대의 과거 평균' 규칙.
2단계: predict_demand 안쪽만 회귀모델(기온·강수 포함)로 바꾸면 화면은 그대로다.
"""
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# 재배치 규칙: 등급별로 예측 수요의 몇 %를 재배치할지 (운영 가정, 화면에서 조정 가능)
REBALANCE_RATIO = {"높음": 0.15, "보통": 0.08, "낮음": 0.0}


def load_hourly(path, weather_path=None) -> pd.DataFrame:
    """시간대별 대여 건수 (+ 있으면 기온 temp, 강수 rain 을 시각 기준으로 결합)."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    if weather_path is not None:
        w = pd.read_csv(weather_path, parse_dates=["datetime"])
        df = df.merge(w[["datetime", "temp", "rain"]], on="datetime", how="left")
    return df.sort_values("datetime").reset_index(drop=True)


def fetch_forecast(lat: float = 37.5665, lon: float = 126.978, days: int = 16) -> pd.DataFrame:
    """Open-Meteo 시간별 예보(서울 기본, 키 불필요) -> datetime, temp, rain. 실패하면 예외를 그대로 던진다."""
    import requests
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
        "latitude": lat, "longitude": lon, "hourly": "temperature_2m,precipitation",
        "timezone": "Asia/Seoul", "forecast_days": days})
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.DataFrame({"datetime": pd.to_datetime(h["time"]), "temp": h["temperature_2m"],
                         "rain": h["precipitation"]}).dropna()


def split_by_date(df: pd.DataFrame, test_days: int = 28):
    """날짜 기준 분할: 마지막 test_days 일을 검증용으로. 시간 순서를 섞지 않는다."""
    cut = df["datetime"].dt.normalize().max() - pd.Timedelta(days=test_days - 1)
    return df[df["datetime"] < cut], df[df["datetime"] >= cut]


HOLIDAYS = pd.to_datetime([
    # 2025 (임시공휴일·대체공휴일 포함)
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-01", "2025-03-03",
    "2025-05-05", "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-05",
    "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-12-25",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-01", "2026-03-02",
    "2026-05-01", "2026-05-05", "2026-05-24", "2026-05-25", "2026-06-03", "2026-06-06",
    "2026-08-15", "2026-08-17", "2026-09-24", "2026-09-25", "2026-09-26", "2026-10-03", "2026-10-05",
    "2026-10-09", "2026-12-25"]).date
FEATURES = ["hour", "dow", "is_holiday", "temp", "rain"]
_models: dict = {}


def predict_rule(train: pd.DataFrame, when):
    """1단계 규칙: 같은 요일·시간대 과거 평균."""
    key = train.assign(dow=train["datetime"].dt.dayofweek, hour=train["datetime"].dt.hour)
    table = key.groupby(["dow", "hour"])["rentals"].mean()
    w = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(when)))
    return np.array([table.get((t.dayofweek, t.hour), table.mean()) for t in w])


def _features(w: pd.DatetimeIndex, temp, rain) -> pd.DataFrame:
    return pd.DataFrame({"hour": w.hour, "dow": w.dayofweek, "is_holiday": [int(t.date() in HOLIDAYS) for t in w],
                         "temp": np.broadcast_to(temp, len(w)), "rain": np.broadcast_to(rain, len(w))})


def _model(train: pd.DataFrame):
    key = (len(train), train["datetime"].max())
    if key not in _models:
        tr = train.dropna(subset=["temp", "rain"])
        _models.clear()
        _models[key] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(
            _features(pd.DatetimeIndex(tr["datetime"]), tr["temp"].values, tr["rain"].values), tr["rentals"])
    return _models[key]


def predict_demand(train: pd.DataFrame, when, temp=None, rain=None):
    """when(시각 또는 시각 목록)의 대여 건수 예측. 화면이 부르는 유일한 예측 함수.

    2단계: 기온·강수가 주어지면 회귀모델, 없으면 1단계 규칙으로 대체한다.
    """
    single = isinstance(when, (pd.Timestamp, str))
    w = pd.DatetimeIndex(np.atleast_1d(pd.to_datetime(when)))
    if temp is None or rain is None:
        out = predict_rule(train, w)
    else:
        out = np.maximum(_model(train).predict(_features(w, temp, rain)), 0)
    return float(out[0]) if single else out


def evaluate(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res, ss_tot = ((y_true - y_pred) ** 2).sum(), ((y_true - y_true.mean()) ** 2).sum()
    return {"MAE": float(np.abs(y_true - y_pred).mean()), "R2": float(1 - ss_res / ss_tot)}


def demand_level(pred: float, q25: float, q75: float) -> str:
    return "높음" if pred >= q75 else "낮음" if pred <= q25 else "보통"


def rebalance_plan(pred: float, q25: float, q75: float, ratio: dict | None = None) -> dict:
    """예측 수요 -> 등급 -> 재배치 수량 규칙. (수량 = 예측 × 등급별 비율, 올림)"""
    ratio = ratio or REBALANCE_RATIO
    lv = demand_level(pred, q25, q75)
    qty = math.ceil(pred * ratio[lv])
    action = {"높음": "인기 대여소 선재배치 + 정비 보류",
              "보통": "평시 재배치 유지",
              "낮음": "재배치 최소화, 정비·점검 집중"}[lv]
    return {"level": lv, "bikes": qty, "action": action}
