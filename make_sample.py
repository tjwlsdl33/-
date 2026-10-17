"""샘플 데이터 생성: date, rentals, temp, rain (실제 데이터 구조와 동일)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
doy = dates.dayofyear.values
temp = 12.5 - 14 * np.cos(2 * np.pi * (doy - 15) / 365) + rng.normal(0, 3, len(dates))
wet = rng.random(len(dates)) < np.where((dates.month >= 6) & (dates.month <= 8), 0.4, 0.2)
rain = np.where(wet, rng.gamma(1.2, 12, len(dates)), 0.0).round(1)

comfort = np.exp(-((temp - 21) ** 2) / (2 * 9**2))  # 21도 부근이 최적, 너무 덥거나 추우면 감소
weekend = (dates.dayofweek >= 5).astype(float)
base = 30000 + 70000 * comfort
base *= 1 - 0.12 * weekend
base *= np.exp(-rain / 25)
rentals = (base * rng.normal(1, 0.07, len(dates))).astype(int)

pd.DataFrame({"date": dates.date, "rentals": rentals, "temp": temp.round(1), "rain": rain}).to_csv(
    "data/sample.csv", index=False
)
print("saved data/sample.csv", len(dates))
