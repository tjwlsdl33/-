"""기상청 ASOS 시간자료(cp949, 여러 파일 가능) -> data/weather.csv (datetime, temp, rain). 강수 공란=무강수=0."""
import sys
from pathlib import Path

import pandas as pd

d = pd.concat([pd.read_csv(f, encoding="cp949") for f in sys.argv[1:]])  # 여러 기간 파일 합치기
w = pd.DataFrame({"datetime": pd.to_datetime(d["일시"]), "temp": d["기온(°C)"],
                  "rain": d["강수량(mm)"].fillna(0)})
w = w.drop_duplicates("datetime").sort_values("datetime")
w["temp"] = w["temp"].interpolate(limit=3)  # 짧은 결측만 보간
w.to_csv(Path(__file__).parent / "data" / "weather.csv", index=False)
print(len(w), w["datetime"].min(), w["datetime"].max(), w.isna().sum().to_dict())
