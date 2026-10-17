"""따릉이 대여이력(cp949, 월별 CSV) -> 시간대별 대여 건수 data/hourly_rentals.csv (datetime, rentals).

사용: python build_hourly.py <대여이력 CSV 폴더>
"""
import subprocess
import sys
from pathlib import Path

import pandas as pd

src = Path(sys.argv[1] if len(sys.argv) > 1 else "../jovyan")
out = Path(__file__).parent / "data" / "hourly_rentals.csv"
parts = []
for f in sorted(src.glob("서울특별시 공공자전거 대여이력 정보_*.csv")):
    cnt = pd.Series(dtype="int64")
    # 잘린 파일(따옴표 안에서 EOF)도 읽도록 cut 으로 둘째 열(대여일시)만 줄 단위로 뽑는다
    p = subprocess.Popen(["cut", "-d,", "-f2", str(f)], stdout=subprocess.PIPE)
    for ch in pd.read_csv(p.stdout, header=None, skiprows=1, encoding="latin-1", dtype=str, quoting=3, chunksize=1_000_000):
        t = pd.to_datetime(ch.iloc[:, 0].str.strip('"'), errors="coerce").dropna().dt.floor("h")
        cnt = cnt.add(t.value_counts(), fill_value=0)
    p.wait()
    cnt = cnt.sort_index().iloc[:-1]  # 파일 끝의 불완전할 수 있는 마지막 시간대는 버린다
    parts.append(cnt)
    print(f.name, int(cnt.sum()), flush=True)
s = pd.concat(parts).groupby(level=0).sum().sort_index().astype(int)
# 잘린 달(2503, 2506 등)의 빈 구간을 0건으로 채우지 않는다: 없는 시간은 그대로 둔다
s.rename_axis("datetime").rename("rentals").reset_index().to_csv(out, index=False)
print("saved", out, len(s))
