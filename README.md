# 날씨 기반 공유 자전거 수요 예측

    pip install streamlit scikit-learn plotly pandas
    python make_sample.py      # 샘플 데이터 생성 (data/sample.csv)
    streamlit run app.py

실제 데이터는 `data/rentals.csv` (컬럼: date, rentals, temp, rain)로 넣으면 자동으로 샘플 대신 사용됩니다.
- rentals: 일별 대여 건수 (공공자전거 대여이력 집계)
- temp: 일 평균기온(℃), rain: 일 강수량(mm) (기상청 ASOS 서울)
