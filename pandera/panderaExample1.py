# Validating a DataFrame Against a Schema

import pandas as pd
import pandera as pa
from pandera import Column, Check

# csv 파일을 pandas dataframe 으로 로딩
df = pd.read_csv('movies.csv')

# 예상 되는 dataframe 스키마 사양 정의
schema_success = pa.DataFrameSchema({
    "title" : Column(str),
    "rating" : Column(str),
    "year" : Column(int),
    "runtime" : Column(int),
})

# 예상 되는 스키마로 dataframe 검증
schema_success.validate(df)

# 잘못된 스키마
schema_fail = pa.DataFrameSchema({
    "title" : Column(int), # 에러
    "rating" : Column(str),
    "year" : Column(int),
    "runtime" : Column(int),
})

# 잘못된 스키마로 dataframe 검증
schema_fail.validate(df)