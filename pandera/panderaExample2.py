import pandas as pd
import pandera as pa

from pandera import Column, Check

# csv 파일을 pandas dataframe 으로 로딩
df = pd.read_csv('movies.csv')
rating_list = list(df['rating'].unique())

# 잘못된 데이터 삽입
# df.loc[len(df)] = [None, "R", 2000, 122]
# df.loc[len(df)] = ["Title", None, 2000, 122]
# df.loc[len(df)] = ["Title", "R", 3000, 122]
# df.loc[len(df)] = ["Title", "R", 2000, -123]

print(df)
schema_column = pa.DataFrameSchema({
    "title" : Column(str, nullable=False),
    "rating" : Column(str, pa.Check.isin(rating_list), nullable=False),       # Checking for Train-Test Contamination
    "year" : Column(int, pa.Check.in_range(1900, 2024)),    # Checking for Feature Importance Stability
    "runtime" : Column(int, Check(lambda x:x>0)),
}, unique=["title", "year"]
)

try :
    schema_column.validate(df)
    print("\n------ Validate success ------")
except pa.errors.SchemaError as exc:
    print("\n------ Validate error ------")
    print(exc)