import pandas as pd
import numpy as np

# 누락된 값 제거
df1 = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
print(df1)
print("---RESULT 1------------")
cleaned_df = df1.dropna()
print(cleaned_df)
print()

# 중복된 값 제거
df2 = pd.DataFrame({'A': [1, 2, 1], 'B': [4, 5, 4]})
print(df2)
print("---RESULT 2------------")
duplicates = df2[df2.duplicated()]
df_unique = df2.drop_duplicates()
print(df_unique)
print()
# print("\n중복된 값")
# print(duplicates)

# 누락된 값 채우기
df3 = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
print(df3)
print("---RESULT 3------------")

# df.fillna(method='ffill')은 없어질 메소드라서 다음과 같은 방법을 추천함.
# df_filled = df.fillna(method='ffill')
df_filled = df3.ffill()
print(df_filled)