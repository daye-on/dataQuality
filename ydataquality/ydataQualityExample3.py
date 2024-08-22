from ydata_quality import DataQuality
from ydata_quality.bias_fairness import BiasFairness
import pandas as pd

# Load data
df = pd.read_csv('./datasets/transformed/census_10k.csv')
bf = BiasFairness(df, sensitive_features=['race', 'sex'], label='income')

# 데이터 개선
def improve_quality(df: pd.DataFrame):
    df = df.replace({'relationship': {'Husband': 'Married', 'Wife': 'Married'}})
    # Substitute gender-based 'Husband'/'Wife' for generic 'Married'

    # Duplicates
    df = df.drop(columns=['workclass2']) # Remove the duplicated column (app1)
    df = df.drop_duplicates()            # Remove exact feature value duplicates (app2)
    return df

# 이전의 "프록시 식별" 품질 경고가 사라짐을 확인할 수 있음
# 새로운 연관 결과를 확인하기 위해 임계값을 낮추고,
# 'relationship'과 'sex' 간의 연관 측정값이 0.65 -> 0.48 로 떨어졌음을 확인할 수 있음
clean_df = improve_quality(df)
better_dq = DataQuality(df=clean_df)
result = better_dq.evaluate()

# 특정 모듈에서의 검증 결과도 개선됨을 확인할 수 있음
better_bf = BiasFairness(df=clean_df, sensitive_features=['race', 'sex'], label='income')
_ = better_bf.evaluate()

print(better_bf.proxy_identification(th=0.45))