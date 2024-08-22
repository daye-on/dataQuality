from ydata_quality.bias_fairness import BiasFairness
import pandas as pd

# Load data
df = pd.read_csv('./datasets/transformed/census_10k.csv')

# 일부 모듈은 특정 파라미터가 필수인 모듈도 있음
# 민감한 정보를 가진 칼럼에 대해 독립적으로 실행함
bf = BiasFairness(df, sensitive_features=['race', 'sex'], label='income')
_ = bf.evaluate()


# 칼럼 'relationship'과 'sex'는 높은 상관관계를 가짐
# 예를 들어, HUSBAND = MALE, WIFE=FEMALE
# 그리고 relationship은 민감한 속성이므로 정보를 유출할 수 있음
print(bf.get_warnings(test='Proxy Identification')) # relationship_sex    0.650656
# print(df[['relationship', 'sex']].value_counts().sort_index())