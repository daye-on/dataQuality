from ydata_quality import DataQuality
import pandas as pd

# Load data
df = pd.read_csv('./datasets/transformed/census_10k.csv')

# Create the main class that holds all quality modules
dq = DataQuality(df)

# run the tests
results = dq.evaluate()

# CSV 파일 보면 중복된 칼럼 확인 가능 (workclass, workclass2)
print(dq.get_warnings(test='Duplicate Columns')[0].data)