import numpy as np
import pandas as pd

import os
os.environ['OMP_NUM_THREADS'] = '1' # LOKY_MAX_CPU_COUNT 변경 에러

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepchecks.tabular.datasets.classification import iris
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite

"""
간단한 아이리스 데이터 세트를 사용하고 
다중 클래스 분류를 위한 간단한 랜덤 포레스트 모델을 훈련할 것
(학습 모델 - RandomForestClassifier)
"""
# Load Data
iris_df = iris.load_data(data_format='Dataframe', as_train_test=False)
label_col = 'target'
df_train, df_test = train_test_split(iris_df, stratify=iris_df[label_col], random_state=0)

# Train Model
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(df_train.drop(label_col, axis=1), df_train[label_col])

"""
데이터세트 객체 초기화
"""
ds_train = Dataset(df_train, label=label_col, cat_features=[])
ds_test =  Dataset(df_test,  label=label_col, cat_features=[])

"""
Deepcheck suite 실행
"""
suite = full_suite()
suite.run(train_dataset=ds_train, test_dataset=ds_test, model=rf_clf)
