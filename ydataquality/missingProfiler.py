import statsmodels.api as sm
from ydata_quality.missings import MissingsProfiler

df = sm.datasets.get_rdataset('baseball', 'plyr').data
mp = MissingsProfiler(df=df, random_state=42)
results = mp.evaluate()

print(mp.null_count())