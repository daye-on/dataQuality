import numpy as np

array = np.array([1, 2, np.nan, 4])
print(array)

print("\n----- nan 식별 ----------")
nan_indices = np.where(np.isnan(array))
print(nan_indices)

print("\n----- nan 제외 평균계산 -----")
mean_val = np.nanmean(array)
print(mean_val)

print("\n----- nan 값 변경 -----")
array_no_nan = np.nan_to_num(array, nan=-1)
print(array_no_nan)