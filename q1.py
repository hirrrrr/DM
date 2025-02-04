import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

file_path = '/content/drive/MyDrive/SEM 8/DATA MINING LAB/automobile_spare.csv'
df = pd.read_csv(file_path, header=None)

max_sales = df.max(axis=1)[0]
print("maximum number of spares sold:", max_sales)

num_days=df.eq(max_sales).sum().sum()
print("number of days on which max sales achieved:", num_days)

no_sales=df.eq(0).sum().sum()
print("number of no sales days:", no_sales)

first_half=df.values.flatten()[:30]
second_half=df.values.flatten()[30:60]

if(sum(first_half) > sum(second_half)):
  print("First half sum is higher with : ", sum(first_half))

else:
  print("Second half sum is higher with : ", sum(second_half))

print("Maximum in first half: ", max(first_half))
print("Maximum in second half: ", max(second_half))

ages = [13, 40, 45, 16, 22, 25, 25, 22, 25, 25, 30, 52, 70, 33, 35, 15, 35, 36, 33, 16, 19, 20, 35, 35, 46, 20, 21]

mean=np.mean(ages)
print("mean:", mean)

median=np.median(ages)
print("median:", median)

mode=stats.mode(ages)
print("mode:", mode[0])

mid_range=(min(ages)+max(ages))/2
print("midrange:", mid_range)

q=np.quantile(ages, [0, 0.25, 0.5, 0.75, 1])

print("first quartile: Q1:", q[1])
print("third quartile: Q3:", q[3])

plt.boxplot(ages)

plt.show()

#Q3)
file_path = '/content/drive/MyDrive/SEM 8/DATA MINING LAB/PS1/student.csv'
df = pd.read_csv(file_path)

size=df.shape
print("size of data set:", size)

missing_values=df.isnull().sum().sum()
print("number of missing_values:", missing_values)

states=set(df["statname"])
print("number of states:", len(states))

schools = [col for col in df.columns if col in [f'sch_{i}' for i in range(1, 8)]]
print("number of school categories:", len(schools))
print(schools)

print()
single_teacher=[col for col in df.columns if 'sing_tch' in col.lower()]
print("percentage of schools have single teacher in secondary level:", (len(single_teacher)/len(schools))*100)

print("Highest literacy rate state:", df[df["literacy_rate"]==max(df["literacy_rate"])]["statname"].iloc[0])
print("Lowest literacy rate state:", df[df["literacy_rate"]==min(df["literacy_rate"])]["statname"].iloc[0])
