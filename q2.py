import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Q1)

ages = [13, 40, 45, 16, 22, 25, 25, 22, 25, 25, 30, 52, 70, 33, 35, 15, 35, 36, 33, 16, 19, 20, 35, 35, 46, 20, 21]
sns.set(style="white")
sns.histplot(ages, color="m")

plt.xlabel("age")
plt.title("Histogram")
plt.show()

print()

sns.boxplot(data=ages, color='lightcoral')
plt.title("Box plot")
plt.ylabel("Ages")
plt.yticks(np.arange(min(ages), max(ages) + 1, 3))
plt.show()

print()
plt.figure()

bins = ['0-19', '20-29', '30-39', '40-49', '50-59', '60+']

age_counts = [0] * len(bins)

for age in ages:
    if age < 20:
        age_counts[0] += 1
    elif age < 30:
        age_counts[1] += 1
    elif age < 40:
        age_counts[2] += 1
    elif age < 50:
        age_counts[3] += 1
    elif age < 60:
        age_counts[4] += 1
    else:
        age_counts[5] += 1

colors = ['gold', 'lightcoral', 'lightskyblue', 'yellowgreen', 'cyan', 'purple', 'orange']
plt.title("Pie chart")
plt.pie(age_counts, labels=bins, autopct='%1.1f%%', colors=colors, textprops={'fontsize':7})
plt.show()

#Q2)

file_path = '/content/drive/MyDrive/SEM 8/DATA MINING LAB/PS2/iris.csv'
df = pd.read_csv(file_path)

plt.figure()
scatter_plot = sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species', style='species', s=20)
plt.title("Scatter plot")
plt.show()

plt.figure(figsize=(12, 6))
swarm_plot=sns.swarmplot(x='petal_width',y='sepal_width',data=df, color='purple', size=5)
plt.title("Swarm plot of Petal width by sepal width")
plt.show()

print()

plt.figure(figsize=(10, 6))
sns.swarmplot( x='sepal_width', y='species', data=df, color='#8DB600')
plt.title('Swarm Plot of Sepal width by Species')
plt.xticks(np.arange(min(df.sepal_width), max(df.sepal_width) + 1, 0.2))
plt.ylabel('Species')
plt.xlabel('Sepal width (cm)')
plt.show()

print()

plt.figure(figsize=(10,6))
sns.swarmplot(x='sepal_length', y='species', data=df, color='lightpink')
plt.title('Swarm plot of sepal length by species')
plt.xticks(np.arange(min(df.sepal_length), max(df.sepal_length) + 1, 0.2))
plt.ylabel('species')
plt.xlabel('sepal length')
plt.show()

print()

#histogram

plt.figure()
sns.histplot(data=df, x='sepal_width', hue='species', fill=True, common_norm=False, palette='viridis' )
plt.title('Histogram of Sepal width by species')
plt.xlabel('Sepal width (cm)')
plt.ylabel('Count')
plt.show()

print()

#density plot
plt.figure()
sns.kdeplot(data=df, x='sepal_width', hue='species', fill=True, common_norm=False, palette='viridis')
plt.title('Density Plot of Sepal Width by Species')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Density')
plt.show()

#Q3)

file_path = '/content/drive/MyDrive/SEM 8/DATA MINING LAB/PS2/iris.csv'
df = pd.read_csv(file_path)

plt.figure(figsize=(10,10))
sns.boxplot(data=df.drop(columns='species'))
plt.title('Box Plots of All Features (Excluding Class)')
plt.yticks(np.arange(0,9,0.2))
plt.show()
print()

#Q4)

file_path = '/content/drive/MyDrive/SEM 8/DATA MINING LAB/PS2/iris.csv'
df = pd.read_csv(file_path)


plt.figure(figsize=(10,10))
sns.violinplot(data=df, x='species', y='petal_length')
plt.title('Violin plot of petal length by species')
plt.yticks(np.arange(0,9,0.2))
plt.show()
