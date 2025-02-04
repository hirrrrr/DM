 pip install ucimlrepo

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns

# fetch dataset
credit_approval = fetch_ucirepo(id=27)

# data (as pandas dataframes)
X = credit_approval.data.features
y = credit_approval.data.targets

# metadata
#print(credit_approval.metadata)

# variable information
variable=credit_approval.variables
print(credit_approval.variables)

print("\n\nFEATURES: ")
print(X)
print("\n\nTarget variables")
print(y)

#1. Identify the types of the attribute in the above dataset.
#data types

print("\n\nATTRIBUTE TYPES")
attribute_types = X.dtypes
print("Attr.\tDatatype")
print(attribute_types)

print("\n\nAttr. name\tDatatype")
 
for i in range(len(variable)):
  curr_row=variable.iloc[i].to_dict()
  print("{0}\t{1}".format(curr_row['name'], curr_row['type']))
  print()

#2)Analyze the spread and distribution of all the numerical attributes.

print("\nAnalyzing the spread and distribution of numerical attributes\n")
i=0
plt.figure(figsize=(10,10))
numerical_attributes=X.select_dtypes(include=['float64', 'int64'])

for col in X.columns:
  if X[col].dtype=='float64' or X[col].dtype=='int64':
    #numerical_attributes.append(col)
    i+=1
    plt.subplot(3,3,i)
    sns.histplot(X[col], kde=True, bins=10, color='#EAB8E4')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#3)Create scatter plots of all possible combinations of numerical attributes in the given dataset. Interpret the scatter plots.

#PAIR PLOT EXPLANATION:

# Scatter Plots: For each pair of numerical variables, a scatter plot is generated,
# allowing you to observe potential correlations and relationships.

# Diagonal Plots: The diagonal of the grid can display histograms or kernel density estimates (KDE)
# of each variable, providing insights into their distributions.
# You can customize this to show nothing or a different type of plot.

print("\n\n")

sns.pairplot(numerical_attributes, diag_kind=None)
plt.suptitle('Scatter Plots of Numerical Attributes', y=1.02)
plt.show()
print("hello")

#5) Generate boxplot to identify the outliers on any numeric attribute.
i=0
plt.figure(figsize=(10,10))
for attr in numerical_attributes:
  i+=1
  plt.subplot(3,3,i)
  plt.title(f'Boxplot of {attr}')
  sns.boxplot(X[attr], color='lightcoral')

plt.tight_layout()
plt.show()

#9. Use the appropriate method to remove the nosiy data (Try to use equal width or
#equal frequency binning)

#Equal width binning (refer note for explanation)
numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
num_bins=5

# Loop through the columns of the DataFrame
for col in numerical_columns:
# Remove previously created binned and smoothed columns if they exist
    if col + '_wbinned' in X.columns:
        X.drop([col + '_wbinned', col + '_wsmoothed'], axis=1, inplace=True)

        # Perform equal width binning
    binned_data = pd.cut(X[col], bins=num_bins, labels=False)
    X[col + '_wbinned'] = binned_data

    # Calculate the mean of each bin
    bin_means = X.groupby(binned_data)[col].mean()

    # Replace the original binned values with the mean of the bin
    X[col + '_wsmoothed'] = binned_data.map(bin_means)

    # Print the original, binned, and smoothed data for this column
    print(X[[col, col + '_wbinned', col + '_wsmoothed']])

#Equal frequence binning:

#We can use the pd.qcut() function, which divides the data into bins such that
#each bin contains the same number of data points.

numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
num_bins=5

# Loop through the columns of the DataFrame
for col in numerical_columns:

    # Remove previously created binned and smoothed columns if they exist
    if col + '_fbinned' in X.columns:
        X.drop([col + '_fbinned', col + '_fsmoothed'], axis=1, inplace=True)

        # Perform equal width binning
    binned_data = pd.qcut(X[col], q=num_bins, labels=False, duplicates='drop')
    X[col + '_fbinned'] = binned_data

        # Calculate the mean of each bin
    bin_means = X.groupby(binned_data)[col].mean()

        # Replace the original binned values with the mean of the bin
    X[col + '_fsmoothed'] = binned_data.map(bin_means)

        # Print the original, binned, and smoothed data for this column
    print(X[[col, col + '_fbinned', col + '_fsmoothed']])

#10. Encode the categorical attributes using Label Encoder or One Hot Encoder, if needed.


#LABEL ENCODING

from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder

label_encoder = LabelEncoder()
categorical_columns=['A13', 'A12', 'A10', 'A9', 'A7', 'A6', 'A5', 'A4', 'A1']

# Loop through the categorical columns
for col in categorical_columns:
    #print(X[col].dtype)
    if col+'_label_encoded' in X.columns:
        X.drop(col+'_label_encoded', axis=1, inplace=True)

    X[col+'_label_encoded'] = label_encoder.fit_transform(X[col])
    print("After encoding")
    print(X[[col, col+'_label_encoded']])

print(categorical_columns)

# ONE HOT CODING
categorical_columns=['A13', 'A12', 'A10', 'A9', 'A7', 'A6', 'A5', 'A4', 'A1']


for col in categorical_columns:
  encoded_columns = pd.get_dummies(X[col], prefix=col)
  print(encoded_columns)
  X = pd.concat([X, encoded_columns], axis=1)

print(X.columns)

#11)Normalize the attribute values using various normalization techniques (such as
# min-max normalization, z-score normalization, and normalization by decimal
# scaling).

#numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns

numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
numerical_data=X[numerical_columns]

#  Min-Max Normalization
min_max_normalized = numerical_data[numerical_columns].apply(
    lambda col: (col - col.min()) / (col.max() - col.min())
)

#  Z-Score Normalization
z_score_normalized = numerical_data[numerical_columns].apply(
    lambda col: (col - col.mean()) / col.std()
)

#  Decimal Scaling Normalization
decimal_scaling_normalized = numerical_data[numerical_columns].apply(
    lambda col: col / (10 ** np.ceil(np.log10(col.abs().max())))
)

data_min_max = numerical_data.copy()
data_z_score = numerical_data.copy()
data_decimal_scaling = numerical_data.copy()

data_min_max[numerical_columns] = min_max_normalized
data_z_score[numerical_columns] = z_score_normalized
data_decimal_scaling[numerical_columns] = decimal_scaling_normalized

# Print normalized results
print("Original Data (First 5 rows):\n", numerical_data[numerical_columns].head())
print("\nMin-Max Normalized (First 5 rows):\n", min_max_normalized.head())
print("\nZ-Score Normalized (First 5 rows):\n", z_score_normalized.head())
print("\nDecimal Scaling Normalized (First 5 rows):\n", decimal_scaling_normalized.head())

# 12 Apply the different types of feature selection methods on the given dataset to
# identify the prominent features

numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
numerical_data=X[numerical_columns]
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

X_encoded = X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtypes == 'object' else col)

from sklearn.feature_selection import chi2, SelectKBest

chi_scores = chi2(X_encoded, y)
chi2_features = pd.Series(chi_scores[0], index=X.columns)
chi2_features.sort_values(ascending=False, inplace=True)
print("\nChi-Square Scores:\n", chi2_features)

# Step 5: Select the top 5 features based on Chi-Square scores
selected_features_chi2 = SelectKBest(chi2, k=5).fit_transform(X_encoded, y)
print("Top 5 Features Selected (Chi-Square):", chi2_features.head(5).index.tolist())

#13) Dimensionality reduction - pca
numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
numerical_data=X[numerical_columns]

# Standardize the numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numerical_data)

pca = PCA(n_components=None)
X_pca = pca.fit_transform(X_scaled)

explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', color='red')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by Principal Components')
plt.grid()
plt.show()

# Choose number of components explaining >95% variance
n_components = np.argmax(explained_variance >= 0.95) + 1
print(f"Number of components explaining 95% variance: {n_components}")

# Reduce data to optimal components
pca_optimal = PCA(n_components=n_components)
X_reduced = pca_optimal.fit_transform(X_scaled)
print("Reduced Data Shape:", X_reduced.shape)

#14 Apply correlation techniques to identify those features that are highly contributing in
# the approval of credit cards to a customer.


numerical_columns=['A15', 'A14', 'A11', 'A8', 'A3', 'A2']
numerical_data=X[numerical_columns]
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#15 Generate visualization of the reduced dataset and elaborate your interpretations.
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='lightcoral', alpha=0.7)  # Scatter plot for the first two components
plt.title('PCA Reduced Data (First Two Principal Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
