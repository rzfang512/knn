#Q0.
#1. The difference between regress and  classification is that regression predicts a numeric outcome, whereas classification predicts a categorical outcome.
#2. For classification, confusion table is a cross-tabulation of predicted and actual values. It helps us understand the performance of the model.
#3. SSE is used to aggregate the squared errors into a single metric of fit.
#4. Overfitting is when the model is too complex to reliably explain the phenomenon. On the other hand, underfiting occurs when the model is too simple to reliably explain the phenomenon.
#5. Splitting the data into training and testing sets, and choosing by evaluating accuracy or SSE on the test set, improved model performance by prevents overfitting problems to quirks in the training set.
##With classification, we can report a class label as a prediction or a probability distribution over class labels. Please explain the strengths and weaknesses of each approach.
#6. Strength for reporting a class label as prediction is that it provides simple, straightforward decisions. However, the weaknes is there is no measure of confidence.
##Strengths for reporting class label as probability distribution are flexible decision rules and richer evaluation. The weaknesses are there is more complex output and still requires a final rule.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean the dataset
df = pd.read_csv('USA_cars_datasets.csv')
# Keep only required columns
df = df[['price', 'year', 'mileage']]

## 1.

# Check for missing values
print("Missing values in each column:")
print(df.isna().sum())

# Display the first few rows of the data
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Show the dimensions of the dataset
print("\nDataset dimensions (rows, columns):")
print(df.shape)

# The is not any NA to handle. The head of the data are printed price, year, and mileage. The dimension of the dataset are (2499,3)

## 2.
# Min-Max normalization
df['year'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
df['mileage'] = (df['mileage'] - df['mileage'].min()) / (df['mileage'].max() - df['mileage'].min())

# Display normalized data (first 5 rows)
print("First 5 rows after normalization:")
print(df.head())

## 3.
X = df[['year', 'mileage']]  # predictors
y = df['price']              # target

# Perform 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Show the size of each split
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

## 4.
k_values = [3, 10, 25, 50, 100, 300]
mse_results = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mse_results.append((k, mse))


    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"k = {k} | MSE = {mse:.2f}")
    plt.grid(True)
    plt.show()

print("\nMean Squared Error for different k values:")
for k, mse in mse_results:
    print(f"k = {k}: MSE = {mse:.2f}")

# As k increases the variance of the graph decreases. When k starts at 3, there is problem of overfitting.
# But as k increases, the variance starts to decrease and there is problem of underfitting.

## 5.
optimal_k, min_mse = min(mse_results, key=lambda x: x[1])
print(f"Optimal k = {optimal_k} with MSE = {min_mse:.2f}")
# Optimal k for this data is 50, with MSE = 110202549.3

## 6.
# When k has a smaller value, the model closely fit the training data and producted very scattered preditions which shows sgn of overfitting.
# When k has a larger value, the model is underfitting. Around k=50, it shows the best prediction accuracy because it is the optimal value of k.

# Visualization Question 1

## 1.
df = pd.read_csv('college_completion.csv')

## 2.
print("Dataset dimensions (rows, columns):", df.shape)

print("Number of observations:", df.shape[0])

print("\nVariables included:")
print(df.columns.tolist())

print("\nFirst 5 rows of the dataset:")
print(df.head())
# The dimension for the dataset is (3798,63). There are a lot of included variable name. See printed output.

## 3.
crosstab = pd.crosstab(df['control'], df['level'])

print(crosstab)

# After cross tabulate control and level, it shows that public schools dominate 2 year space and private not for profit are mainly 4 year institution.
# Private for profit have a almore even split between 2 year and 4 year.

## 4.
grad_data = pd.to_numeric(df['grad_100_value'], errors='coerce').dropna()

# Histogram
plt.figure(figsize=(8, 4))
sns.histplot(grad_data, bins=30, kde=False)
plt.title('Histogram of Graduation Rate (100% Time)')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Kernel Density Plot
plt.figure(figsize=(8, 4))
sns.kdeplot(grad_data)
plt.title('Kernel Density Estimate of Graduation Rate (100% Time)')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Boxplot
plt.figure(figsize=(8, 2))
sns.boxplot(x=grad_data)
plt.title('Boxplot of Graduation Rate (100% Time)')
plt.xlabel('Graduation Rate (%)')
plt.grid(True)
plt.show()

# Statistical Description
print("Statistical Description of Graduation Rate (100% Time):")
print(grad_data.describe())

## 5.
df['grad_100_value'] = pd.to_numeric(df['grad_100_value'], errors='coerce')

# KDE plot by control
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='grad_100_value', hue='control', common_norm=False)
plt.title('KDE of Graduation Rate by Control')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# KDE plot by level
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='grad_100_value', hue='level', common_norm=False)
plt.title('KDE of Graduation Rate by Level')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Grouped statistical description by control
print("\nGrouped Summary by Control:")
print(df.groupby('control')['grad_100_value'].describe())

# Grouped statistical description by level
print("\nGrouped Summary by Level:")
print(df.groupby('level')['grad_100_value'].describe())

# Private not for profit school with 4 year education have the highest graudation rates.

## 6.
df['levelXcontrol'] = df['level'] + ', ' + df['control']

# KDE plot grouped by levelXcontrol
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x='grad_100_value', hue='levelXcontrol', common_norm=False)
plt.title('KDE of Graduation Rate by Level and Control')
plt.xlabel('Graduation Rate (%)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Private not for profit with 4 year institutions has the best graduation rate.

## 7.
# KDE plot of aid_value (overall)
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='aid_value')
plt.title('KDE of Student Aid Value (Overall)')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# KDE plot grouped by level
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='aid_value', hue='level', common_norm=False)
plt.title('KDE of Student Aid Value by Level')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# KDE plot grouped by control
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='aid_value', hue='control', common_norm=False)
plt.title('KDE of Student Aid Value by Control')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# Grouped descriptive statistics
print("Descriptive Statistics by Level:")
print(df.groupby('level')['aid_value'].describe())

print("\nDescriptive Statistics by Control:")
print(df.groupby('control')['aid_value'].describe())

# The shape of the graph is right skewed which shows that most institutions offer moderate aid. The long tail
# of the graph streches the graph because there are some high value aid offered. KDE graph has a peak and right skewed.
# KDE of student aid value by level shows that 4 year institution has wider variance, whereas 2 year college has a peak
# between 0-10000.

## 8.
df['aid_value'] = pd.to_numeric(df['aid_value'], errors='coerce')
df['grad_100_value'] = pd.to_numeric(df['grad_100_value'], errors='coerce')

# Remove rows with missing values in either variable
df_clean = df.dropna(subset=['aid_value', 'grad_100_value'])

# 1. Overall scatterplot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='aid_value', y='grad_100_value', alpha=0.6)
plt.title('Graduation Rate vs. Student Aid (All Institutions)')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Graduation Rate (%)')
plt.grid(True)
plt.show()

# 2. Scatterplot grouped by level
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='aid_value', y='grad_100_value', hue='level', alpha=0.6)
plt.title('Graduation Rate vs. Student Aid by Level')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Graduation Rate (%)')
plt.grid(True)
plt.show()

# 3. Scatterplot grouped by control
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='aid_value', y='grad_100_value', hue='control', alpha=0.6)
plt.title('Graduation Rate vs. Student Aid by Control')
plt.xlabel('Average Aid Value ($)')
plt.ylabel('Graduation Rate (%)')
plt.grid(True)
plt.show()

# The graph shows a general upward trend, as average student aid increases, graduation rates also tend to increase.
# The relationship is also not perfectly linear because many institutions with low amount of air still have moderate graudation rates
# Grouped by level, 2 year institutions are tightly clustered to the left low quadrant and 4 year institutions show a greater spread.
# Grouped by control, private not for profit institution show the most strong positive association among all types. Private
# for profit offers low aid and also have the lowest graudation rates.