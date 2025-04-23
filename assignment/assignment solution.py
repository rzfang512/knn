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