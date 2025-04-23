from PIL import Image
import pandas as pd
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

## 1.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train shape:", X_train.shape)  # (60000, 28, 28)
print("X_test shape:", X_test.shape)    # (10000, 28, 28)
print("y_train shape:", y_train.shape)  # (60000,)
print("y_test shape:", y_test.shape)    # (10000,)

np.set_printoptions(edgeitems=30, linewidth=100000)
for i in range(5):
    print(f"Label: {y_test[i]}\n")
    print(X_test[i], '\n')
    plt.contourf(np.rot90(X_test[i].T))
    plt.title(f"Digit = {y_test[i]}")
    plt.axis('off')
    plt.show()

X_train_flat = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flat = X_test.reshape(X_test.shape[0], -1) / 255.0

accuracies = []
k_vals = range(1, 10)
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)
    preds = knn.predict(X_test_flat)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc:.4f}")

plt.plot(k_vals, accuracies, marker='o')
plt.title("Accuracy vs k for kNN on MNIST")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

## 2.

best_k = k_vals[np.argmax(accuracies)]
print(f"\nBest k = {best_k}")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_flat, y_train)
y_pred = knn.predict(X_test_flat)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with k={best_k}: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (k={best_k})")
plt.show()

i = 0

print("X_train shape:", X_train.shape)      # (60000, 28, 28)
print("X_test shape:", X_test.shape)        # (10000, 28, 28)
print("X_train[i] shape:", X_train[i].shape)  # (28, 28)
print("X_test[i] shape:", X_test[i].shape)    # (28, 28)
print("y_train shape:", y_train.shape)      # (60000,)
print("y_test shape:", y_test.shape)        # (10000,)

plt.imshow(X_train[i], cmap='gray')
plt.title(f"Label: {y_train[i]}")
plt.axis('off')
plt.show()

## 3.
print("Original X_train shape:", X_train.shape)  # (60000, 28, 28)
print("Original X_test shape:", X_test.shape)    # (10000, 28, 28)


X_train_flat = X_train.reshape(X_train.shape[0], 784)
X_test_flat = X_test.reshape(X_test.shape[0], 784)

print("Reshaped X_train shape:", X_train_flat.shape)  # (60000, 784)
print("Reshaped X_test shape:", X_test_flat.shape)    # (10000, 784)

index = 0
X_test_single = X_test[index].reshape((1, 784))
print(f"Shape of X_test[{index}] after reshaping:", X_test_single.shape)  # (1, 784)

## 4.
X_train_flat = X_train.reshape(X_train.shape[0], 784) / 255.0  # Normalize pixel values
X_test_flat = X_test.reshape(X_test.shape[0], 784) / 255.0

k_values = range(1, 11)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_flat, y_train)
    y_pred = knn.predict(X_test_flat)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k}, Test Accuracy = {acc:.4f}")

plt.plot(k_values, accuracies, marker='o')
plt.title("kNN Classifier Accuracy on MNIST")
plt.xlabel("k (number of neighbors)")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()

best_k = k_values[np.argmax(accuracies)]
print(f"\n✅ Optimal k: {best_k} with accuracy = {max(accuracies):.4f}")

## 5.
best_k = k_values[np.argmax(accuracies)]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_flat, y_train)

y_pred = knn_best.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f" Accuracy on test set with k = {best_k}: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (k={best_k})")
plt.grid(False)
plt.show()

# The most frequent mistakes includes misclassifying 4 as 9 which would be because vertical strokes and loop structure.
# 3 and 5 also casues confusion and there are few errors between 7 and 1.

## 6.
img = Image.open("your_image.jpg").convert('RGB')
img = img.resize((10, 10))  # Resize for simplicity, can skip or adjust size

# Convert image to NumPy array
img_array = np.array(img)  # shape: (height, width, 3)

# Get image dimensions
height, width, _ = img_array.shape

# Create tabular RGB data
data = []

for row in range(height):
    for col in range(width):
        r, g, b = img_array[row, col]
        data.append({'Row': row, 'Col': col, 'Red': r, 'Green': g, 'Blue': b})

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

# Save to CSV if needed
df.to_csv("image_rgb_table.csv", index=False)

# By using img, we are able to convert the image to represent a color photo.