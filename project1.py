import csv
import numpy as np

def load_mnist(filename, limit=1000):
    data = []
    labels = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            labels.append(int(row[0]))
            data.append(np.array(row[1:], dtype=np.float32) / 255.0)  # Normalize
    return np.array(data), np.array(labels)

def knn_predict(X_train, y_train, x_test, k=3):
    distances = np.linalg.norm(X_train - x_test, axis=1)
    k_indices = distances.argsort()[:k]
    k_labels = y_train[k_indices]
    return np.bincount(k_labels).argmax()

X, y = load_mnist("mnist_train.csv", limit=1200)
X_train, y_train = X[:1000], y[:1000]
X_test, y_test = X[1000:], y[1000:]

correct = 0
for i in range(len(X_test)):
    pred = knn_predict(X_train, y_train, X_test[i])
    if pred == y_test[i]:
        correct += 1

print("Accuracy:", correct / len(X_test))
