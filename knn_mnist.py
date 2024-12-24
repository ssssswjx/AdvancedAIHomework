import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd

# 加载MNIST数据集
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # 归一化

# 划分训练集和测试集
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 不同k值的实验
k_values = [1, 3, 5, 7]
results = []

for k in k_values:
    print(f"Training KNN with k={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    error_count = (y_test != y_pred).sum()
    results.append({
        'model': 'KNN',
        'k': k,
        'accuracy': accuracy,
        'error_count': error_count
    })
    print(f"Accuracy: {accuracy:.4f}, Errors: {error_count}")

# 保存实验结果
results_df = pd.DataFrame(results)
results_df.to_csv('knn_results.csv', index=False)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(k_values, [r['accuracy'] for r in results], 'o-', linewidth=2)
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.title('KNN: K Value vs Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.savefig('knn_results.png')
plt.show()

# 创建对比柱状图
plt.figure(figsize=(10, 6))
plt.bar([str(k) for k in k_values], 
        [r['error_count'] for r in results],
        color='lightcoral')
plt.xlabel('K Value')
plt.ylabel('Error Count')
plt.title('KNN: K Value vs Error Count')
plt.grid(True, axis='y')
plt.savefig('knn_errors.png')
plt.show()
