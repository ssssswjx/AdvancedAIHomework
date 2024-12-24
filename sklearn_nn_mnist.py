from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 加载数据集
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0

# 划分数据集
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 实验不同的学习率和隐层节点数
learning_rates = [0.1, 0.01, 0.001, 0.0001]
hidden_layer_sizes = [(500,), (1000,), (1500,), (2000,)]
results = []

for lr in learning_rates:
    for hidden in hidden_layer_sizes:
        print(f"Training NN with learning_rate={lr}, hidden_layers={hidden}")
        mlp = MLPClassifier(hidden_layer_sizes=hidden, 
                          learning_rate_init=lr,
                          max_iter=100,
                          random_state=42,
                          solver='sgd',)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        error_count = (y_test != y_pred).sum()
        results.append({
            'model': 'Neural Network',
            'learning_rate': lr,
            'hidden_layers': hidden[0],
            'accuracy': accuracy,
            'error_count': error_count
        })
        print(f"Accuracy: {accuracy:.4f}, Errors: {error_count}")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('nn_results.csv', index=False)

# 可视化结果
plt.figure(figsize=(12, 6))
for lr in learning_rates:
    df_subset = results_df[results_df['learning_rate'] == lr]
    plt.plot(df_subset['hidden_layers'], 
             df_subset['accuracy'], 
             'o-', 
             label=f'Learning Rate = {lr}')

plt.xlabel('Hidden Layer Size')
plt.ylabel('Accuracy')
plt.title('SKLearn Neural Network: Architecture vs Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('sklearn_nn_results.png')
plt.show()
