import numpy as np
from sklearn.datasets import fetch_openml
import pandas as pd
import matplotlib.pyplot as plt

class NumpyNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        batch_size = X.shape[0]
        
        delta3 = output
        delta3[range(batch_size), y] -= 1
        
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = np.dot(delta3, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # 更新权重
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

# 数据加载和预处理
print("Loading MNIST dataset...")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X / 255.0  # 归一化
y = y.astype(int)

# 划分数据集
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 实验参数设置
hidden_sizes = [500, 1000, 1500, 2000]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
results = []

# 训练和评估
for hidden_size in hidden_sizes:
    for lr in learning_rates:
        print(f"\nTraining with hidden_size={hidden_size}, learning_rate={lr}")
        
        # 初始化模型
        model = NumpyNeuralNetwork(
            input_size=784,
            hidden_size=hidden_size,
            output_size=10,
            learning_rate=lr
        )
        
        # 训练过程
        batch_size = 128
        epochs = 100
        
        for epoch in range(epochs):
            # 随机打乱训练数据
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # 批量训练
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]
                
                # 前向传播
                output = model.forward(X_batch)
                # 反向传播
                model.backward(X_batch, y_batch, output)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch + 1} completed")
        
        # 评估模型
        predictions = []
        batch_size = 256  # 测试时使用更大的批量以加速
        for i in range(0, len(X_test), batch_size):
            X_batch = X_test[i:i+batch_size]
            output = model.forward(X_batch)
            pred = np.argmax(output, axis=1)
            predictions.extend(pred)
        
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y_test)
        error_count = np.sum(predictions != y_test)
        
        # 记录结果
        results.append({
            'model': 'Numpy NN',
            'hidden_size': hidden_size,
            'learning_rate': lr,
            'accuracy': accuracy,
            'error_count': error_count
        })
        
        print(f"Results for hidden_size={hidden_size}, learning_rate={lr}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Error count: {error_count}")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('numpy_nn_results.csv', index=False)

# 可视化结果
plt.figure(figsize=(12, 6))
for lr in learning_rates:
    df_subset = results_df[results_df['learning_rate'] == lr]
    plt.plot(df_subset['hidden_size'], 
             df_subset['accuracy'], 
             'o-', 
             label=f'Learning Rate = {lr}')

plt.xlabel('Hidden Layer Size')
plt.ylabel('Accuracy')
plt.title('Numpy Neural Network: Architecture vs Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('numpy_nn_results.png')
plt.show()
