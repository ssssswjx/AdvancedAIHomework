import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 超参数
num_epochs = 100
batch_size = 64
learning_rate = 0.001

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_data = torchvision.datasets.MNIST(root='../mnist_data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
val_data = torchvision.datasets.MNIST(root='../mnist_data', 
                                      train=False, 
                                      transform=transforms.ToTensor(),
                                      download=False)                               
tain_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)                                     
val_dataloader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)    

model = torchvision.models.resnet18(weights=None)  

#修改第一层卷积层
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
# 修改最后一层
model.fc = nn.Linear(512, 10)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 记录训练过程
train_losses = []
train_accuracies = []
val_accuracies = []
results = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tain_dataloader:
        # 将数据移动到正确的设备
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算训练集准确率
    epoch_loss = running_loss / len(tain_dataloader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    # 验证集评估
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_acc = 100 * val_correct / val_total
    val_accuracies.append(val_acc)
    
    # 记录结果
    results.append({
        'epoch': epoch + 1,
        'train_loss': epoch_loss,
        'train_accuracy': epoch_acc,
        'val_accuracy': val_acc
    })
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
          f'Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('cnn_results.csv', index=False)

# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')

# 可视化训练过程
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('cnn_training.png')
plt.show()

# 最终混淆矩阵可视化
from sklearn.metrics import confusion_matrix
import seaborn as sns

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_dataloader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('cnn_confusion_matrix.png')
plt.show()

# 打印最终结果
final_train_acc = train_accuracies[-1]
final_val_acc = val_accuracies[-1]
print(f'\nFinal Results:')
print(f'Training Accuracy: {final_train_acc:.2f}%')
print(f'Validation Accuracy: {final_val_acc:.2f}%')