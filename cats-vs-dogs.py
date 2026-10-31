import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import time
import random
import numpy as np
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def get_cats_dogs_subset(dataset):
    indices = [i for i, (_, label) in enumerate(dataset) if label in (3, 5)]
    return Subset(dataset, indices)

train_datasets = datasets.CIFAR10(root="D:/d130/data", download=False, transform=transform_train, train=True)
test_datasets = datasets.CIFAR10(root="D:/d130/data", download=False, train=False, transform=transform_test)

cat_dog_train = get_cats_dogs_subset(train_datasets)
cat_dog_test = get_cats_dogs_subset(test_datasets)

print(f"训练集猫狗图片数量: {len(cat_dog_train)}") 
print(f"测试集猫狗图片数量: {len(cat_dog_test)}")

cat_dog_train_loader = DataLoader(cat_dog_train, shuffle=True, batch_size=64)
cat_dog_test_loader = DataLoader(cat_dog_test, shuffle=False, batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备：{device}")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name, param in model.named_parameters():
    if "layer1" in name or "layer2" in name or "conv1" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            y_binary = (y - 3) // 2 
            pred = model(X).argmax(dim=1)
            correct += (pred == y_binary).sum().item()
            total += y.size(0)
    return correct / total

epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    for X, y in cat_dog_train_loader:
        X, y = X.to(device), y.to(device)
        y_binary = (y - 3) // 2 
        
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y_binary)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(cat_dog_train_loader)
    test_acc = evaluate(model, cat_dog_test_loader)
    cost_time = time.time() - start_time
    print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}% | 耗时: {cost_time:.1f}s")