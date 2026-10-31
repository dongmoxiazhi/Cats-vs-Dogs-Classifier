# 🐱🐶 Cats vs Dogs Image Classifier

基于 PyTorch 的猫狗图像二分类项目。使用预训练的 ResNet18 进行迁移学习，实现了极高的准确率和极快的训练速度。

## 📊 实验结果
- **Test Accuracy**: 89.75% (5 Epochs)
- **Training Time**: ~20s / epoch (RTX 5060)

## 🛠️ 技术亮点
- **Data Augmentation**: 使用 `RandomHorizontalFlip` 和 `Normalize` 防止过拟合。
- **Transfer Learning**: 加载 ImageNet 预训练权重，冻结底层 `layer1/layer2` 提取通用特征，仅微调 `layer3/layer4` 和分类头，训练速度提升 3 倍以上。
- **Subset Data**: 从 CIFAR-10 中精准抽离 10000 张猫/狗图片，并将标签映射为二分类任务（0为猫，1为狗）。

## 🚀 快速开始
1. 安装依赖：`pip install torch torchvision`
2. 运行代码：`python train.py` (确保本地数据集路径正确)