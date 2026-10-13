from torchvision import transforms
from common import get_loaders, train_and_eval, IMAGENET_MEAN, IMAGENET_STD
from torchvision import models
import torch
import torch.nn as nn


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

if __name__ == '__main__':
    train_loader, val_loader = get_loaders("./datasets", batch_size=16,
                                           train_transform=train_transform)
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Linear(net.fc.in_features, 2)

    for name, param in net.named_parameters():
        param.requires_grad = 'fc' in name

    acc = train_and_eval(net, train_loader, val_loader, epochs=5, lr=1e-3)
    print(f"\n【第三步 预训练+数据增强】最终准确率: {acc:.4f}")
