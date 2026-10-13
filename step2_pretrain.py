import torch
import torch.nn as nn
from torchvision import models


class AlexNetManual(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


if __name__ == '__main__':
    from common import get_loaders, train_and_eval
    batch_size = 16
    lr = 0.0001
    epochs = 30
    device = 'cpu'

    official = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    net = AlexNetManual(num_classes=1000)
    net.load_state_dict(official.state_dict())
    net.classifier[6] = nn.Linear(4096, 2)
    net.to(device)

    # 冻结：只训练classifier层
    for name, param in net.named_parameters():
        param.requires_grad = name.startswith("classifier")

    train_loader, val_loader = get_loaders("./datasets", batch_size=batch_size)
    train_and_eval(net, train_loader, val_loader, epochs=epochs, lr=lr, device=device)
