import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2))

    def forward(self, X):
        return self.net(X)


def init_weight(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)


class DatasetLoader(Dataset):
    def __init__(self, dataset_path):
        self.images_path = []
        self.labels = []
        with open(dataset_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(' ', 1)
                if len(parts) == 2:
                    self.images_path.append(parts[0])
                    self.labels.append(int(parts[1]))

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return image_transform(image)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        label = self.labels[index]
        image = self.preprocess_image(image_path)
        return image, label

    def __len__(self):
        return len(self.labels)


def dataloader(base_path, batch_size):
    imgs_abs_path = [os.path.join(base_path, i) for i in os.listdir(base_path)
                     if i.lower().endswith(('.jpg', '.jpeg', '.png'))]
    test_size = int(0.2 * len(imgs_abs_path))
    test_files = random.sample(imgs_abs_path, test_size)

    train_txt = "trainDatasets.txt"
    test_txt = "testDatasets.txt"

    with open(test_txt, 'w', encoding='utf-8') as f1, open(train_txt, 'w', encoding='utf-8') as f2:
        for i in imgs_abs_path:
            label = 0 if 'cat' in i.lower() else 1
            if i in test_files:
                f1.write(f"{i} {label}\n")
            else:
                f2.write(f"{i} {label}\n")
    TrainDataset = DatasetLoader(train_txt)
    TestDataset = DatasetLoader(test_txt)
    return DataLoader(TrainDataset, batch_size=batch_size, shuffle=True), DataLoader(TestDataset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    batch_size = 16
    lr = 0.0001
    epochs = 30
    device = 'cpu'
    net = AlexNet()
    net.apply(init_weight)
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    trainer = torch.optim.Adam(net.parameters(), lr=lr)

    base_path = r"./datasets"
    train_iter, test_iter = dataloader(base_path, batch_size=batch_size)

    print("开始训练...")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for X, Y in train_iter:
            X, Y = X.to(device), Y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, Y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            running_loss += l.item() * X.size(0)

        net.eval()
        all_preds = []
        all_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in test_iter:
                X, Y = X.to(device), Y.to(device)
                y_hat = net(X)
                loss = loss_fn(y_hat, Y)
                val_loss += loss.item() * X.size(0)
                _, preds = torch.max(y_hat, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(Y.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        print(f"EPOCH: {epoch+1}/{epochs}")
        print(f"  -> Test Loss: {val_loss / len(test_iter.dataset):.4f}")
        print(f"  -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1‑Score: {f1:.4f}\n")
