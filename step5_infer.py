import sys
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["cat 猫", "dog 狗"]


def load_model(weight_path="best_model.pth"):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(torch.load(weight_path, map_location="cpu"))
    net.eval()
    return net


def predict(image_path, net):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(x), dim=1)[0]
    pred = probs.argmax().item()
    return CLASSES[pred], probs[pred].item(), probs


if __name__ == '__main__':
    net = load_model(sys.argv[1] if len(sys.argv) > 1 else "best_model.pth")
    path = sys.argv[2] if len(sys.argv) > 2 else "./datasets/cat.1.jpg"
    label, conf, probs = predict(path, net)
    print(f"图片: {path}")
    print(f"预测: {label}，置信度 {conf:.2%}")
    print(f"各类概率: 猫 {probs[0]:.2%} | 狗 {probs[1]:.2%}")
