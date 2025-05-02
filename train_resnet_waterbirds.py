import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
import pandas as pd
import os
import sys

# Waterbirds Datasetの定義 (train, val, test の3つの分割)
class WaterbirdsDataset(Dataset):
    def __init__(self, root_dir, metadata_path, split, transform=None):
        self.root_dir = root_dir
        self.meta = pd.read_csv(metadata_path)
        self.meta = self.meta[self.meta['split'] == split]  # 0: train, 1: val, 2: test
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = os.path.join(self.root_dir, row['img_filename'])
        image = Image.open(img_path).convert("RGB")
        label = int(row['y'])  # 0: landbird, 1: waterbird
        if self.transform:
            image = self.transform(image)
        return image, label

def main():
    # ResNet-50 をロードし出力層を2クラス用に変更
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # データ変換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # データセットの分割
    root_dir = 'datasets/waterbirds/waterbird_complete95_forest2water2'
    metadata_path = 'datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv'
    train_set = WaterbirdsDataset(root_dir=root_dir, metadata_path=metadata_path, split=0, transform=transform)
    val_set = WaterbirdsDataset(root_dir=root_dir, metadata_path=metadata_path, split=1, transform=transform)
    test_set = WaterbirdsDataset(root_dir=root_dir, metadata_path=metadata_path, split=2, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)
    test_loader = DataLoader(test_set, batch_size=64)

    # 学習ループ
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total:.4f}, Acc: {acc:.4f}")

    # モデルの保存
    torch.save(model.state_dict(), 'resnet50_waterbirds.pth')
    print("✅ モデルを保存しました：resnet50_waterbirds.pth")

    # 評価
    def evaluate(model, loader, split_name="Test"):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = (all_preds == all_labels).float().mean().item()
        cm = confusion_matrix(all_labels, all_preds)
        print(f"{split_name} Accuracy: {acc:.4f}")
        print(classification_report(all_labels, all_preds, target_names=["landbird", "waterbird"]))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["landbird", "waterbird"], yticklabels=["landbird", "waterbird"])
        plt.title(f'{split_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    evaluate(model, val_loader, split_name="Validation")
    evaluate(model, test_loader, split_name="Test")


    log_file = open('training_log.txt', 'w')
    sys.stdout = sys.stderr = log_file
    log_file.close()

if __name__ == "__main__":
    main()
