import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
DROPOUT_PROB = 0.5
IMAGE_SIZE = 224
DATA_DIR = './archive/minet'
NUM_CLASSES = 7

class SENet(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SENet, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def build_mobilenetv2_model(num_classes=7, dropout=0.5):
    model = models.mobilenet_v2(pretrained=True)

    # Replace classifier head
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.last_channel, num_classes)
    )

    return model


def build_mobilenetv2_senet(num_classes=7, dropout=0.5):
    base_model = models.mobilenet_v2(pretrained=True)
    
    for i, block in enumerate(base_model.features):
        if isinstance(block, models.mobilenetv2.InvertedResidual) and block.use_res_connect:
            block.add_module("se", SENet(block.conv[-1].num_features))
    
    base_model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(base_model.last_channel, num_classes)
    )
    return base_model


def build_vgg16_model(num_classes=7, dropout=0.5):
    model = models.vgg16(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = True  # fine-tune all layers

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(4096, num_classes)
    )
    return model


def build_resnet50_model(num_classes=7, dropout=0.5):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = True  # fine-tune all layers

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def build_inceptionv3_model(num_classes=7, dropout=0.5):
    model = models.inception_v3(pretrained=True, aux_logits=True)
    for param in model.parameters():
        param.requires_grad = True

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes)
    )
    return model


def build_alexnet_model(num_classes=7, dropout=0.5):
    model = models.alexnet(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = True

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes)
    )
    return model

def train_model(model, optimizer, criterion, model_name):
    best_acc = 0.0
    history = {'train_acc': [], 'val_acc': [], 'train_loss': []}

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if 'inception' in model_name and model.training:
                outputs, _ = model(images)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct.double() / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)

        val_acc = val_correct.double() / len(val_loader.dataset)
        history['val_acc'].append(val_acc.item())

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'best_model_{model_name}.pth')

    plot_learning_curve(history, model_name)

def test_model(model, model_name):
    model.load_state_dict(torch.load(f'best_model_{model_name}.pth'))
    model.eval()
    test_correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if 'inception' in model_name:
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            else:
                outputs = model(images)
        
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct.double() / len(test_loader.dataset)
    print(f"✅ Test Accuracy ({model_name}): {test_acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # Classification report
    print(classification_report(all_labels, all_preds, digits=4))
    
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return test_acc.item(), precision, recall, f1


def plot_learning_curve(history, model_name):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title(f'Loss - {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'learning_curve_{model_name}.png')
    plt.close()



if __name__ == '__main__':
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    val_set.dataset.transform = val_test_transforms
    test_set.dataset.transform = val_test_transforms

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)


    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_fn in [build_alexnet_model, build_vgg16_model, build_mobilenetv2_senet]:
        model_name = model_fn.__name__.replace("build_", "")
        print(f"\n🚀 Trénujem model: {model_name}")
        
        model = model_fn(num_classes=NUM_CLASSES, dropout=DROPOUT_PROB).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
        start_time = time.time()
        train_model(model, optimizer, criterion, model_name)
        duration = time.time() - start_time
        acc, precision, recall, f1 = test_model(model, model_name)
        results.append({
            'Model': model_name,
            'Test Accuracy (%)': round(acc * 100, 2),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1 Score': round(f1, 4),
            'Train Time (s)': round(duration, 2)
        })
    
    df = pd.DataFrame(results)
    print("\n📋 Summary of all model performances:")
    print(df)
    df.to_csv("model_comparison_results.csv", index=False)

