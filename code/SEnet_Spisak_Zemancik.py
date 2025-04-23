import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# ── SETTINGS ────────────────────────────────────────────────────────────────
SEED           = 231
DATA_DIR       = "/content/dataset/minet"
BATCH_SIZE     = 32
NUM_EPOCHS     = 50
FREEZE_EPOCHS  = 5
LR             = 1e-4
IMAGE_SIZE     = 224
CROP_SIZE      = IMAGE_SIZE - 80
NUM_CLASSES    = 7
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── SPLIT DATASET 60/20/20 ───────────────────────────────────────────────────
full_ds = datasets.ImageFolder(DATA_DIR)
indices, labels = list(range(len(full_ds))), full_ds.targets

train_idx, rem_idx, y_train, y_rem = train_test_split(
    indices, labels, train_size=0.6, stratify=labels, random_state=SEED
)
val_idx, test_idx, _, _ = train_test_split(
    rem_idx, y_rem, train_size=0.5, stratify=y_rem, random_state=SEED
)

# ── TRANSFORMS ───────────────────────────────────────────────────────────────
# TL‑only: just resize + normalize
tl_only_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# TL+DA: resize → random choice of 4 aug → resize → to tensor → normalize

edge_crop = transforms.Compose([
    transforms.FiveCrop(CROP_SIZE),
    transforms.Lambda(lambda crops: crops[0])
])
"""
tl_da_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomChoice([
        transforms.CenterCrop(CROP_SIZE),
        edge_crop,
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8,1.2)),
        transforms.ColorJitter(brightness=0.4),
    ]),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
"""



from torch.utils.data import ConcatDataset

# 1) define each of the 5 transforms
tf_identity = tl_only_tf  # Resize+Normalize
tf_center   = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
])
tf_edge     = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    edge_crop,
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
])
tf_zoom     = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8,1.2)),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
])
tf_bright   = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ColorJitter(brightness=0.4),
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor(), transforms.Normalize(MEAN,STD),
])








# val/test: resize + normalize
val_test_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ── SUBSET WRAPPER ───────────────────────────────────────────────────────────
class SubsetWithTF(Dataset):
    def __init__(self, base_ds, indices, tfm):
        self.samples = base_ds.samples
        self.indices = indices
        self.tfm     = tfm
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        path, lbl = self.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), lbl

train_ds_tl   = SubsetWithTF(full_ds, train_idx, tl_only_tf)
ds_id   = SubsetWithTF(full_ds, train_idx, tf_identity)
ds_ctr  = SubsetWithTF(full_ds, train_idx, tf_center)
ds_edge = SubsetWithTF(full_ds, train_idx, tf_edge)
ds_zn   = SubsetWithTF(full_ds, train_idx, tf_zoom)
ds_br   = SubsetWithTF(full_ds, train_idx, tf_bright)

# 3) concatenate them into one big 5× dataset
train_da_5x = ConcatDataset([ds_id, ds_ctr, ds_edge, ds_zn, ds_br])
val_ds        = SubsetWithTF(full_ds, val_idx,   val_test_tf)
test_ds       = SubsetWithTF(full_ds, test_idx,  val_test_tf)

loader_tl     = DataLoader(train_ds_tl, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_da     = DataLoader(train_da_5x, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_val    = DataLoader(val_ds,      batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loader_test   = DataLoader(test_ds,     batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

dataset_sizes = {
    'train_tl': len(train_ds_tl),
    'train_da': len(train_da_5x),
    'val':      len(val_ds),
    'test':     len(test_ds),
}

# ── SENet MODULE ─────────────────────────────────────────────────────────────
class SENetBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels,   bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ── TRAIN / VALIDATION LOOP ─────────────────────────────────────────────────
def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs, freeze_epochs=0):
    since = time.time()
    best_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0

    # initial freeze if requested
    if freeze_epochs > 0 and hasattr(model, 'features'):
        for p in model.features.parameters():
            p.requires_grad = False

    for epoch in range(num_epochs):
        # unfreeze after freeze_epochs
        if epoch == freeze_epochs and hasattr(model, 'features'):
            for p in model.features.parameters():
                p.requires_grad = True

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        for phase, loader in (('train', train_loader), ('val', val_loader)):
            model.train(phase=='train')
            running_loss = running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(1)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_corrects+= (preds == labels).sum().item()

            ds_size = dataset_sizes['train_da'] if phase=='train' else dataset_sizes['val']
            epoch_loss = running_loss / ds_size
            epoch_acc  = running_corrects / ds_size
            print(f"{phase:>4}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase=='val' and epoch_acc>best_acc:
                best_acc, best_wts = epoch_acc, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    print(f"\nBest val Acc: {best_acc:.4f}")
    return model

# ── TEST LOOP ────────────────────────────────────────────────────────────────
def test_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            preds = model(inputs).argmax(1)
            correct += (preds == labels).sum().item()
    acc = correct / dataset_sizes['test']
    print(f"Test Acc: {acc:.4f}")
    return acc

# ── EXPERIMENTS ─────────────────────────────────────────────────────────────
results = {}
crit = nn.CrossEntropyLoss()
# 1) TL‑only (resize+normalize)

print("\n=== Experiment: TL‑only ===")
m1 = models.mobilenet_v2(pretrained=True)
#for p in m1.features.parameters():
#    p.requires_grad = False
m1.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(m1.last_channel, NUM_CLASSES)
)
m1 = m1.to(DEVICE)
opt1 = optim.Adam(filter(lambda p: p.requires_grad, m1.parameters()), lr=LR)
crit = nn.CrossEntropyLoss()

m1 = train_model(m1, crit, opt1, loader_tl, loader_val, NUM_EPOCHS, freeze_epochs=0)
results['TL-only'] = test_model(m1, loader_test)

# 2) TL+DA
print("\n=== Experiment: TL+DA ===")
m2 = models.mobilenet_v2(pretrained=True)
#for p in m2.features.parameters():
#    p.requires_grad = False
m2.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(m2.last_channel, NUM_CLASSES)
)
m2 = m2.to(DEVICE)
opt2 = optim.Adam(filter(lambda p: p.requires_grad, m2.parameters()), lr=LR)

m2 = train_model(m2, crit, opt2, loader_da, loader_val, NUM_EPOCHS, freeze_epochs=0)
results['TL+DA'] = test_model(m2, loader_test)

# 3) TL+DA+SENet
print("\n=== Experiment: TL+DA+SENet ===")
backbone = models.mobilenet_v2(pretrained=True)
#for p in backbone.features.parameters():
#    p.requires_grad = False

m3 = nn.Sequential(
    backbone.features,
    SENetBlock(backbone.last_channel),
    nn.AdaptiveMaxPool2d(1),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.Linear(backbone.last_channel, NUM_CLASSES)
)
m3 = m3.to(DEVICE)
opt3 = optim.Adam(filter(lambda p: p.requires_grad, m3.parameters()), lr=LR)

m3 = train_model(m3, crit, opt3, loader_da, loader_val, NUM_EPOCHS, freeze_epochs=0)
results['TL+DA+SENet'] = test_model(m3, loader_test)

# ── SUMMARY ─────────────────────────────────────────────────────────────────
print("\nSummary of test accuracies:")
for name, acc in results.items():
    print(f"  {name:12s} → {acc:.4f}")
