import os, time, copy
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

# ── SETTINGS ───────────────────────────────────────────────────────────────
SEED        = 499
DATA_DIR    = "/content/dataset/minet"
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
BASE_SIZE   = 224
NUM_CLASSES = 7
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN, STD   = [0.485,0.456,0.406], [0.229,0.224,0.225]

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


full = datasets.ImageFolder(DATA_DIR)
idxs, labels = list(range(len(full))), full.targets
train_idx, rem_idx, y_train, y_rem = train_test_split(
    idxs, labels, train_size=0.6, stratify=labels, random_state=SEED
)
val_idx, test_idx, _, _ = train_test_split(
    rem_idx, y_rem, train_size=0.5, stratify=y_rem, random_state=SEED
)

def build_mobilenet_v2(pretrained: bool):
     m = models.mobilenet_v2(pretrained=pretrained)
     return m, m.last_channel

# ── ARCHITECTURES ──────────────────────────────────────────────────────────
archs = {
    'MobileNetV2':  build_mobilenet_v2,
    'ResNet50':   lambda p: (models.resnet50(pretrained=p),   None),
    'InceptionV3':lambda p: (models.inception_v3(pretrained=p, aux_logits=True), None),
    'AlexNet':    lambda p: (models.alexnet(pretrained=p),    9216),
    'VGG16':      lambda p: (models.vgg16(pretrained=p),      25088),
}

# ── HELPERS ─────────────────────────────────────────────────────────────────
class SubsetWithTF(Dataset):
    def __init__(self, base_ds, idxs, tfm):
        self.samples, self.idxs, self.tfm = base_ds.samples, idxs, tfm
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        path,label = self.samples[self.idxs[i]]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), label

def train_model(model, crit, opt, train_loader, val_loader, epochs, freeze_bb=False):
    history = {'train_acc':[], 'val_acc':[], 'train_loss':[], 'val_loss':[]}
    best_wts, best_acc = copy.deepcopy(model.state_dict()), 0.0

    # optionally freeze backbone
    if freeze_bb and hasattr(model, 'features'):
        for p in model.features.parameters(): p.requires_grad = False

    for e in range(epochs):
        for phase, loader in (('train', train_loader), ('val', val_loader)):
            model.train(phase=='train')
            run_loss=run_corr=0
            for x,y in loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                with torch.set_grad_enabled(phase=='train'):
                    out = model(x)
                    # Handle InceptionV3 which returns InceptionOutputs(logits, aux_logits)
                    if hasattr(out, 'logits') and hasattr(out, 'aux_logits'):
                        loss1 = crit(out.logits, y)
                        loss2 = crit(out.aux_logits, y)
                        loss = loss1 + 0.4 * loss2  # combine main and auxiliary losses
                        preds = out.logits.argmax(1)
                    
                    # Handle InceptionV3 as tuple (older versions of PyTorch)
                    elif isinstance(out, tuple) and len(out) == 2:
                        loss1 = crit(out[0], y)
                        loss2 = crit(out[1], y)
                        loss = loss1 + 0.4 * loss2
                        preds = out[0].argmax(1)
                    
                    # All other models
                    else:
                        loss = crit(out, y)
                        preds = out.argmax(1)
                    
                    if phase=='train':
                        loss.backward(); opt.step()
                run_loss   += loss.item()*x.size(0)
                run_corr   += (preds==y).sum().item()
            size = len(loader.dataset)
            epoch_loss = run_loss/size
            epoch_acc  = run_corr/size
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc)
            if phase=='val' and epoch_acc>best_acc:
                best_acc, best_wts = epoch_acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_wts)
    return model, history

def test_model(model, crit, loader):
    model.eval()
    run_loss=run_corr=0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        with torch.no_grad():
            out = model(x)
            loss= crit(out,y)
        run_loss += loss.item()*x.size(0)
        run_corr += (out.argmax(1)==y).sum().item()
    size = len(loader.dataset)
    return run_corr/size, run_loss/size

def plot_acc(hists, tests, title):
    plt.figure(figsize=(6,4))
    colors = ['C0','C1','C2']
    for i,(k,h) in enumerate(hists.items()):
        ep = range(1, len(h['train_acc'])+1)
        plt.plot(ep, h['train_acc'], marker='s', linestyle='-',  color=colors[i], label=f'{k} train')
        plt.plot(ep, h['val_acc'],   marker='o', linestyle='--', color=colors[i], label=f'{k} val')
        # test marker
        ta,_ = tests[k]
        plt.plot(len(ep), ta, marker='*', ms=12, color=colors[i], label=f'{k} test={ta:.3f}')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title(f'{title} Accuracy'); plt.ylim(0,1)
    plt.legend(loc='lower right', fontsize='small'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.show()

def plot_loss(hists, tests, title):
    plt.figure(figsize=(6,4))
    colors = ['C0','C1','C2']
    for i,(k,h) in enumerate(hists.items()):
        ep = range(1, len(h['train_loss'])+1)
        plt.plot(ep, h['train_loss'], marker='s', linestyle='-',  color=colors[i], label=f'{k} train-loss')
        plt.plot(ep, h['val_loss'],   marker='o', linestyle='--', color=colors[i], label=f'{k} val-loss')
        # test loss
        _, tl = tests[k]
        plt.plot(len(ep), tl, marker='*', ms=12, color=colors[i], label=f'{k} test-loss={tl:.3f}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title(f'{title} Loss'); plt.legend(loc='upper right', fontsize='small')
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ── MAIN LOOP ────────────────────────────────────────────────────────────────
for name, build_fn in archs.items():
    print(f'\n==== {name} ====')
    # pick correct input size
    input_size = 299 if name=='InceptionV3' else BASE_SIZE
    crop_size  = input_size - 80

    # make transforms
    tf_scratch  = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD),
    ])
    tf_tl       = tf_scratch  # freeze backbone, same
    edge        = transforms.Compose([transforms.FiveCrop(crop_size), transforms.Lambda(lambda c: c[0])])
    tf_id       = tf_scratch
    tf_center   = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.CenterCrop(crop_size),
                                      transforms.Resize((input_size,input_size)),
                                      transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tf_edge     = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      edge,
                                      transforms.Resize((input_size,input_size)),
                                      transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tf_zoom     = transforms.Compose([transforms.RandomResizedCrop(input_size,scale=(0.8,1.2)),
                                      transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tf_bright   = transforms.Compose([transforms.Resize((input_size,input_size)),
                                      transforms.ColorJitter(brightness=0.4),
                                      transforms.Resize((input_size,input_size)),
                                      transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    tf_val      = tf_scratch

    # datasets & loaders
    full_ds = datasets.ImageFolder(DATA_DIR)
    # static 5x DA train
    ds0 = SubsetWithTF(full_ds, train_idx, tf_id)
    ds1 = SubsetWithTF(full_ds, train_idx, tf_center)
    ds2 = SubsetWithTF(full_ds, train_idx, tf_edge)
    ds3 = SubsetWithTF(full_ds, train_idx, tf_zoom)
    ds4 = SubsetWithTF(full_ds, train_idx, tf_bright)
    train_da_ds = ConcatDataset([ds0,ds1,ds2,ds3,ds4])

    ds_scratch = SubsetWithTF(full_ds, train_idx, tf_scratch)
    ds_tl      = SubsetWithTF(full_ds, train_idx, tf_tl)
    ds_val     = SubsetWithTF(full_ds, val_idx,   tf_val)
    ds_test    = SubsetWithTF(full_ds, test_idx,  tf_val)

    loaders = {
        'scratch': DataLoader(ds_scratch, batch_size=BATCH_SIZE, shuffle=True),
        'tl':      DataLoader(ds_tl,      batch_size=BATCH_SIZE, shuffle=True),
        'da':      DataLoader(train_da_ds,batch_size=BATCH_SIZE, shuffle=True),
        'val':     DataLoader(ds_val,     batch_size=BATCH_SIZE, shuffle=False),
        'test':    DataLoader(ds_test,    batch_size=BATCH_SIZE, shuffle=False),
    }
    sizes = {k: len(v.dataset if isinstance(v.dataset,Dataset) else v) for k,v in loaders.items()}

    # placeholders
    histories, test_results = {}, {}
    crit = nn.CrossEntropyLoss()

    # --- 1) Scratch ---
    m, feat = build_fn(False)
    # replace head
    if name=='ResNet50':
        nf = m.fc.in_features; m.fc = nn.Linear(nf, NUM_CLASSES)
    elif name == 'InceptionV3':
        nf = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(nf, NUM_CLASSES)
        nf = m.fc.in_features
        m.fc = nn.Linear(nf, NUM_CLASSES)
    else:
        setattr(m, 'classifier',
                nn.Sequential(nn.Dropout(0.5), nn.Linear(feat, NUM_CLASSES)))
    m = m.to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=LR)
    m,h = train_model(m, crit, opt, loaders['scratch'], loaders['val'], EPOCHS, freeze_bb=False)
    ta,tl = test_model(m, crit, loaders['test'])
    histories['scratch'],  test_results['scratch']  = h, (ta,tl)

    # --- 2) TL‑only ---
    m, feat = build_fn(True)
    if name=='ResNet50':
        nf = m.fc.in_features; m.fc = nn.Linear(nf, NUM_CLASSES)
    elif name == 'InceptionV3':
        nf = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(nf, NUM_CLASSES)
        nf = m.fc.in_features
        m.fc = nn.Linear(nf, NUM_CLASSES)
    else:
        setattr(m, 'classifier',
                nn.Sequential(nn.Dropout(0.5), nn.Linear(feat, NUM_CLASSES)))
    # freeze backbone
    if hasattr(m,'features'):
        for p in m.features.parameters(): p.requires_grad=False
    m = m.to(DEVICE)
    opt = optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=LR)
    m,h = train_model(m, crit, opt, loaders['tl'], loaders['val'], EPOCHS, freeze_bb=False)
    ta,tl = test_model(m, crit, loaders['test'])
    histories['tl'],         test_results['tl']       = h, (ta,tl)

    # --- 3) TL+DA ---
    m, feat = build_fn(True)
    if name=='ResNet50':
        nf = m.fc.in_features; m.fc = nn.Linear(nf, NUM_CLASSES)
    elif name == 'InceptionV3':
        nf = m.AuxLogits.fc.in_features
        m.AuxLogits.fc = nn.Linear(nf, NUM_CLASSES)
        nf = m.fc.in_features
        m.fc = nn.Linear(nf, NUM_CLASSES)
    else:
        setattr(m, 'classifier',
                nn.Sequential(nn.Dropout(0.5), nn.Linear(feat, NUM_CLASSES)))
    #if hasattr(m,'features'):
    #    for p in m.features.parameters(): p.requires_grad=False
    m = m.to(DEVICE)
    opt = optim.Adam(filter(lambda p:p.requires_grad, m.parameters()), lr=LR)
    m,h = train_model(m, crit, opt, loaders['da'], loaders['val'], EPOCHS, freeze_bb=False)
    ta,tl = test_model(m, crit, loaders['test'])
    histories['da'],         test_results['da']       = h, (ta,tl)

    # ── PLOTS ───────────────────────────────────────────────────────────────
    plot_acc(histories, test_results, f"{name} (in={input_size})")
    plot_loss(histories, test_results, f"{name} (in={input_size})")