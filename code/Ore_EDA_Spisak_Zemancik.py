import os
import random
from collections import Counter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Path to your dataset root (each subfolder = one ore class)
DATA_DIR = './minet'

# 1) Count images per class
classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
counts = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in classes}

# Plot class distribution
plt.figure(figsize=(8,5))
plt.bar(counts.keys(), counts.values(), color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Ore Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution')
plt.tight_layout()
plt.show()

# 2) Sample images in a 4+3 layout (centered)
fig = plt.figure(figsize=(12, 6))
top_row_y = 0.55
bottom_row_y = 0.05
image_width = 0.22
image_height = 0.4
h_spacing = 0.02

for i in range(4):
    left = i * (image_width + h_spacing)
    ax = fig.add_axes([left, top_row_y, image_width, image_height])
    fn = random.choice(os.listdir(os.path.join(DATA_DIR, classes[i])))
    img = Image.open(os.path.join(DATA_DIR, classes[i], fn)).convert('RGB')
    ax.imshow(img)
    ax.set_title(classes[i])
    ax.axis('off')

total_width = 3 * image_width + 2 * h_spacing
start_left = (1 - total_width) / 2

for j in range(3):
    left = start_left + j * (image_width + h_spacing)
    ax = fig.add_axes([left, bottom_row_y, image_width, image_height])
    fn = random.choice(os.listdir(os.path.join(DATA_DIR, classes[4 + j])))
    img = Image.open(os.path.join(DATA_DIR, classes[4 + j], fn)).convert('RGB')
    ax.imshow(img)
    ax.set_title(classes[4 + j])
    ax.axis('off')

plt.show()

# 3) Check image size variability
sizes = []
for cls in classes:
    for fname in os.listdir(os.path.join(DATA_DIR, cls)):
        w,h = Image.open(os.path.join(DATA_DIR, cls, fname)).size
        sizes.append((w,h))

ws, hs = zip(*sizes)
plt.figure(figsize=(6,6))
plt.scatter(ws, hs, alpha=0.2)
plt.xlabel('Width (px)')
plt.ylabel('Height (px)')
plt.title('Image Size Distribution')
plt.show()

# 4) Color histogram per class
target_class = 'malachite'
class_dir = os.path.join(DATA_DIR, target_class)
r_hist, g_hist, b_hist = np.zeros(256), np.zeros(256), np.zeros(256)

for fname in os.listdir(class_dir):
    img = Image.open(os.path.join(class_dir, fname)).convert('RGB')
    r, g, b = np.array(img).transpose(2, 0, 1)
    r_hist += np.bincount(r.flatten(), minlength=256)
    g_hist += np.bincount(g.flatten(), minlength=256)
    b_hist += np.bincount(b.flatten(), minlength=256)

plt.figure(figsize=(8, 4))
plt.plot(r_hist, color='red', alpha=0.6, label='Red')
plt.plot(g_hist, color='green', alpha=0.6, label='Green')
plt.plot(b_hist, color='blue', alpha=0.6, label='Blue')
plt.title(f'Color Histogram - {target_class.capitalize()}')
plt.xlabel('Pixel Intensity')
plt.ylabel('Pixel Count')
plt.legend()
plt.tight_layout()
plt.show()