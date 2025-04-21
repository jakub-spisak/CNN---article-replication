# CNN---article-replication

## Dataset

- **Source**: Public ore image dataset from Kaggle  
- **Size**: 957 images  
- **Classes**: 7 ore/mineral types  
- **Splits**:
  - Training: 60%  
  - Validation: 20%  
  - Test: 20%
- **Site**: https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset

---

## Models

1. **AlexNet**  
2. **VGG16**  
3. **ResNet50**  
4. **InceptionV3**  
5. **MobileNetV2**  
6. **Improved CNN**: MobileNetV2 + Squeeze‑and‑Excitation (SE) block  

---

## Methodology

1. **Transfer Learning**  
   - Initialized all networks with ImageNet‑pretrained weights 
   - Replaced final FC layer for 7‑class output  
   - Fine‑tuned *all* layers using a low learning rate (1×10⁻⁴) and dropout (p=0.5)  

2. **Data Augmentation**  
   - Random horizontal & vertical flips  
   - Random resized crops / scaling (0.8–1.2×)  
   - Small rotations (±20°)  
   - Color jitter (±20% brightness/contrast)  
   - Expanded training set ~5× (4 augmented variants per image)  

3. **SE Attention Module**  
   - Inserted one Squeeze‑and‑Excitation block after the MobileNetV2 feature extractor  
   - Bottleneck FC layers with reduction ratio *r*=8  
   - Channel‑wise reweighting before final classifier  

4. **Training Details**  
   - Optimizer: Adam, lr = 1×10⁻⁴  
   - Epochs: 50  
   - Hardware: NVIDIA GPU workstation  

---

## Results

| Model                    | Test Accuracy | Precision | Recall | F1‑Score |
|--------------------------|--------------:|----------:|-------:|---------:|
| AlexNet                  | 80.0 %        | 0.80      | 0.80   | 0.80     |
| VGG16                    | 83.0 %        | 0.83      | 0.83   | 0.83     |
| ResNet50                 | 87.0 %        | 0.87      | 0.87   | 0.87     |
| InceptionV3              | 85.0 %        | 0.85      | 0.85   | 0.85     |
| MobileNetV2              | 94.0 %        | 0.94      | 0.94   | 0.94     |
| MobileNetV2 + SENet (SE) | 96.9 %        | 0.97      | 0.97   | 0.97     |

- **Key Findings**:  
  - Transfer learning + augmentation dramatically improved all models over scratch training.  
  - MobileNetV2 outperformed heavier architectures on this small dataset.  
  - Adding an SE block to MobileNetV2 boosted accuracy from ~94 % to ~96.9 %.  
  - Learning curves showed rapid convergence (≈95 % validation accuracy by epoch 20).  
  - Confusion matrix analysis confirmed high per‑class performance with only minor off‑diagonal errors.  
