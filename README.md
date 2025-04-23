# CNN---article-replication

## Dataset

- **Source**: Public ore image dataset from Kaggle  
- **Size**: 957 images  
- **Classes**: 7 ore/mineral types (e.g., Malachite, Biotite, etc.)  
- **Splits**:
  - Training: 60%  
  - Validation: 20%  
  - Test: 20%  
- **Note**: Moderate class imbalance observed (e.g., Malachite: 235, Biotite: 68)  
- **Site**: [Kaggle Dataset](https://www.kaggle.com/datasets/asiedubrempong/minerals-identification-dataset)

---

## Models

1. **AlexNet**  
2. **VGG16**  
3. **ResNet50**  
4. **InceptionV3**  
5. **MobileNetV2** *(used instead of MobileNetV1 due to availability in PyTorch)*  
6. **Improved CNN**: MobileNetV2 + Squeeze-and-Excitation (SE) block  

---

## Methodology

1. **Transfer Learning (TL)**  
   - Used ImageNet-pretrained weights  
   - Final FC layer replaced with 7-class classifier  
   - Fine-tuned all layers with a low learning rate (1×10⁻⁴)  

2. **Data Augmentation (DA)**  
   - Techniques: center & edge crops, zoom (0.8–1.2×), color jitter (±20%), etc.  
   - Each training image expanded into 5 versions  
   - Improved robustness against lighting, background, and scale variance  

3. **SE Attention Module**  
   - SE block applied to MobileNetV2 output  
   - Reduction ratio *r* = 8  
   - Recalibrated channel-wise features before final classification  

4. **Training Protocol**  
   - Optimizer: Adam  
   - Epochs: 20/50  
   - Dropout: p = 0.5  
   - Image normalization based on dataset-wide mean and std  
   - Backbone freezing improved SE module stability  

---

## Results

| Model                    | Test Accuracy |
|--------------------------|--------------:|
| AlexNet (TL + DA)        | 85.9 %        |
| VGG16 (TL + DA)          | 83.2 %        |
| ResNet50 (TL + DA)       | 88.0 %        |
| InceptionV3 (TL + DA)    | 88.0 %        |
| MobileNetV2 (TL + DA)    | 90.6 %        |
| MobileNetV2 + SE (Frozen)| 93.2 %        |
| MobileNetV2 + SE (Full FT)| 90.1 %       |

- **Key Findings**:  
  - TL + DA significantly improved generalization and stability  
  - MobileNetV2 achieved top performance among base models  
  - SE block added ~2.6% accuracy boost over MobileNetV2 alone  
  - Freezing MobileNetV2 backbone for initial epochs enhanced SE performance  
  - Confusion matrices show strong class-level precision and low inter-class confusion  

---

## Notes and Extensions

- The original study used **MobileNetV1** with ~96.89% reported accuracy; this work used **MobileNetV2** (PyTorch)  
- Class imbalance and lighting variability required aggressive augmentation  
- Future work includes trying CBAM attention, model ensembling, and AutoAugment  
