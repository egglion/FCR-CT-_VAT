# FCR-CT-_VAT
Code for the paper “Few-Shot Specific Emitter Identification based on Feature Contrast CNN-Transformer network" 

---

# FCR-CT (VAT) Framework for Few-Shot Specific Emitter Identification

This repository contains the implementation of the **FCR-CT (VAT)** framework proposed in our paper:

**Few-Shot Specific Emitter Identification based on Feature Contrast CNN-Transformer network**

### **Abstract**
Few-shot specific emitter identification (SEI) presents challenges in scenarios with limited labeled data and multi-category classification tasks. Our proposed framework, FCR-CT (Feature Contrastive Reconstruction with CNN-Transformer), incorporates self-supervised learning and semi-supervised learning with virtual adversarial training (VAT). This method optimizes feature extraction and enhances robustness in low-data environments. For more details, please refer to the full paper.

---

### **Features**
1. **Two-Stage Learning**:
   - **Self-Supervised Pretraining**: Uses a CNN-Transformer encoder-decoder to reconstruct input signals and optimize feature space representation.
   - **Semi-Supervised Fine-Tuning**: Incorporates VAT to refine feature boundaries and improve classification accuracy.

2. **Feature Contrastive Loss**:
   - Aligns CNN and Transformer outputs to enhance intra-class compactness and inter-class separability.

3. **Robust Classification**:
   - Leverages labeled and unlabeled data to improve model robustness and mitigate classification challenges.

---
# FCR-CT (VAT) Framework

## Model Architecture
The architecture of the proposed FCR-CT (VAT) framework is shown below:

![Model Architecture](fig3.fig)
### **Repository Structure**
```
├── data/                 # Datasets (ADS-B signals)
├── base_models.py        # 小零件1
├── MAE_T.py                # Masked CNN-Transformer encoder-decoder
├── mymain_vat.py              # Training and testing script for code
├── subsss.py               # 小零件2
├── Visualization.py      # 验证
├── README.md             # Project documentation
```

---
### **Results**

The following table summarizes the performance of FCR-CT (VAT) compared to existing methods on the ADS-B dataset under 20-shot conditions (20 labeled samples per category):

| Method         | 10 Categories (%) | 20 Categories (%) | 30 Categories (%) |
|----------------|--------------------|--------------------|--------------------|
| CVCNN          | 90.12             | 71.50             | 62.44             |
| SA-CNN         | 88.32             | 70.25             | 56.03             |
| CNN-MAT        | 89.75             | 72.00             | 62.44             |
| **FCR-CT (VAT)** | **90.52**         | **74.65**         | **68.65**         |

---


### **Contact**
For any questions or issues, please open an issue or contact the author at 1208441627@qq.com.

---

