# Results & Robustness Analysis

The MPD-Net (Parallel) was compared against a Sequential baseline (PD-Net) and a Transfer Learning approach (InceptionV3) across three datasets. The results demonstrate the superior robustness of the parallel architecture in handling real-world, noisy data.

## Performance Evaluation

The model achieved near-perfect classification on the high-quality Italian dataset, demonstrating its capability to learn subtle vocal biomarkers in controlled conditions.

**Italian Dataset (Vowel /a/):**
* **AUC:** 1.00
* **Sensitivity:** 100.00%
* **Specificity:** 100.00%

![Comprehensive Model Evaluation on Italian Dataset](image_47d7f8.png)
*Figure 1: Comprehensive evaluation on the Italian dataset. Left: Confusion Matrix showing perfect classification. Middle: ROC Curve with AUC=1.00. Right: Training history showing rapid convergence of loss and accuracy.*

### Robustness Across Datasets

| Dataset | Quality | MPD-Net (Proposed) AUC | PD-Net (Sequential) AUC | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Italian** | Studio | **1.00** | 0.99 | Near perfect separation in lab conditions. |
| **UAMS** | Phone | **0.871** | 0.649 | MPD-Net retains discriminative power; Sequential degrades. |
| **mPower** | Wild | **0.653** | 0.500 | Sequential model collapsed (random guess); MPD-Net remained robust. |

## 🔍 Interpretability (XAI) Analysis

To validate that the model is learning meaningful biological features rather than noise, we employed advanced Explainable AI techniques.

### 1. SHAP Analysis (Global Feature Importance)
SHAP (SHapley Additive exPlanations) values reveal which specific frequency bands and time steps influenced the model's decision the most.

![SHAP Global and Local Analysis](Screenshot%202025-12-08%20130237.png)
*Figure 2: SHAP Analysis. Left: Top-20 most important global features. Right: Average SHAP heatmaps for Healthy vs. Parkinson's patients, and the Difference Map (PD - Healthy) highlighting the specific spectral regions driving the diagnosis.*

### 2. Grad-CAM (Attention Mapping)
Grad-CAM visualizes the spatial attention of the CNN branch, confirming that the model focuses on relevant vocal segments rather than silence or artifacts.

![Grad-CAM Attention Maps](Screenshot%202025-12-08%20130254.png)
*Figure 3: Grad-CAM Analysis. The heatmaps show the model's attention (red/yellow regions) distributed across the temporal structure of the audio, indicating a comprehensive analysis strategy rather than "tunnel vision" on a single artifact.*
