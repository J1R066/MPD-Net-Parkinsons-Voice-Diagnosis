# Results & Robustness Analysis

The MPD-Net (Parallel) was compared against a Sequential baseline (PD-Net) and a Transfer Learning approach (InceptionV3) across three datasets. The results demonstrate the superior robustness of the parallel architecture in handling real-world, noisy data.

## 🏆 Performance Evaluation

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

---

# Data Processing & Feature Extraction

This module is responsible for the end-to-end preparation of audio data for the MPD-Net model. It includes built-in visualization tools to verify data integrity before training.

## Visualization & Verification

### Augmentation Verification
Before training, the system verifies that augmentations do not corrupt the signal structure. The figure below illustrates the effect of data augmentation on a sample waveform from the Italian dataset.

![Augmentation Effect on a Sample Waveform](augmentation_visualization_A_DEFAULT.png)
*Figure 4: The top graph shows the original raw audio signal. The bottom graph displays the same signal after pitch shifting. Note that the temporal structure remains consistent, preserving the diagnostic content.*

### Feature Inspection
To ensure spectral features are computed correctly, the pipeline generates heatmaps for random samples. These 2D representations serve as the parallel inputs for the MPD-Net model.

![Feature Extraction Examples](feature_visualizations_A_DEFAULT.png)
*Figure 5: Feature Extraction. Top row: Mel Spectrograms visualizing energy distribution. Bottom row: MFCCs capturing spectral envelope and vocal tract characteristics.*

## Usage

The data processing script is designed to be executed directly from the command line interface (CLI).

### Command Line Arguments

To run the script, use the following flags:

* **--dataset**: Specifies which dataset to process (`ITALIAN_DATASET`, `UAMS_DATASET`, `MPOWER_DATASET`).
* **--mode**: Specifies the subset of audio files (`A` for vowel /a/, `ALL_VALIDS` for all tasks).
* **--feature_mode**: Specifies the feature set (`DEFAULT` extracts Mel Spectrograms + MFCCs).

### Execution Examples

```bash
# Process the high-quality Italian dataset (Vowel /a/)
python process_data_and_extract_features.py --dataset ITALIAN_DATASET --mode A

# Process the noisy mPower dataset (All voice tasks)
python process_data_and_extract_features.py --dataset MPOWER_DATASET --mode ALL_VALIDS