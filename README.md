# MPD-Net: Robust Parkinson's Disease Diagnosis via Parallel Voice Analysis

## Abstract
This repository houses the implementation of **MPD-Net (Multi-stream Parkinson's Disease Network)**, a parallel hybrid deep learning architecture designed for the non-invasive screening of Parkinson's Disease (PD) using voice analysis. While deep learning has shown promise in controlled environments, performance often degrades on real-world data. This project addresses this gap by proposing a parallel architecture that integrates **CNNs**, **LSTMs**, and **Multi-Head Attention** to process spectral features simultaneously.

The study validates that parallel processing offers superior robustness against noise compared to sequential models, mitigating the "tunnel vision" phenomenon often observed in deep learning models trained on noisy acoustic data.

## Table of Contents
- [Abstract](#abstract)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results & Robustness Analysis](#results--robustness-analysis)
- [Interpretability & XAI Analysis](#interpretability--xai-analysis)
- [Overall Conclusion](#overall-conclusion)
- [Discussion](#discussion)
- [Future Directions](#future-directions)
- [Citation](#citation)
- 
## Datasets
This study evaluates model performance across three distinct datasets representing a spectrum of audio quality, from studio-grade to real-world "wild" data.

1. **Italian Parkinson's Voice and Speech:** High-quality, studio-recorded data (495 PD, 336 HC). Used as the controlled baseline.
2. **UAMS (University of Arkansas for Medical Sciences):** Telephonic quality recordings with limited bandwidth (40 PD, 41 HC). Represents moderate noise conditions.
3. **mPower (Sage Bionetworks):** Crowdsourced smartphone recordings collected via iPhone app.
    * **Selection Criteria:** To minimize confounding factors and ensure data consistency, the raw dataset was filtered to include only subjects aged **50 to 70 years** who did not have other medical conditions affecting voice. Additionally, recordings with excessive environmental noise or artifacts were manually excluded.
    * **Final Count:** This process resulted in a challenging test set of **188 PD patients and 210 Healthy Controls (HC)**, representing high-noise, uncontrolled real-world environments.

Note: Due to privacy agreements, this repository contains the processing code. You must request access to the raw data from the respective owners.

## Methodology

### 1. Pre-processing & Feature Extraction
The pipeline processes raw audio (sustained vowel /a/) into dual spectral representations.

* **Signal Processing:** 16kHz sampling, segmented into 30ms frames, normalized, and padded to 3 seconds.
* **Data Augmentation:** To prevent overfitting, signals undergo pitch shifting (+/- 2 semitones), gain adjustment, and white noise injection (SNR 10-30 dB).
* **Features:**
    * **Mel Spectrogram:** (30 bands x 94 frames) capturing energy distribution.
    * **MFCCs:** (30 coefficients) capturing vocal tract properties.

### 2. MPD-Net Architecture
Unlike sequential models (e.g., CNN -> LSTM), MPD-Net processes the input (194 x 60 combined features) through three parallel streams to capture diverse acoustic characteristics simultaneously.

* **Stream 1 (Spatial):** CNN blocks to extract local spectral patterns.
* **Stream 2 (Temporal):** LSTM layers to model long-term dependencies and tremors.
* **Stream 3 (Focus):** Multi-Head Attention to highlight critical signal segments.
* **Fusion:** Outputs are concatenated and passed through a Dense layer (128 units) for binary classification.

## Results & Robustness Analysis

The MPD-Net (Parallel) was compared against a Sequential baseline (PD-Net) and a Transfer Learning approach (InceptionV3).

| Dataset | Quality | MPD-Net (Proposed) AUC | PD-Net (Sequential) AUC | Observation |
| :--- | :--- | :--- | :--- | :--- |
| **Italian** | Studio | **1.00** | 0.99 | Near perfect separation in lab conditions. |
| **UAMS** | Phone | **0.871** | 0.649 | MPD-Net retains discriminative power; Sequential degrades. |
| **mPower** | Wild | **0.653** | 0.500 | Sequential model collapsed (random guess); MPD-Net remained robust. |

## Interpretability & XAI Analysis

To ensure the model's decisions are transparent and to diagnose the root causes of performance degradation on real-world data, this study employed two complementary Explainable AI (XAI) techniques. These methods revealed that model failure on noisy data is often due to a "Tunnel Vision" phenomenon rather than a lack of model capacity.

### 1. SHAP (SHapley Additive exPlanations)
SHAP analysis was used to quantify the contribution of individual input features (specific time-frequency points in the spectrogram) to the model's final diagnosis.

* **Methodology:** We computed Global SHAP values to rank feature importance and generated "Difference Maps" by subtracting the average healthy attention map from the average Parkinson's attention map. This highlighted exactly which patterns drove the model to classify a sample as pathological.
* **Distributed Strategy (Clean Data):** On the high-quality Italian dataset, SHAP values were distributed across the entire duration of the signal. The model relied on a complex combination of features from both Mel Spectrograms and MFCCs, indicating it learned robust, biological representations of the voice.
* **Shortcut Learning (Noisy Data):** On the mPower dataset, SHAP analysis revealed that feature importance collapsed onto a small cluster of features at the very beginning of the recording. This indicated the model was bypassing the actual voice signal to rely on a "shortcut" or artifact.

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)
Grad-CAM was applied to the Convolutional Neural Network (CNN) stream of the MPD-Net to visualize the spatial attention of the model within the 2D input matrix.

* **Methodology:** Heatmaps were generated from the final convolutional layer to visualize which regions of the input image triggered the highest activation.
* **Spectral Signatures (Clean Data):** On clean data, the attention maps displayed distinct vertical bands distributed throughout the audio sample. These correspond to specific spectral signatures (such as formants and harmonic deviations) that are clinically relevant to Parkinson's dysphonia.
* **Tunnel Vision (Noisy Data):** On the noisy mPower data, Grad-CAM visually confirmed the "Tunnel Vision" phenomenon. The model's attention was restricted to a tiny, high-intensity spot at the absolute start of the spectrogram—likely a recording artifact (e.g., a microphone click or silence) rather than the vocal phonation.

## Overall Conclusion
This research demonstrates that model architecture plays a decisive role in the reliability of AI-based medical diagnostics. While both parallel (MPD-Net) and sequential (PD-Net) architectures can achieve near-perfect accuracy in controlled laboratory settings, their behaviors diverge significantly in real-world environments.

The parallel MPD-Net architecture proved to be superior in terms of robustness and generalization. By processing spatial, temporal, and attentional features simultaneously, it maintained a distributed representation of the input signal even in the presence of heavy noise. In contrast, sequential architectures exhibited fragility, often collapsing into trivial solutions (predicting a single class) or overfitting to recording artifacts. Additionally, the re-evaluation of pre-trained models highlighted the critical importance of rigorous validation protocols; improper data splitting can lead to drastically inflated performance metrics that do not hold up in practice.

## Discussion
The disparity between the results on the Italian dataset (AUC 1.00) and the mPower dataset (AUC 0.653) underscores the "reality gap" currently facing AI in voice analysis.

* **Sensitivity vs. Specificity:** In clean conditions, the parallel model favored high sensitivity (identifying almost all patients), making it suitable for screening. The sequential model favored high specificity (few false alarms), making it suitable for confirmation. However, this distinction vanished in noisy conditions where the sequential model failed.
* **The Role of XAI:** The application of SHAP and Grad-CAM moved beyond simple visualization to become a core debugging tool. It revealed that the performance drop in "wild" data was not merely due to noise masking the signal, but due to the model actively learning to focus on invalid shortcuts (Tunnel Vision). This insight is invaluable for future feature engineering.
* **Architecture robustness:** The parallel design of MPD-Net likely prevents error propagation that occurs in sequential models. In a sequential CNN-LSTM, if the CNN fails to extract clean features from noise, the LSTM receives corrupted data. In a parallel design, the LSTM and Attention heads operate on the raw features independently, providing a form of redundancy that preserves diagnostic signal.

## Future Directions
To bridge the gap between current research and clinical application, future work should focus on the following areas:

1.  **Online Screening Platform:** Development of a web or mobile-based application to collect a wider variety of voice samples. This would not only serve as a screening tool but also generate the large-scale, diverse datasets needed to train the next generation of robust models.
2.  **Advanced Domain Adaptation:** Implementing techniques to specifically address the domain shift between studio recordings and smartphone data. This could include adversarial training to make the model invariant to recording device characteristics.
3.  **Privacy-Preserving Learning:** As data collection scales, implementing federated learning or other privacy-preserving techniques to ensure patient data remains secure while allowing for model improvements.
4.  **Noise-Robust Feature Engineering:** Investigating feature extraction methods that are inherently less sensitive to environmental noise and recording artifacts compared to standard Mel Spectrograms.

## Citation
If you use this code for your research, please cite the repository:

> **Baharvand, F.** (2025). *MPD-Net-Parkinsons-Voice-Diagnosis* . GitHub. https://github.com/baharvand79/MPD-Net-Parkinsons-Voice-Diagnosis

Or use the BibTeX entry:

```bibtex
@misc{baharvand2025mpdnet,
  author = {Baharvand, Fatemeh},
  title = {MPD-Net-Parkinsons-Voice-Diagnosis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/baharvand79/MPD-Net-Parkinsons-Voice-Diagnosis](https://github.com/baharvand79/MPD-Net-Parkinsons-Voice-Diagnosis)}}
}
```
