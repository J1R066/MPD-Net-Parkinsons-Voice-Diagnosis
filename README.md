# MPD-Net-Parkinsons-Voice-Diagnosis
### MPD-Net: Robust Parkinson's Disease Diagnosis via Parallel Voice Analysis

This repository presents the implementation and evaluation of **MPD-Net (Multi-stream Parkinson's Disease Network)**, a hybrid deep learning architecture for the non-invasive screening and diagnosis of Parkinson's Disease (PD) through acoustic analysis of sustained vowels (specifically /a/)

#### **Project Goal**
The core objective is to achieve a superior balance between diagnostic **accuracy**, computational **efficiency**, and **robustness** against the low-quality, noisy data collected in real-world environments (e.g., the mPower dataset).

#### **Key Features**
* **Parallel, Hybrid Architecture (MPD-Net):** Integrates **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)**, and **Multi-Head Attention** mechanisms in parallel streams to robustly process diverse spectral features simultaneously.
* **Feature Fusion:** Utilizes a combined input of **Mel Spectrograms** and **Mel-Frequency Cepstral Coefficients (MFCCs)** for comprehensive acoustic analysis.
* **Enhanced Robustness:** Demonstrated superior performance and gradual degradation on noisy, real-world data (mPower) compared to sequential models, mitigating "tunnel vision" and ensuring greater reliability.
* **Interpretability Analysis:** Includes advanced explainability techniques (**SHAP** and **Grad-CAM**) to visualize and understand model decisions, identifying the crucial time-frequency patterns used for diagnosis.

#### **Performance Highlights**
* Achieved near-perfect performance on high-quality lab data (**AUC $\approx 1.00$ on Italian Dataset**).
* Maintained a higher level of diagnostic capability on noisy real-world data (**AUC $= 0.653$ on mPower**) compared to the baseline sequential models, validating its robustness.

## 🎧 Pre-processing and Data Augmentation

The goal of pre-processing is to transform variable-quality raw audio files into uniform feature representations suitable for the hybrid deep learning architecture.

---

### 1. Signal Preparation
* **Input Signal:** Raw audio signals, each approximately 3 seconds long, were used with a **16 kHz sampling rate**.
* **Segmentation:** Continuous signals were divided into **30-millisecond frames** for short-term analysis.
* **Uniform Length:** Signal lengths were made uniform using **padding** (adding zeros) or **trimming** to maintain a consistent 3-second duration. For the mPower dataset, silence at the beginning and end of the audio was removed.
* **Amplitude Normalization:** Signal amplitudes were scaled to a uniform range to minimize variations caused by recording intensity.

### 2. Data Augmentation and Balancing
To increase the dataset's size and variety and to prevent **overfitting**, the following data augmentation techniques were applied to the raw audio signals, generating three new versions for every original file:
* **Pitch Shifting:** The voice pitch was slightly altered (by $\pm 2$ semitones, coefficient 1.5) to simulate the variations expected across different speakers and make the model more resistant to minor pitch differences.
* **Gain Augmentation:** The volume or loudness was randomly adjusted ($\pm 10\%$) to account for differences in microphone distance or speaking intensity.
* **White Noise Addition:** White noise was added with a Signal-to-Noise Ratio (**SNR**) between $10-30 \text{ dB}$ to enhance the model's resistance to environmental noise.

Additionally, **Class Balancing** was performed by randomly copying samples from the minority class (either healthy or patient) to match the number of samples in the majority class, which resulted in a final balanced set of **3496 samples**.

---

## Feature Description

The deep learning models rely on a fusion of two powerful and complementary spectral features: **Mel Spectrograms** and **Mel-Frequency Cepstral Coefficients (MFCCs)**.

### 1. Mel Spectrogram (Mel-Spectrogram)

The Mel Spectrogram is a **two-dimensional visual representation** of the audio signal's energy distribution across different frequencies over time. It is widely used because its frequency axis is scaled according to the **Mel scale**, which aligns more closely with human auditory perception.

The process involves:
* **Short-time Fourier Transform (STFT):** The signal is broken down into short frames (30 ms with 10 ms overlap), and the frequency spectrum is calculated for each frame.
* **Mel Scale Mapping:** The resulting linear frequency spectrum is filtered using **30 Mel filter banks**.
* **Output:** The final output is a 2D matrix of **$30 \times 94$ dimensions** (number of Mel filters $\times$ number of time frames) representing the evolution of energy in Mel bands over time.

### 2. Mel-Frequency Cepstral Coefficients (MFCCs)

MFCCs are a **compact, highly efficient, and perceptually-based representation** of the spectral characteristics of the human vocal tract. They are particularly effective at capturing changes in the acoustic properties of the vocal tract (formants) that are affected by speech disorders in PD.

The process involves:
* **Pre-emphasis:** A filter is applied to the raw signal to boost higher frequencies and compensate for the natural attenuation of the vocal tract.
* **Mel Filter Bank and Log Power Spectrum:** The spectrum is mapped to the Mel scale, and the logarithm of the power is taken to compress the dynamic range and increase feature stability.
* **Discrete Cosine Transform (DCT):** A DCT is applied to the log Mel power spectrum to decorrelate the Mel bands and produce the final Cepstral Coefficients.
* **Output:** The result is a 2D matrix with **$30 \times 94$ dimensions**, containing the key information about speech features like vowels and consonants.

### 3. Feature Fusion

Both the Mel Spectrograms and the MFCCs are **concatenated** and fed simultaneously to the parallel streams of the MPD-Net model. The combined feature input has a final shape of **$194 \times 60$**. Additional **metadata** such as the subject's age and gender is also extracted and included in the feature list.

## MPD-Net Architecture Overview

The **Multi-stream Parkinson's Disease Network (MPD-Net)** is a hybrid and parallel deep learning architecture designed for robust diagnosis of Parkinson's Disease (PD) from vocal features. It integrates three specialized feature processing streams that operate simultaneously to enhance model stability and feature diversity against real-world noise.

---

### 1. Architectural Design (Parallel Hybrid Model)

The architecture takes a combined 2D spectral feature map (Mel Spectrograms + MFCCs) and splits it into three parallel paths before the final classification layers.

| Component | Architecture/Layers | Role in Feature Extraction |
| :--- | :--- | :--- |
| **Input** | Fused **Mel Spectrogram** and **MFCCs** (2D matrix) | Provides rich time-frequency information for simultaneous analysis across domains. |
| **1. CNN Stream** | Two Conv2D blocks (32 and 64 filters) + MaxPooling2D | Extracts **spatial patterns** and **local features** (e.g., changes in frequency intensity over small time intervals). |
| **2. LSTM Stream** | Two LSTM layers (64 and 32 units) + Dropout | Captures **temporal dependencies** and **long-term dynamics** over the signal sequence (e.g., progressive changes in articulation or sustained vocal instability). |
| **3. Multi-Head Attention Stream** | Multi-Head Attention layer | Focuses the model on the **most critical time-frequency regions** that carry the highest diagnostic information (e.g., specific segments with tremor characteristics). |
| **Fusion** | Concatenation (Haq) | Combines the learned representations from the three parallel, specialized feature extractors. |
| **Classifier Head** | Dense Layer (128 units, ReLU) $\rightarrow$ Output Layer (1 unit, Sigmoid) | A bottleneck layer followed by binary classification to predict the final outcome ($\in \{Healthy, Parkinson's\}$). |

---

### 2. Training and Optimization Parameters

The MPD-Net was optimized to reduce computational complexity and enhance generalization. The reduction in the dense layer size compared to the reference model contributed to a significant reduction in the total parameter count.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Parameters** | $\approx 700,000$ | Represents a $\sim 69\%$ reduction compared to the reference PD-Net. |
| **Optimizer** | Adam | Adaptive Moment Estimation algorithm for efficient training. |
| **Loss Function** | Binary Cross-Entropy | Standard loss function for binary classification problems. |
| **Epochs / Batch Size** | 30 / 32 | The number of training iterations and samples processed per parameter update. |
| **Regularization** | Dropout (0.5) and L2 (0.001) | Used to prevent **overfitting** and encourage the learning of robust, generalized features. |


## 🏆 Performance Evaluation and Comparative Analysis

The **MPD-Net** model was comprehensively evaluated across multiple datasets to test both its **accuracy** on controlled data and its **robustness** on challenging, real-world data. A comparison with the sequential **PD-Net** model and a critical re-evaluation of the **Pre-trained** model are summarized below.

---

### 1. Evaluation of Pre-trained Model (Re-evaluation)

The initial high performance reported for the Convolutional Neural Network (CNN) based **Pre-trained** model was found to be misleading due to a methodological flaw (failure to strictly separate training and validation data), leading to **severe overfitting**.

* **UAMS Dataset:** When the Pre-trained model was correctly re-evaluated with strict data separation, its average **AUC score dropped dramatically** from $\approx 0.97$ (overfitted result) to **$\approx 0.76$**.
* **mPower Dataset (Real-World):** The performance collapse was even more pronounced on the noisy mPower dataset. The average **AUC dropped** from $\approx 0.91$ (overfitted result) to **$\approx 0.596$**, which is near random chance.

This re-evaluation confirmed that **accurate validation protocols are critical** for reliable reporting of deep learning model capability.

---

### 2. MPD-Net vs. PD-Net (Parallel vs. Sequential Architectures)

The core finding of this study is the superior **robustness** of the parallel MPD-Net architecture when faced with degraded or noisy data.

#### **Controlled Conditions (Italian Dataset)**

On the high-quality, controlled Italian dataset, both models achieved excellent results, but demonstrated different strengths:

* **MPD-Net (Parallel):** Achieved **AUC $\approx 1.00$** and demonstrated perfect **Sensitivity ($\mathbf{100\%}$)** for the vowel /a/. This makes it an **ideal screening tool** where missing a patient is unacceptable.
    * *Feature Space:* Showed superior feature learning, increasing the separability index **d-prime to $\mathbf{180.2}$** for the vowel /a/, indicating complete class separation.
* **PD-Net (Sequential):** Achieved **AUC $\approx 0.998$** and achieved perfect **Specificity ($\mathbf{100\%}$)** for the vowel /a/, producing **zero False Positives**. This makes it better suited for **diagnostic confirmation** where avoiding false alarms is critical.

#### **Challenging Conditions (UAMS & mPower Datasets)**

When tested on challenging data with varying quality and noise, the fundamental difference in robustness became clear:

| Model | UAMS Dataset (Telephonic Data) | mPower Dataset (Real-World Noise) |
| :--- | :--- | :--- |
| **MPD-Net (Parallel)** | **AUC $\mathbf{= 0.871}$**. **Specificity $\mathbf{100\%}$**. | **AUC $\mathbf{= 0.653}$**. Demonstrated a **gradual degradation** and maintained performance slightly better than random chance. |
| **PD-Net (Sequential)** | **AUC $\mathbf{= 0.649}$**. **Specificity $69.70\%$**. | **AUC $\mathbf{= 0.500}$**. Suffered a **catastrophic failure** (model collapse). The model achieved **Specificity of $\mathbf{0\%}$** by classifying every sample as "Parkinson's". |

The results prove that the **MPD-Net's parallel architecture** is inherently **more resilient** (robust) to noise and variations in data quality than the sequential PD-Net.

---

### 3. Interpretability Analysis (SHAP & Grad-CAM)

Advanced interpretability techniques were used to diagnose the cause of performance degradation in the MPD-Net:

* **On Italian Data (Clean):** The model used a **comprehensive and distributed strategy**, paying attention to subtle, long-term patterns across the entire signal for both Mel Spectrograms and MFCCs. This indicates meaningful learning.
* **On UAMS Data (Medium Noise):** The model began to show a tendency for "tunnel vision" by concentrating its attention on a **specific, small area** at the beginning of the signal instead of analyzing the entire voice sample.
* **On mPower Data (High Noise):** The model suffered from extreme **"Tunnel Vision"**. Instead of performing a comprehensive analysis, its attention collapsed onto a single, **small, and highly concentrated area** (usually the very beginning of the signal in the low-frequency Mel Spectrogram region). The model relied almost entirely on this potentially invalid segment for its decision, which directly explains its poor performance and inability to generalize on real-world data.

## 🔍 Interpretability Methods for MPD-Net

To understand how the **MPD-Net** model reaches a diagnostic decision and to diagnose the causes of performance degradation on noisy data, this research employed two advanced Explainable AI (**XAI**) techniques: **SHAP** (SHapley Additive exPlanations) and **Grad-CAM** (Gradient-weighted Class Activation Mapping).

---

### 1. SHAP (SHapley Additive exPlanations)

SHAP is a method that focuses on the **feature level** to quantify the impact of each input feature on the model’s final prediction.

* **Function:** It calculates the average marginal contribution of a feature value across all possible combinations of features.
* **Application in Study:** SHAP was used on the complex, matrix-based input (**Mel Spectrograms** and **MFCCs**) to determine which specific frequency bands (features) at which points in time were most important for the diagnosis.
* **Key Insight:** The SHAP difference maps (Parkinson's minus Healthy) clearly highlighted the **shift in the model's decision strategy** when moving from clean data to noisy data.

### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM is a visualization technique that focuses on the **spatial level** of the input by producing heatmaps.

* **Function:** It highlights the regions within the input image or 2D feature map (the Mel Spectrogram/MFCC matrix) that caused the highest activation in the final convolutional layer, meaning those regions contributed the most to the model's output.
* **Application in Study:** Grad-CAM provided a **visual confirmation** of the model's "gaze" or "attention". This was applied to the output of the CNN stream to visually check if the model was looking at the entire sound or just small segments.
* **Key Insight:** Grad-CAM visually confirmed the phenomenon of **"Tunnel Vision"** by showing the attention collapsing onto a tiny, highly concentrated region at the start of the mPower signals.

---

### Comparison of Methods

The two methods are complementary:
* **SHAP** provides a **quantitative and numerical** rank of feature importance.
* **Grad-CAM** provides a **qualitative, visual map** of the influential regions within the 2D feature input.

Together, they enabled a deep understanding of why the model succeeded on clean data (distributed attention) and failed on noisy data (collapsed attention).

