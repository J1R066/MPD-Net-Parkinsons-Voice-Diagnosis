# MPD-Net-Parkinsons-Voice-Diagnosis
### MPD-Net: Robust Parkinson's Disease Diagnosis via Parallel Voice Analysis

[cite_start]This repository presents the implementation and evaluation of **MPD-Net (Multi-stream Parkinson's Disease Network)**, a hybrid deep learning architecture for the non-invasive screening and diagnosis of Parkinson's Disease (PD) through acoustic analysis of sustained vowels (specifically /a/)[cite: 1619, 1621].

#### **Project Goal**
[cite_start]The core objective is to achieve a superior balance between diagnostic **accuracy**, computational **efficiency**, and **robustness** against the low-quality, noisy data collected in real-world environments (e.g., the mPower dataset)[cite: 299, 1623, 1626].

#### **Key Features**
* [cite_start]**Parallel, Hybrid Architecture (MPD-Net):** Integrates **Convolutional Neural Networks (CNN)**, **Long Short-Term Memory (LSTM)**, and **Multi-Head Attention** mechanisms in parallel streams to robustly process diverse spectral features simultaneously[cite: 867, 869, 1529].
* [cite_start]**Feature Fusion:** Utilizes a combined input of **Mel Spectrograms** and **Mel-Frequency Cepstral Coefficients (MFCCs)** for comprehensive acoustic analysis[cite: 1620, 1621].
* [cite_start]**Enhanced Robustness:** Demonstrated superior performance and gradual degradation on noisy, real-world data (mPower) compared to sequential models, mitigating "tunnel vision" and ensuring greater reliability[cite: 1513, 1514, 1527, 1623].
* [cite_start]**Interpretability Analysis:** Includes advanced explainability techniques (**SHAP** and **Grad-CAM**) to visualize and understand model decisions, identifying the crucial time-frequency patterns used for diagnosis[cite: 881, 996, 1518, 1519].

#### **Performance Highlights**
* [cite_start]Achieved near-perfect performance on high-quality lab data (**AUC $\approx 1.00$ on Italian Dataset**)[cite: 1624].
* [cite_start]Maintained a higher level of diagnostic capability on noisy real-world data (**AUC $= 0.653$ on mPower**) compared to the baseline sequential models, validating its robustness[cite: 1627].
