# MPD-Net: Data Processing & Feature Extraction

This module is responsible for the end-to-end preparation of audio data for the MPD-Net model. It handles raw signal processing, data augmentation, class balancing, and spectral feature extraction.

<img width="896" height="665" alt="image" src="https://github.com/user-attachments/assets/9b2c6855-409d-4dac-a00d-2561f5623207" />


## Visualization & Verification

The pipeline includes built-in visualization functions to verify data integrity and ensure that signal processing steps are performing correctly before the machine learning phase begins.

### Augmentation Verification
Before training, the system verifies that augmentations do not corrupt the signal structure. The figure below illustrates the effect of data augmentation on a sample waveform from the Italian dataset.

The top graph shows the original raw audio signal of a vowel phonation. The bottom graph displays the same signal after applying pitch shifting. Note that the temporal structure and amplitude envelope remain consistent, ensuring the augmentation introduces variability without destroying the diagnostic content of the signal.

![Augmentation Effect on a Sample Waveform](augmentation_visualization_A_DEFAULT.png)

### Feature Inspection
To ensure spectral features are computed correctly, the pipeline generates heatmaps for random samples from both Healthy Control and Parkinson's classes.

The figure below demonstrates the extracted spectral features. The top row displays **Mel Spectrograms**, visualizing energy distribution across frequency bands over time (using the magma colormap). The bottom row shows the corresponding **Mel-Frequency Cepstral Coefficients (MFCCs)**, capturing the spectral envelope and timbral characteristics (using the coolwarm colormap). These 2D representations serve as the parallel inputs for the MPD-Net model.

![Feature Extraction Examples: Mel Spectrogram and MFCC](feature_visualizations_A_DEFAULT.png)

## Usage

The data processing script is designed to be executed directly from the command line interface (CLI). It accepts several arguments that allow the user to customize the dataset selection and processing mode without modifying the codebase.

### Command Line Arguments

To run the script, use the following flags:

* **--dataset**: Specifies which dataset to process.
    * Options: `ITALIAN_DATASET`, `UAMS_DATASET`, `MPOWER_DATASET`.
    * Default: `ITALIAN_DATASET`.

* **--mode**: Specifies the subset of audio files to analyze.
    * `A`: Processes only the sustained vowel /a/ recordings (recommended for standard baseline comparison).
    * `ALL_VALIDS`: Processes all valid voice tasks defined in the configuration (e.g., other vowels or reading tasks).
    * Default: `A`.

* **--feature_mode**: Specifies the type of features to extract.
    * `DEFAULT`: Extracts both Mel Spectrograms and MFCCs and resizes them to fixed dimensions for the hybrid model.
    * Default: `DEFAULT`.

### Execution Examples

1.  **Processing the Italian Dataset:**
    To process the high-quality Italian dataset focusing strictly on the vowel /a/, run the following command:
    
    ```bash
    python process_data_and_extract_features.py --dataset ITALIAN_DATASET --mode A
    ```

2.  **Processing the UAMS Dataset:**
    To process the telephonic quality UAMS dataset:
    
    ```bash
    python process_data_and_extract_features.py --dataset UAMS_DATASET --mode A
    ```

3.  **Processing the mPower Dataset:**
    To process the large-scale, noisy mPower dataset using all valid voice tasks available:
    
    ```bash
    python process_data_and_extract_features.py --dataset MPOWER_DATASET --mode ALL_VALIDS
    ```

### Output Artifacts

Upon successful execution, the script will generate the following outputs in the respective dataset directory:

1.  **manifest.csv**: A mapping of filenames to patient metadata (Age, Sex).
2.  **features_MODE_DEFAULT.npz**: A compressed NumPy file containing the final training tensors:
    * `mel_spectrogram`: The 3D array of spectrogram features.
    * `mfcc`: The 3D array of cepstral features.
    * `labels`: The binary classification labels (0 for Healthy, 1 for Parkinson's).
    * `age` & `sex`: Demographic vectors.
3.  **summary.csv**: A report detailing the number of files processed, dropped, and augmented.
