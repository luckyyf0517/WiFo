Here is the comprehensive training reproduction documentation for the WiFo model, based on the provided research paper.

***

# Training Manual: WiFo (Wireless Foundation Model)

**Subject:** WiFo Pre-training and Reproduction Guidelines
**Based on:** Liu et al., "WiFo: wireless foundation model for channel prediction" (Science China Information Sciences, 2025)
**Scope:** Data generation, preprocessing, hyperparameter configuration, and pre-training workflow.

---

## 1. Environment Requirements

To reproduce the training results reported in the paper, the following computational environment is recommended.

### 1.1 Hardware Specifications
*   **GPU:** 4 $\times$ NVIDIA GeForce RTX 4090 (or equivalent with TF32 support).
*   **CPU:** AMD EPYC 7763 64-Core Processor (or equivalent high-performance CPU).
*   **RAM:** 256 GB.

### 1.2 Software Dependencies
*   **Precision:** TensorFloat-32 (TF32) must be enabled for matrix multiplications.
*   **Channel Generator:** QuaDRiGa (QUAsi Deterministic RadIo channel GenerAtor) compliant with 3GPP standards.

---

## 2. Data Preparation

The WiFo model requires a large-scale heterogeneous dataset generated via simulation.

### 2.1 Dataset Generation (QuaDRiGa)
Generate 16 distinct datasets (indexed D1 through D16) for the pre-training phase. Each dataset must adhere to the specific system configurations outlined below to ensure diversity in carrier frequency, bandwidth, and antenna geometry.

**Global Settings:**
*   **System:** MISO-OFDM.
*   **BS Antenna:** Uniform Planar Array (UPA).
*   **User Antenna:** Single antenna.
*   **Antenna Spacing:** Half-wavelength at central frequency.
*   **Trajectory:** Linear motion with random initial positions.

**Configuration Table (D1–D16):**

| Dataset | $f_c$ (GHz) | Sub-carriers ($K$) | Time Slots ($T$) | UPA ($N_h \times N_v$) | Scenario | Speed (km/h) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| D1 | 1.5 | 128 | 24 | $1 \times 4$ | UMi+NLoS | 3–50 |
| D2 | 1.5 | 128 | 24 | $2 \times 4$ | RMa+NLoS | 120–300 |
| D3 | 1.5 | 64 | 16 | $1 \times 8$ | Indoor+LoS | 0–10 |
| D4 | 1.5 | 32 | 16 | $4 \times 8$ | UMa+LoS | 30–100 |
| D5 | 2.5 | 64 | 24 | $2 \times 2$ | RMa+NLoS | 120–300 |
| D6 | 2.5 | 128 | 24 | $2 \times 4$ | UMi+LoS | 3–50 |
| D7 | 2.5 | 32 | 16 | $4 \times 8$ | UMa+LoS | 30–100 |
| D8 | 2.5 | 64 | 16 | $4 \times 4$ | Indoor+NLoS | 0–10 |
| D9 | 4.9 | 128 | 24 | $1 \times 4$ | UMi+NLoS | 3–50 |
| D10 | 4.9 | 64 | 24 | $2 \times 4$ | RMa+LoS | 120–300 |
| D11 | 4.9 | 64 | 16 | $4 \times 4$ | UMa+NLoS | 30–100 |
| D12 | 4.9 | 32 | 16 | $4 \times 8$ | Indoor+LoS | 0–10 |
| D13 | 5.9 | 64 | 24 | $2 \times 8$ | RMa+LoS | 120–300 |
| D14 | 5.9 | 128 | 24 | $2 \times 4$ | UMi+NLoS | 3–50 |
| D15 | 5.9 | 64 | 16 | $4 \times 4$ | Indoor+LoS | 0–10 |
| D16 | 5.9 | 32 | 16 | $4 \times 8$ | UMa+NLoS | 30–100 |

### 2.2 Data Splitting and Augmentation
*   **Sample Count:** Generate 12,000 samples per dataset.
*   **Split Ratio:**
    *   Training: 9,000 samples.
    *   Validation: 1,000 samples.
    *   Inference (Test): 2,000 samples.
*   **Total Training Set:** 160,000 samples (aggregated from D1–D16).
*   **Noise Injection:** Add complex Gaussian noise with a Signal-to-Noise Ratio (SNR) of **20 dB** to all samples during training and inference.

### 2.3 Preprocessing Pipeline
1.  **Standardization:** Normalize CSI samples using the mean and variance calculated per dataset.
2.  **Real-Value Conversion:** Convert complex CSI $H \in \mathbb{C}^{T \times K \times N}$ into a real-valued tensor $\tilde{H} \in \mathbb{R}^{2 \times T \times K \times N}$ (channels represent real and imaginary parts).
3.  **3D Patching:** Apply 3D convolution to segment the tensor into non-overlapping patches.
    *   **Patch Size:** $(t, k, n) = (4, 4, 4)$.

---

## 3. Training Configuration

### 3.1 Model Architecture (WiFo-Base)
Ensure the model is instantiated with the following parameters for the "Base" conﬁguration:
*   **Encoder:** Depth=6, Width=512, Heads=8.
*   **Decoder:** Depth=4, Width=512, Heads=8.
*   **Positional Encoding:** Space-Time-Frequency Positional Encoding (STF-PE) using absolute SinCos functions.

### 3.2 Hyperparameters
Configure the training loop with the parameters defined in Table 3 of the reference paper.

| Parameter | Value |
| :--- | :--- |
| **Optimizer** | AdamW |
| **Optimizer Betas** | $\beta_1 = 0.9$, $\beta_2 = 0.999$ |
| **Weight Decay** | 0.05 |
| **Batch Size** | 128 |
| **Total Epochs** | 200 |
| **Base Learning Rate** | $5 \times 10^{-4}$ |
| **LR Schedule** | Cosine Decay |
| **Warmup Epochs** | 5 |

---

## 4. Pre-training Workflow

The pre-training phase utilizes a self-supervised Masked Autoencoder (MAE) approach. The model must be trained to reconstruct masked CSI tokens.

### 4.1 Batch Processing Logic
For every batch of data during the training loop, the data must be shuffled. The model is trained on **three distinct reconstruction tasks** sequentially.

### 4.2 Masking Strategies (Tasks)
Apply the following masking strategies to the input tokens. The loss is calculated for the reconstructed parts only.

**1. Random-Masked Reconstruction**
*   **Objective:** Capture 3D structured features.
*   **Method:** Randomly mask tokens isotropically across Space-Time-Frequency dimensions.
*   **Masking Ratio ($R_r$):** 85%.

**2. Time-Masked Reconstruction**
*   **Objective:** Learn causal relationships over time (Time-domain prediction).
*   **Method:** Mask all tokens where the time coordinate $pos_t \ge (T/t - \lfloor R_t \cdot T/t \rfloor)$. Effectively masks the "future" steps.
*   **Masking Ratio ($R_t$):** 50%.

**3. Frequency-Masked Reconstruction**
*   **Objective:** Learn variations between adjacent frequency bands (Frequency-domain prediction).
*   **Method:** Mask all tokens where the frequency coordinate $pos_f \ge (K/k - \lfloor R_f \cdot K/k \rfloor)$. Effectively masks the "upper" frequency bands.
*   **Masking Ratio ($R_f$):** 50%.

### 4.3 Loss Calculation
*   **Metric:** Mean Squared Error (MSE).
*   **Formula:** $L = \frac{1}{|\omega|} \| H[\omega] - H_{pred}[\omega] \|_F^2$, where $\omega$ represents the subset of masked elements.
*   **Optimization:** Compute the gradient based on the mean loss of the three reconstruction tasks for the current batch.

---

## 5. Output and Checkpointing

*   **Model Weights:** Save the encoder and decoder weights.
*   **Zero-Shot Readiness:** The resulting pre-trained model (WiFo) requires no fine-tuning for downstream tasks. It can be directly applied to inference on unseen datasets (e.g., D17–D19) by applying the specific masking strategy (Time or Frequency) corresponding to the desired prediction task.