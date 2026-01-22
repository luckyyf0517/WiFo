# Project Context

## Purpose
WiFo (Wireless Foundation Model) is a Masked Autoencoder (MAE) based neural network designed for Space-Time-Frequency (STF) channel reconstruction in wireless communications. The project aims to:

- Predict wireless Channel State Information (CSI) using a foundation model approach
- Treat channel prediction (both time and frequency domains) as a unified "image" reconstruction problem
- Enable zero-shot generalization to unseen datasets without fine-tuning

**Publication**: "WiFo: Wireless Foundation Model for Channel Prediction" - SCIENCE CHINA Information Sciences, June 2025

## Tech Stack

**Core Languages & Frameworks:**
- Python 3.9+ (minimum requirement)
- PyTorch 2.4.1 (main ML framework)
- NumPy 1.26.4 for numerical computations
- Scikit-learn 1.3.2 for utilities
- TensorBoard 2.20.0 for training visualization

**ML Components:**
- TimM 0.9.2 for vision transformer components
- Einops for tensor manipulation

**Data Handling:**
- H5Py 3.14.0 & HDF5Storage 0.1.19 for .mat file handling

**System Utilities:**
- Setproctitle for process naming

**Hardware Requirements:**
- NVIDIA GPU + CUDA (tested with CUDA 12.1)
- Recommended: 4 × NVIDIA GeForce RTX 4090 for full training
- 256GB RAM for large-scale training

## Project Conventions

### Code Style
- **Naming**: snake_case for files and variables, PascalCase for classes (e.g., `WiFo`, `MaskStrategy`)
- **File Organization**: Modular design with clear separation of concerns
- **Imports**: Standard library imports first, then third-party, then local modules
- **Docstrings**: Google-style docstrings for functions and classes

### Architecture Patterns
- **Modular Design**: Clean separation across `src/` files (model, data, training, utilities)
- **Masked Autoencoder**: Self-supervised learning with encoder-decoder architecture
- **3D Tensor Processing**: Time-Frequency-Spatial dimensions handled as unified 3D tensors
- **Transformer-Based**: Multi-head attention for spatial-temporal-frequency feature extraction
- **Patch-Based Processing**: Fixed patch size (4,4,4) for 3D convolutions

**Source Structure:**
```
src/
├── main.py          # Entry point & configuration (argparse)
├── model.py         # MAE architecture (WiFo class)
├── embed.py         # Token & positional embeddings
├── mask_strategy.py # Three masking strategies
├── train.py         # Training loop
├── dataloader.py    # Data loading & preprocessing
└── utils.py         # Utility functions
```

### Testing Strategy
- Test scripts located in `doc/` directory for data flow verification
- 16 pre-training datasets (D1-D16) with 9k train / 1k val / 2k test split
- 17+ testing datasets for zero-shot evaluation
- Noise injection: 20 dB SNR during training/inference
- No fine-tuning required - zero-shot inference only

### Git Workflow
- Main branch: `main`
- Commit style: Conventional commits (e.g., `docs:`, `chore:`, `feat:`)
- Recent patterns: descriptive messages with clear purpose

## Domain Context

**Problem**: Wireless channel prediction in MIMO-OFDM systems. Traditional approaches require task-specific models for time-domain prediction (future CSI) and frequency-domain prediction (missing subcarriers).

**WiFo's Solution**: Single foundation model for all prediction tasks with zero-shot generalization to unseen datasets.

**Three Masking Strategies:**
1. **Random Masking (85%)**: Captures 3D structured features
2. **Temporal Masking (50%)**: Learns causal time relationships
3. **Frequency Masking (50%)**: Learns frequency band variations

**Data Format**:
- Input: Complex CSI tensor $H \in \mathbb{C}^{T \times K \times N}$
  - T: Time steps
  - K: Frequency subcarriers
  - N: Number of antennas
- Processed as real + imaginary parts: $\tilde{H} \in \mathbb{R}^{2 \times T \times K \times N}$

**Model Sizes**: tiny, little, small, base (different dimension configurations)

## Important Constraints

**Data Constraints:**
- Fixed patch size: (4,4,4) for 3D convolutions
- Each dataset: 12,000 samples fixed
- Input tensors must maintain real/imaginary splitting throughout
- Physics-compliant 3D sinusoidal positional embeddings required

**Training Constraints:**
- Cosine learning rate decay with warmup
- Loss computed only on masked patches (MSE)
- Multiple masking strategies applied randomly per batch
- No fine-tuning allowed (zero-shot only)

**Hardware Constraints:**
- Requires NVIDIA GPU with sufficient VRAM for 3D convolutions
- Large-scale training requires 256GB RAM

## External Dependencies

**Required Tools:**
- **QuaDRiGa**: Channel simulator for data generation (3GPP compliant)
- **HDF5**: Dataset format (.mat files)

**Pre-trained Resources:**
- HuggingFace model hub: `liuboxun/WiFo` for pre-trained weights

**Dataset Sources:**
- 16 pre-training datasets (D1-D16) with diverse wireless scenarios
- 17+ zero-shot evaluation datasets covering various configurations

**Citation Requirements:**
- Academic paper citation required for research use
- Apache 2.0 license for model weights