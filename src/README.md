# WiFo Source Code Structure

## Directory Layout

```
src/
├── config/         # Configuration management (YAML loading, CLI parsing, validation)
├── data/           # Data loading and preprocessing
├── models/         # WiFo model architecture
├── training/       # PyTorch Lightning module wrapper
├── utils/          # Utility functions
├── train.py        # Main training entry point
└── eval.py         # Evaluation/Inference entry point
```

## Configuration Management

WiFo uses a YAML-based configuration system with command-line override support using dot notation.

### Training with Configuration Files

#### Using a config file
```bash
# Use paper baseline configuration
python src/train.py --config configs/base_training.yaml

# Use quick test configuration (fast iteration)
python src/train.py --config configs/quick_test.yaml
```

#### Overriding config parameters with dot notation
```bash
# Override model size
python src/train.py --config configs/base.yaml --model.size tiny

# Override learning rate
python src/train.py --config configs/base.yaml --training.optimizer.lr 1e-3

# Override dataset
python src/train.py --config configs/base.yaml --data.dataset DS1

# Multiple overrides
python src/train.py --config configs/base.yaml --model.size tiny --training.batch_size 64
```

#### Without config file (backward compatible)
```bash
# Uses default configuration with CLI overrides
python src/train.py --data.dataset DS1 --model.size base
```

### Evaluation with Configuration Files

```bash
# Zero-shot evaluation using config
python src/eval.py --config configs/evaluation/zero_shot.yaml

# Override checkpoint path
python src/eval.py --config configs/evaluation/zero_shot.yaml --paths.checkpoint_path weights/lightning/wifo_tiny.ckpt

# Override dataset for specific test
python src/eval.py --config configs/evaluation/zero_shot.yaml --data.dataset D17
```

### Configuration Structure

The configuration is organized into logical sections:

- **experiment**: Metadata about the experiment (name, tags, note)
- **model**: Model architecture parameters (size, patch_size, t_patch_size, pos_emb, etc.)
- **training**: Training hyperparameters (optimizer, scheduler, masking, batch_size)
- **data**: Data loading and preprocessing (dataset, data_path, splits, num_workers)
- **trainer**: PyTorch Lightning Trainer settings (accelerator, devices, precision, max_epochs)
- **paths**: File paths for inputs and outputs (output_dir, log_dir, checkpoint_path)
- **system**: System settings (seed, device_id, process_name)

### Important: Dot Notation Required

All CLI parameter overrides MUST use dot notation matching the config structure. No shortcuts are allowed.

**Correct:**
- `--model.size tiny`
- `--training.optimizer.lr 1e-3`
- `--data.dataset DS1`
- `--paths.checkpoint_path weights/lightning/wifo_base.ckpt`

**Incorrect (will NOT work):**
- `--size tiny`
- `--lr 1e-3`
- `--dataset DS1`

### Default Configuration Files

- **configs/base_training.yaml** - Paper baseline configuration with all hyperparameters from the paper
- **configs/quick_test.yaml** - Fast iteration config (tiny model, minimal epochs)
- **configs/evaluation/zero_shot.yaml** - Zero-shot evaluation on all test datasets

### Config Persistence

Active configuration is automatically saved to the output directory for reproducibility:
- Training: `{output_dir}/config.yaml`
- Evaluation: `{output_dir}/config.yaml`

This includes all CLI overrides applied to the base config.

### Configuration Validation

The system validates configuration before running:
- Checks for required parameters
- Validates parameter values (e.g., model.size must be one of: tiny, little, small, base)
- Validates parameter types
- Shows clear error messages with valid options

## Weights

Pre-trained weights are available in `weights/lightning/`:
- `wifo_tiny.ckpt` - Tiny model (366K params)
- `wifo_little.ckpt` - Little model (1.4M params)
- `wifo_small.ckpt` - Small model (5.4M params)
- `wifo_base.ckpt` - Base model (21.5M params)

To convert legacy `.pkl` weights to Lightning format:
```bash
python scripts/convert_weights.py --size base --input weights/release/wifo_base.pkl --t_patch_size 4 --patch_size 4
```

## Legacy Code

Old training scripts have been moved to `tmp/reference/`:
- `train.py` - Old TrainLoop class (manual PyTorch training)
- `main.py` - Old entry point

## Module Import Examples

```python
# Import model
from src.models.model import WiFo_model

# Import Lightning module
from src.training.lightning_module import WiFoLightningModule

# Import DataModule
from src.data.data_module import WiFoDataModule

# Import configuration utilities
from src.config import load_config, save_config, parse_cli_overrides, validate_config
```

## Quick Reference

### Paper Hyperparameters (from configs/base_training.yaml)
- Optimizer: AdamW (lr=5e-4, betas=[0.9, 0.999], weight_decay=0.05)
- Scheduler: Cosine decay (warmup=5 epochs, total=200 epochs)
- Batch Size: 128
- Masking: Random=85%, Temporal=50%, Frequency=50%
- Model: Base (t_patch_size=4, patch_size=4)
- Datasets: D1*D2*...*D16 (all pre-training datasets)

### Common Workflows

**Quick development test:**
```bash
python src/train.py --config configs/quick_test.yaml
```

**Reproduce paper results:**
```bash
python src/train.py --config configs/base_training.yaml
```

**Evaluate on new dataset:**
```bash
python src/eval.py --config configs/evaluation/zero_shot.yaml --data.dataset D17
```

**Custom experiment:**
```bash
python src/train.py --config configs/base.yaml --model.size tiny --data.dataset DS1 --training.optimizer.lr 1e-3
```
